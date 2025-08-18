/**
 * @file LocalMapping.cpp
 * @brief Local mapping module implementation for PPG-SLAM
 */

#include "LocalMapping.h"
#include "LoopClosing.h" 
#include "Matcher.h"
#include "Optimizer.h"
#include "Tracking.h"

#include <mutex>
#include <chrono>
#include <iostream>

void MSLocalMapping::Launch(Map *pMap)
{
    if (!pMap) {
        std::cerr << "ERROR: MSLocalMapping::Launch - Map pointer is null" << std::endl;
        return;
    }

    // Initialize control flags
    mbResetRequested = false;
    mbFinishRequested.store(false);
    mpMap = pMap;
    mbAbortBA = false;
    mbStopped = false;
    mbStopRequested = false;
    mbNotStop = false;
    mbLocalMappingIdle = true;
    
    // Clear existing data structures
    {
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        mlNewKeyFrames.clear();
    }
    
    // Start the local mapping thread
    try {
        mptLocalMapping = new std::thread(&MSLocalMapping::Run, this);
        std::cout << "Local Mapping thread launched successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to launch Local Mapping thread: " << e.what() << std::endl;
        throw;
    }
}

void MSLocalMapping::Run()
{
    std::cout << "Local Mapping thread started" << std::endl;
    
    while (!mbFinishRequested.load())
    {
        mbLocalMappingIdle = false;

        try {
            if (CheckNewKeyFrames())
            {
                ProcessNewKeyFrame();
                mbAbortBA = false;

                // Search for additional matches if no new keyframes waiting
                if (!CheckNewKeyFrames())
                {
                    SearchInNeighbors();
                } 

                // Perform local bundle adjustment
                if (mpMap->KeyFramesInMap() > 2)
                {
                    if (mpMap->isImuInitialized())
                    {
                        const bool bLarge = MSTracking::get().GetMatchesInliers() > 75;
                        Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpMap, 
                                                 bLarge, !mpMap->GetInertialBA());
                    }
                    else
                    {
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);
                    }
                }

                MSLoopClosing::get().InsertKeyFrame(mpCurrentKeyFrame);
            }
            else if (Stop())
            {
                while (isStopped() && !mbFinishRequested.load())
                {
                    std::this_thread::sleep_for(std::chrono::microseconds(1000));
                }
            }
            
            ResetIfRequested();
            mbLocalMappingIdle = true;

            if (mbFinishRequested.load())
                break;

            std::this_thread::sleep_for(std::chrono::microseconds(1000));
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR in Local Mapping main loop: " << e.what() << std::endl;
            mbLocalMappingIdle = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    std::cout << "Local Mapping thread finished" << std::endl;
}

void MSLocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    if (!pKF || pKF->isBad()) {
        std::cerr << "WARNING: MSLocalMapping::InsertKeyFrame - Invalid keyframe" << std::endl;
        return;
    }
    
    {
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        mlNewKeyFrames.push_back(pKF);
    }
    
    mbAbortBA = true;  // Abort current BA for immediate processing
}

bool MSLocalMapping::CheckNewKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    return !mlNewKeyFrames.empty();
}

void MSLocalMapping::ProcessNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        if (mlNewKeyFrames.empty()) {
            std::cerr << "WARNING: MSLocalMapping::ProcessNewKeyFrame - No keyframes in queue" << std::endl;
            return;
        }
        
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }
    
    if (!mpCurrentKeyFrame || mpCurrentKeyFrame->isBad()) {
        std::cerr << "WARNING: MSLocalMapping::ProcessNewKeyFrame - Invalid keyframe retrieved" << std::endl;
        return;
    }
}

void MSLocalMapping::EmptyQueue()
{
    while (CheckNewKeyFrames()) {
        ProcessNewKeyFrame();
    }
}

void MSLocalMapping::SearchInNeighbors()
{
    if (!mpCurrentKeyFrame) {
        std::cerr << "ERROR: MSLocalMapping::SearchInNeighbors - Current keyframe is null" << std::endl;
        return;
    }

    // Get primary covisible neighbors
    constexpr int nn = 30;
    const std::vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    std::vector<KeyFrame*> vpTargetKFs;
    vpTargetKFs.reserve(vpNeighKFs.size() * 2);
    
    // Filter valid primary neighbors
    for (const auto& pKFi : vpNeighKFs)
    {
        if (!pKFi || pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
            
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
    }

    // Extend to secondary neighbors
    const size_t initial_size = vpTargetKFs.size();
    for (size_t i = 0; i < initial_size && !mbAbortBA; ++i)
    {
        const std::vector<KeyFrame*> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        for (const auto& pKFi2 : vpSecondNeighKFs)
        {
            if (!pKFi2 || pKFi2->isBad() || 
                pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || 
                pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
                
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
        }
    }

    // Add temporal neighbors
    KeyFrame* pKFi = mpCurrentKeyFrame->mPrevKF;
    while (vpTargetKFs.size() < 20 && pKFi)
    {
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
        {
            pKFi = pKFi->mPrevKF;
            continue;
        }
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
        pKFi = pKFi->mPrevKF;
    }

    // Fuse map points: current -> targets
    Matcher matcher(mpMap->mpCamera);
    const std::vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    
    for (const auto& pKFi : vpTargetKFs)
    {
        if (pKFi && !pKFi->isBad()) {
            matcher.Fuse(pKFi, vpMapPointMatches);
        }
    }

    if (mbAbortBA) return;

    // Fuse map points: targets -> current
    std::vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

    for (const auto& pKFi : vpTargetKFs)
    {
        if (!pKFi || pKFi->isBad()) continue;
        
        const std::vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();
        for (const auto& pMP : vpMapPointsKFi)
        {
            if (!pMP || pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
                
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

    // Update map points after fusion
    const std::vector<MapPoint*> vpUpdatedMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (const auto& pMP : vpUpdatedMapPointMatches)
    {
        if (pMP && !pMP->isBad())
        {
            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();
        }
    }

    // Fuse map edges (PPG-SLAM specific)
    const std::vector<MapEdge*> vpMapEdgeMatches = mpCurrentKeyFrame->mvpMapEdges;
    std::vector<MapEdge*> vpFuseEdgeCandidates;
    vpFuseEdgeCandidates.reserve(vpTargetKFs.size() * vpMapEdgeMatches.size());
    
    // Collect edge candidates
    for (const auto& pKFi : vpTargetKFs)
    {
        if (!pKFi || pKFi->isBad()) continue;
        
        const std::vector<MapEdge*> vpMapEdgesKFi = pKFi->mvpMapEdges;
        for (const auto& pME : vpMapEdgesKFi)
        {
            if (!pME || pME->isBad()) continue;
            
            const MapPoint* pMPs = pME->mpMPs;
            const MapPoint* pMPe = pME->mpMPe;
            if (!pMPs || !pMPe) continue;
            
            // Check if current keyframe observes both endpoints
            bool bFoundStart = false, bFoundEnd = false;
            for (const auto& pMP : vpUpdatedMapPointMatches)
            {
                if (pMP == pMPs) bFoundStart = true;
                if (pMP == pMPe) bFoundEnd = true;
                if (bFoundStart && bFoundEnd) break;
            }
            
            if (bFoundStart && bFoundEnd)
            {
                vpFuseEdgeCandidates.push_back(pME);
            }
        }
    }

    // Perform edge fusion
    for (const auto& pMECandidate : vpFuseEdgeCandidates)
    {
        if (!pMECandidate || pMECandidate->isBad()) continue;
        
        const MapPoint* pMPs = pMECandidate->mpMPs;
        const MapPoint* pMPe = pMECandidate->mpMPe;
        if (!pMPs || !pMPe) continue;
        
        bool bEdgeExists = false;
        for (const auto& pME : vpMapEdgeMatches)
        {
            if (!pME || pME->isBad()) continue;
                
            if ((pME->mpMPs == pMPs && pME->mpMPe == pMPe) || 
               (pME->mpMPs == pMPe && pME->mpMPe == pMPs))
            {
                bEdgeExists = true;
                try {
                    const std::map<KeyFrame*, int> candidateObs = pMECandidate->getObservations();
                    for (const auto& obs : candidateObs)
                    {
                        if (obs.first && !obs.first->isBad()) {
                            pME->addObservation(obs.first, obs.second);
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "WARNING: Error merging edge observations: " << e.what() << std::endl;
                }
                break;
            }
        }
        
        if (!bEdgeExists)
        {
            try {
                const int edgeIdx = mpCurrentKeyFrame->FineEdgeIdx(pMPs->mnId, pMPe->mnId);
                if (edgeIdx >= 0 && edgeIdx < static_cast<int>(mpCurrentKeyFrame->mvpMapEdges.size()))
                {
                    if (!mpCurrentKeyFrame->mvpMapEdges[edgeIdx])
                    {
                        mpCurrentKeyFrame->mvpMapEdges[edgeIdx] = pMECandidate;
                        pMECandidate->addObservation(mpCurrentKeyFrame, edgeIdx);
                        
                        const_cast<MapPoint*>(pMPs)->addEdge(pMECandidate);
                        const_cast<MapPoint*>(pMPe)->addEdge(pMECandidate);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "WARNING: Error adding new edge observation: " << e.what() << std::endl;
            }
        }
    }
    
    // Validate edges
    for (const auto& pME : mpCurrentKeyFrame->mvpMapEdges)
    {
        if (pME && !pME->isBad())
        {
            try {
                pME->checkValid();
            } catch (const std::exception& e) {
                std::cerr << "WARNING: Error validating edge: " << e.what() << std::endl;
            }
        }
    }

    // Update covisibility graph
    try {
        mpCurrentKeyFrame->UpdateConnections();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to update keyframe connections: " << e.what() << std::endl;
    }
}

void MSLocalMapping::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopRequested = true;
    
    {
        std::unique_lock<std::mutex> lock2(mMutexNewKFs);
        mbAbortBA = true;
    }
}

bool MSLocalMapping::Stop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        std::cout << "Local Mapping STOP" << std::endl;
        return true;
    }
    return false;
}

bool MSLocalMapping::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

bool MSLocalMapping::stopRequested()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopRequested;
}

void MSLocalMapping::Release()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
    
    // Clear pending keyframes (don't delete - they may be referenced elsewhere)
    {
        std::unique_lock<std::mutex> lockKFs(mMutexNewKFs);
        mlNewKeyFrames.clear();
    }

    std::cout << "Local Mapping RELEASE" << std::endl;
}

bool MSLocalMapping::SetNotStop(bool flag)
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    if (flag && mbStopped) {
        return false;  // Cannot set not-stop if already stopped
    }
    mbNotStop = flag;
    return true;
}

void MSLocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void MSLocalMapping::RequestReset()
{
    {
        std::unique_lock<std::mutex> lock(mMutexReset);
        std::cout << "LM: Map reset received" << std::endl;
        mbResetRequested = true;
    }
    
    std::cout << "LM: Map reset, waiting..." << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    constexpr auto timeout = std::chrono::seconds(10);
    
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
        
        {
            std::unique_lock<std::mutex> lock2(mMutexReset);
            if (!mbResetRequested) break;
        }
        
        if (std::chrono::steady_clock::now() - start_time > timeout) {
            std::cerr << "WARNING: Reset timeout - forcing completion" << std::endl;
            std::unique_lock<std::mutex> lock2(mMutexReset);
            mbResetRequested = false;
            break;
        }
    }
    
    std::cout << "LM: Map reset, Done!!!" << std::endl;
}

void MSLocalMapping::ResetIfRequested()
{
    std::unique_lock<std::mutex> lock(mMutexReset);
    if (mbResetRequested)
    {
        std::cout << "LM: Resetting map in Local Mapping..." << std::endl;
        
        {
            std::unique_lock<std::mutex> lockKFs(mMutexNewKFs);
            mlNewKeyFrames.clear();
        }

        mbResetRequested = false;
        std::cout << "LM: End resetting Local Mapping..." << std::endl;
    }
}

void MSLocalMapping::RequestFinish()
{
    mbFinishRequested.store(true);
    
    auto start_time = std::chrono::steady_clock::now();
    constexpr auto timeout = std::chrono::seconds(5);
    
    while (mbFinishRequested.load())
    {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
        
        if (std::chrono::steady_clock::now() - start_time > timeout) {
            std::cerr << "WARNING: Finish timeout - forcing completion" << std::endl;
            break;
        }
    }
    
    std::cout << "Local mapping finished." << std::endl;
}