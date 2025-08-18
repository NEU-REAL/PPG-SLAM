/**
 * @file LoopClosing.cpp
 * @brief Implementation of the Loop Closing module for PPG-SLAM
 * 
 * This file implements a robust loop detection and closure system that combines
 * visual place recognition with geometric verification to detect when the robot
 * revisits previously mapped areas and correct accumulated drift.
 */

#include "LoopClosing.h"

// Project includes
#include "Sim3Solver.h"
#include "Optimizer.h"
#include "Matcher.h"
#include "G2oVertex.h"
#include "G2oEdge.h"

// Standard library includes
#include <mutex>
#include <thread>

using namespace std;

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * @brief Convert g2o::Sim3 to Sophus Sim3f format
 * @param S Input g2o Sim3 transformation
 * @return Equivalent Sophus Sim3f transformation
 * 
 * Helper function to convert between different Sim3 representations
 * used by different parts of the system.
 */
Sim3f toSophus(const g2o::Sim3& S) {
    SO3f rotation_so3(S.rotation().matrix().cast<float>());
    return Sim3f(rotation_so3, S.translation().cast<float>(), (float)S.scale());
}

// =============================================================================
// PUBLIC INTERFACE IMPLEMENTATION
// =============================================================================

void MSLoopClosing::Launch(Map *pMap, const bool bActiveLC)
{
    // Initialize system state
    mbResetRequested = false;
    mbFinishRequested = false;
    mpMap = pMap;
    
    // Initialize Global Bundle Adjustment state
    mbRunningGBA = false;
    mbStopGBA = false;
    mpThreadGBA = nullptr;
    mnFullBAIdx = 0;
    
    // Initialize loop detection state
    mnLoopNumCoincidences = 0;
    mbLoopDetected = false;
    mnLoopNumNotFound = 0;
    mbActiveLC = bActiveLC;
    
    // Initialize keyframe pointers
    mpLastCurrentKF = static_cast<KeyFrame*>(nullptr);
    mpCurrentKF = static_cast<KeyFrame*>(nullptr);
    
    // Launch main processing thread
    mptLoopClosing = new thread(&MSLoopClosing::Run, this);
}

void MSLoopClosing::Run()
{
    while(true)
    {
        // Check for new keyframes to process
        if(CheckNewKeyFrames())
        {
            // Attempt to detect loop closure candidates
            bool bLoopCandidateFound = NewDetectCommonRegions();
            
            if(bLoopCandidateFound && mbLoopDetected)
            {
                bool bValidLoop = true;
                std::cout << "LOOP: *Loop detected" << std::endl;
                
                // Prepare for loop correction
                mg2oLoopScw = mg2oLoopSlw;
                SE3d Twc = mpCurrentKF->GetPoseInverse().cast<double>();
                g2o::Sim3 g2oTwc(Twc.unit_quaternion(), Twc.translation(), 1.0);
                g2o::Sim3 g2oSww_new = g2oTwc * mg2oLoopScw;
                
                // Validate rotation matrix for numerical stability
                Eigen::Matrix3d rotation_matrix = g2oSww_new.rotation().toRotationMatrix();
                bool bValidRotation = true;
                
                // Check for NaN/Inf values
                if (!rotation_matrix.allFinite()) {
                    bValidRotation = false;
                }
                
                // Validate trace bounds for LogSO3 computation
                double tr = rotation_matrix.trace();
                if (tr < -1.0 || tr > 3.0) {
                    bValidRotation = false;
                }
                
                // Compute rotation angles for validation
                Eigen::Vector3d phi;
                if (bValidRotation) {
                    phi = LogSO3(rotation_matrix);
                } else {
                    phi = Eigen::Vector3d::Zero();
                }
                
                cout << "LOOP: Loop validation - Rotation angles (rad): [" << phi(0) << ", " << phi(1) << ", " << phi(2) << "]" << endl;
                
                // Validate loop closure quality (small rotation angles)
                if (fabs(phi(0)) < 0.008f && fabs(phi(1)) < 0.008f && fabs(phi(2)) < 0.349f)
                {
                    // For inertial SLAM, enforce yaw-only correction to maintain IMU constraints
                    if (mpMap->GetInertialBA())
                    {
                        phi(0) = 0;  // Remove roll correction
                        phi(1) = 0;  // Remove pitch correction
                        g2oSww_new = g2o::Sim3(ExpSO3(phi), g2oSww_new.translation(), 1.0);
                        mg2oLoopScw = g2oTwc.inverse() * g2oSww_new;
                    }
                }
                else
                {
                    cout << "LOOP: Loop REJECTED - Rotation too large for reliable closure" << endl;
                    bValidLoop = false;
                }

                // Execute loop closure if validation passed
                if (bValidLoop) 
                {
                    mvpLoopMapPoints = mvpLoopMPs;
                    CorrectLoop();
                }

                // Reset detection state for next iteration
                mpLoopLastCurrentKF->SetErase();
                mpLoopMatchedKF->SetErase();
                mnLoopNumCoincidences = 0;
                mvpLoopMatchedMPs.clear();
                mvpLoopMPs.clear();
                mnLoopNumNotFound = 0;
                mbLoopDetected = false;
            }
            
            // Update tracking state
            mpLastCurrentKF = mpCurrentKF;
        }

        // Handle system reset requests
        ResetIfRequested();

        // Check for shutdown request
        if(mbFinishRequested)
        {
            mbFinishRequested = false;
            break;
        }

        // Small delay to prevent excessive CPU usage
        usleep(5000);
    }
}

void MSLoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    
    // Skip the initial keyframe (ID 0) as it has no loop closure potential
    if(pKF->mnId != 0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool MSLoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return (!mlpLoopKeyFrameQueue.empty());
}

// =============================================================================
// CORE LOOP DETECTION ALGORITHMS
// =============================================================================

bool MSLoopClosing::NewDetectCommonRegions()
{
    // Early exit if loop closing is disabled
    if(!mbActiveLC)
        return false;

    // Extract and prepare current keyframe for processing
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        
        // Prevent keyframe deletion during processing
        mpCurrentKF->SetNotErase();
    }

    // Skip loop detection for non-inertial systems
    if(!mpMap->GetInertialBA())
    {
        mpMap->AddKeyFrame(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Wait for sufficient map size before attempting loop detection
    if(mpMap->GetAllKeyFrames().size() < 12)
    {
        mpMap->AddKeyFrame(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // =============================================================================
    // LOOP DETECTION STATE VARIABLES
    // =============================================================================

    // Current detection state
    bool bLoopDetectedInKF = false;

    // Continue validation of previous detection sequence
    if(mnLoopNumCoincidences > 0)
    {        
        // Compute relative transformation between current and last keyframe
        SE3d mTcl = (mpCurrentKF->GetPose() * mpLoopLastCurrentKF->GetPoseInverse()).cast<double>();
        g2o::Sim3 gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0);
        g2o::Sim3 gScw = gScl * mg2oLoopSlw;
        
        int numProjMatches = 0;
        vector<MapPoint*> vpMatchedMPs;
        
        // Attempt to refine the Sim3 transformation
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpLoopMatchedKF, gScw, 
                                                           numProjMatches, mvpLoopMPs, vpMatchedMPs);
        
        if(bCommonRegion)
        {
            bLoopDetectedInKF = true;
            mnLoopNumCoincidences++;
            
            cout << "LOOP: Loop refinement successful - Coincidences: " << mnLoopNumCoincidences << ", Matches: " << numProjMatches << endl;
            
            // Update detection state
            mpLoopLastCurrentKF->SetErase();
            mpLoopLastCurrentKF = mpCurrentKF;
            mg2oLoopSlw = gScw;
            mvpLoopMatchedMPs = vpMatchedMPs;

            // Require multiple consecutive detections for robust loop closure
            mbLoopDetected = mnLoopNumCoincidences >= 3;
            mnLoopNumNotFound = 0;

            if(!mbLoopDetected)
            {
                // Loop closure detected and validated
                cout << "LOOP: PR: Loop detected with Refined Sim3" << endl;
            }
        }
        else
        {
            bLoopDetectedInKF = false;
            mnLoopNumNotFound++;
            
            // Reset detection if too many failures
            if(mnLoopNumNotFound >= 2)
            {
                mpLoopLastCurrentKF->SetErase();
                mpLoopMatchedKF->SetErase();
                mnLoopNumCoincidences = 0;
                mvpLoopMatchedMPs.clear();
                mvpLoopMPs.clear();
                mnLoopNumNotFound = 0;
            }
        }
    }

    // Early return if loop already confirmed
    if(mbLoopDetected)
    {
        mpMap->AddKeyFrame(mpCurrentKF);
        return true;
    }

    // Extract covisible keyframes for spatial consistency checking
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    
    // Get BoW candidates only if no geometric detection occurred
    vector<KeyFrame*> vpLoopBowCand;
    if(!bLoopDetectedInKF)
    {
        // Search for visually similar keyframes using BoW
        vpLoopBowCand = mpMap->DetectNBestCandidates(mpCurrentKF, 3);
        // if(vpLoopBowCand.size() >= 2)  // Only report when we have meaningful candidates
            // cout << "LOOP: BoW found " << vpLoopBowCand.size() << " promising candidates for KF " << mpCurrentKF->mnId << endl;
    }

    // Process BoW candidates if available
    if(!bLoopDetectedInKF && !vpLoopBowCand.empty())
    {
        mbLoopDetected = DetectCommonRegionsFromBoW(vpLoopBowCand, mpLoopMatchedKF, mpLoopLastCurrentKF, 
                                                   mg2oLoopSlw, mnLoopNumCoincidences, mvpLoopMPs, mvpLoopMatchedMPs);
        if(mbLoopDetected)
            cout << "LOOP: BoW loop detection SUCCESSFUL for KF " << mpCurrentKF->mnId << endl;
    }

    // Add keyframe to map and return detection result
    mpMap->AddKeyFrame(mpCurrentKF);
    if(mbLoopDetected)
        return true;

    // No loop detected - allow keyframe deletion
    mpCurrentKF->SetErase();
    return false;
}

bool MSLoopClosing::DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, 
                                                 int &nNumProjMatches, std::vector<MapPoint*> &vpMPs, 
                                                 std::vector<MapPoint*> &vpMatchedMPs)
{
    // Configuration parameters for refinement
    const int nProjMatches = 30;        // Minimum initial projection matches
    const int nProjOptMatches = 50;     // Minimum matches after optimization
    const int nProjMatchesRep = 100;    // Minimum matches for final validation
    
    // =============================================================================
    // STAGE 1: INITIAL PROJECTION MATCHING
    // =============================================================================
    
    set<MapPoint*> spAlreadyMatchedMPs;
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    if(nNumProjMatches >= nProjMatches)
    {
        // =============================================================================
        // STAGE 2: SIM3 OPTIMIZATION
        // =============================================================================
        
        // Prepare transformation matrices for optimization
        SE3d mTwm = pMatchedKF->GetPoseInverse().cast<double>();
        g2o::Sim3 gSwm(mTwm.unit_quaternion(), mTwm.translation(), 1.0);
        g2o::Sim3 gScm = gScw * gSwm;
        Eigen::Matrix<double, 7, 7> mHessian7x7;

        // Determine if scale should be fixed (depends on sensor configuration)
        bool bFixedScale = mpMap->GetInertialBA();  // Fixed scale for inertial systems
        
        // Optimize Sim3 transformation using matched points
        int numOptMatches = Optimizer::OptimizeSim3(mpMap, mpCurrentKF, pMatchedKF, vpMatchedMPs, 
                                                   gScm, 10, bFixedScale, mHessian7x7, true);

        if(numOptMatches > nProjOptMatches)
        {
            // =============================================================================
            // STAGE 3: FINAL VALIDATION WITH OPTIMIZED TRANSFORMATION
            // =============================================================================
            
            // Create final transformation (scale fixed to 1 for validation)
            g2o::Sim3 gScw_estimation(gScw.rotation(), gScw.translation(), 1.0);

            // Re-match using optimized transformation
            vector<MapPoint*> vpMatchedMP;
            vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(nullptr));

            nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw_estimation, 
                                                     spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
            
            // Final validation requires high number of matches
            if(nNumProjMatches >= nProjMatchesRep)
            {
                gScw = gScw_estimation;
                return true;
            }
        }
    }
    
    return false;
}

bool MSLoopClosing::DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand, KeyFrame* &pMatchedKF2, 
                                             KeyFrame* &pLastCurrentKF, g2o::Sim3 &g2oScw,
                                             int &nNumCoincidences, std::vector<MapPoint*> &vpMPs, 
                                             std::vector<MapPoint*> &vpMatchedMPs)
{
    // =============================================================================
    // CONFIGURATION PARAMETERS
    // =============================================================================
    const int nBoWMatches = 20;         // Minimum BoW feature matches
    const int nBoWInliers = 15;         // Minimum RANSAC inliers
    const int nSim3Inliers = 20;        // Minimum Sim3 optimization inliers
    const int nProjMatches = 50;        // Minimum projection matches
    const int nProjOptMatches = 80;     // Minimum optimized projection matches
    const int nNumCovisibles = 10;      // Number of covisible keyframes to consider

    // Get connected keyframes to avoid matching with nearby frames
    set<KeyFrame*> spConnectedKeyFrames = mpCurrentKF->GetConnectedKeyFrames();

    // Initialize matchers with different sensitivity thresholds
    Matcher matcherBoW(mpMap->mpCamera, 0.9);  // Stricter threshold for BoW
    Matcher matcher(mpMap->mpCamera, 0.75);    // Relaxed threshold for projection

    // =============================================================================
    // VARIABLES FOR BEST CANDIDATE SELECTION
    // =============================================================================
    KeyFrame* pBestMatchedKF = nullptr;
    int nBestMatchesReproj = 0;
    int nBestNumCoincidences = 0;
    g2o::Sim3 g2oBestScw;
    std::vector<MapPoint*> vpBestMapPoints;
    std::vector<MapPoint*> vpBestMatchedMapPoints;

    // =============================================================================
    // MAIN CANDIDATE PROCESSING LOOP
    // =============================================================================
    int index = 0;
    for(KeyFrame* pKFi : vpBowCand)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        // =============================================================================
        // STAGE 1: COVISIBILITY EXPANSION
        // =============================================================================
        
        // Get covisible keyframes to increase matching opportunities
        std::vector<KeyFrame*> vpCovKFi = pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles);
        if(vpCovKFi.empty())
        {
            vpCovKFi.push_back(pKFi);
        }
        else
        {
            // Ensure primary candidate is first in the list
            vpCovKFi.push_back(vpCovKFi[0]);
            vpCovKFi[0] = pKFi;
        }

        // =============================================================================
        // STAGE 2: SPATIAL CONSISTENCY CHECK
        // =============================================================================
        
        // Skip candidates that are spatially too close to current keyframe
        bool bAbortByNearKF = false;
        for(size_t j = 0; j < vpCovKFi.size(); ++j)
        {
            if(spConnectedKeyFrames.find(vpCovKFi[j]) != spConnectedKeyFrames.end())
            {
                bAbortByNearKF = true;
                break;
            }
        }
        if(bAbortByNearKF)
            continue;

        // =============================================================================
        // STAGE 3: BAG-OF-WORDS FEATURE MATCHING
        // =============================================================================
        
        std::vector<std::vector<MapPoint*>> vvpMatchedMPs;
        vvpMatchedMPs.resize(vpCovKFi.size());
        std::set<MapPoint*> spMatchedMPi;
        int numBoWMatches = 0;

        KeyFrame* pMostBoWMatchesKF = pKFi;
        int nMostBoWNumMatches = 0;

        // Initialize matching result containers
        std::vector<MapPoint*> vpMatchedPoints(mpCurrentKF->GetMapPointMatches().size(), 
                                              static_cast<MapPoint*>(nullptr));
        std::vector<KeyFrame*> vpKeyFrameMatchedMP(mpCurrentKF->GetMapPointMatches().size(), 
                                                  static_cast<KeyFrame*>(nullptr));

        // Find BoW matches with all covisible keyframes
        for(size_t j = 0; j < vpCovKFi.size(); ++j)
        {
            if(!vpCovKFi[j] || vpCovKFi[j]->isBad())
                continue;

            int num = matcherBoW.SearchByBoW(mpCurrentKF, vpCovKFi[j], vvpMatchedMPs[j]);
            if (num > nMostBoWNumMatches)
            {
                nMostBoWNumMatches = num;
                pMostBoWMatchesKF = vpCovKFi[j];
            }
        }

        // Consolidate matches from all covisible keyframes
        for(size_t j = 0; j < vpCovKFi.size(); ++j)
        {
            for(size_t k = 0; k < vvpMatchedMPs[j].size(); ++k)
            {
                MapPoint* pMPi_j = vvpMatchedMPs[j][k];
                if(!pMPi_j || pMPi_j->isBad())
                    continue;

                if(spMatchedMPi.find(pMPi_j) == spMatchedMPi.end())
                {
                    spMatchedMPi.insert(pMPi_j);
                    numBoWMatches++;
                    vpMatchedPoints[k] = pMPi_j;
                    vpKeyFrameMatchedMP[k] = vpCovKFi[j];
                }
            }
        }

        // =============================================================================
        // STAGE 4: GEOMETRIC VALIDATION WITH RANSAC
        // =============================================================================
        
        if(numBoWMatches >= nBoWMatches)
        {
            // cout << "LOOP: KF " << pKFi->mnId << " QUALIFIED with " << numBoWMatches << " BoW matches - testing geometry" << endl;
            
            // Determine scale constraint based on sensor configuration
            bool bFixedScale = mpMap->GetInertialBA();
            
            // Initialize Sim3 solver for geometric validation
            Sim3Solver solver(mpCurrentKF, pMostBoWMatchesKF, mpMap->mpCamera, 
                             vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP);
            solver.SetRansacParameters(0.99, nBoWInliers, 300);

            // RANSAC loop for robust Sim3 estimation
            bool bNoMore = false;
            vector<bool> vbInliers;
            int nInliers;
            bool bConverge = false;
            Eigen::Matrix4f mTcm;
            
            while(!bConverge && !bNoMore)
                mTcm = solver.iterate(20, bNoMore, vbInliers, nInliers, bConverge);

            if(bConverge)
            {
                cout << "LOOP: ✓ Sim3 RANSAC SUCCESS: " << nInliers << " inliers (KF " << pKFi->mnId << ")" << endl;
                
                // =============================================================================
                // STAGE 5: PROJECTION MATCHING WITH COVISIBLE POINTS
                // =============================================================================
                
                // Expand covisible region around best matched keyframe
                vpCovKFi.clear();
                vpCovKFi = pMostBoWMatchesKF->GetBestCovisibilityKeyFrames(nNumCovisibles);
                vpCovKFi.push_back(pMostBoWMatchesKF);

                // Collect all map points from the covisible region
                set<MapPoint*> spMapPoints;
                vector<MapPoint*> vpMapPoints;
                for(KeyFrame* pCovKFi : vpCovKFi)
                {
                    for(MapPoint* pCovMPij : pCovKFi->GetMapPointMatches())
                    {
                        if(!pCovMPij || pCovMPij->isBad())
                            continue;

                        if(spMapPoints.find(pCovMPij) == spMapPoints.end())
                        {
                            spMapPoints.insert(pCovMPij);
                            vpMapPoints.push_back(pCovMPij);
                        }
                    }
                }

                // Convert Sim3 solution to world coordinates
                g2o::Sim3 gScm(solver.GetEstimatedRotation().cast<double>(),
                              solver.GetEstimatedTranslation().cast<double>(), 
                              (double)solver.GetEstimatedScale());
                g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(),
                              pMostBoWMatchesKF->GetTranslation().cast<double>(), 1.0);
                g2o::Sim3 gScw = gScm * gSmw;
                Sim3f mScw = toSophus(gScw);

                // Project map points using estimated transformation
                vector<MapPoint*> vpMatchedMP;
                vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), 
                                  static_cast<MapPoint*>(nullptr));
                int numProjMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, 
                                                               vpMatchedMP, 8, 1.5);

                // =============================================================================
                // STAGE 6: SIM3 OPTIMIZATION AND FINAL VALIDATION
                // =============================================================================
                
                if(numProjMatches >= nProjMatches)
                {
                    cout << "LOOP: ✓ Projection SUCCESS: " << numProjMatches << " matches" << endl;
                    
                    // Optimize Sim3 transformation with all matches
                    Eigen::Matrix<double, 7, 7> mHessian7x7;
                    int numOptMatches = Optimizer::OptimizeSim3(mpMap, mpCurrentKF, pKFi, vpMatchedMP, 
                                                               gScm, 10, bFixedScale, mHessian7x7, true);

                    if(numOptMatches >= nSim3Inliers)
                    {
                        cout << "LOOP: ✓ Optimization SUCCESS: " << numOptMatches << " refined matches" << endl;
                        
                        // Final projection with optimized transformation
                        g2o::Sim3 gSmw_final(pMostBoWMatchesKF->GetRotation().cast<double>(),
                                             pMostBoWMatchesKF->GetTranslation().cast<double>(), 1.0);
                        g2o::Sim3 gScw_final = gScm * gSmw_final;
                        Sim3f mScw_final = toSophus(gScw_final);

                        vector<MapPoint*> vpMatchedMP_final;
                        vpMatchedMP_final.resize(mpCurrentKF->GetMapPointMatches().size(), 
                                                static_cast<MapPoint*>(nullptr));
                        int numProjOptMatches = matcher.SearchByProjection(mpCurrentKF, mScw_final, 
                                                                          vpMapPoints, vpMatchedMP_final, 5, 1.0);

                        if(numProjOptMatches >= nProjOptMatches)
                        {
                            cout << "LOOP: ✓ VALIDATION PASSED: " << numProjOptMatches << " final matches" << endl;
                            
                            // =============================================================================
                            // STAGE 7: SPATIAL CONSISTENCY WITH COVISIBLE KEYFRAMES
                            // =============================================================================
                            
                            int nNumKFs = 0;
                            vector<KeyFrame*> vpCurrentCovKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(nNumCovisibles);

                            // Validate transformation consistency with neighboring keyframes
                            for(size_t j = 0; j < vpCurrentCovKFs.size() && nNumKFs < 3; ++j)
                            {
                                KeyFrame* pKFj = vpCurrentCovKFs[j];
                                SE3d mTjc = (pKFj->GetPose() * mpCurrentKF->GetPoseInverse()).cast<double>();
                                g2o::Sim3 gSjc(mTjc.unit_quaternion(), mTjc.translation(), 1.0);
                                g2o::Sim3 gSjw = gSjc * gScw_final;
                                
                                int numProjMatches_j = 0;
                                vector<MapPoint*> vpMatchedMPs_j;
                                bool bValid = DetectCommonRegionsFromLastKF(pKFj, pMostBoWMatchesKF, gSjw,
                                                                           numProjMatches_j, vpMapPoints, vpMatchedMPs_j);
                                if(bValid)
                                    nNumKFs++;
                            }

                            // Check keyframe threshold
                            if(nNumKFs < 3)
                            {
                                // Insufficient keyframes for reliable matching
                                continue;
                            }

                            // Update best candidate if this one is better
                            if(nBestMatchesReproj < numProjOptMatches)
                            {
                                cout << "LOOP: 🏆 NEW CHAMPION: KF " << pMostBoWMatchesKF->mnId 
                                     << " (" << numProjOptMatches << " matches, " << nNumKFs << " spatial)" << endl;
                                
                                nBestMatchesReproj = numProjOptMatches;
                                nBestNumCoincidences = nNumKFs;
                                pBestMatchedKF = pMostBoWMatchesKF;
                                g2oBestScw = gScw_final;
                                vpBestMapPoints = vpMapPoints;
                                vpBestMatchedMapPoints = vpMatchedMP_final;
                            }
                        }
                    }
                }
            }
        }
        index++;
    }

    // =============================================================================
    // FINAL RESULT PROCESSING
    // =============================================================================
    
    if(nBestMatchesReproj > 0)
    {
        cout << "LOOP: 🎯 LOOP CONFIRMED: KF " << pBestMatchedKF->mnId 
             << " selected with " << nBestMatchesReproj << " matches" << endl;
             
        pLastCurrentKF = mpCurrentKF;
        nNumCoincidences = nBestNumCoincidences;
        pMatchedKF2 = pBestMatchedKF;
        pMatchedKF2->SetNotErase();
        g2oScw = g2oBestScw;
        vpMPs = vpBestMapPoints;
        vpMatchedMPs = vpBestMatchedMapPoints;

        // Require at least 3 spatial consistency confirmations
        return nNumCoincidences >= 3;
    }

    return false;
}

bool MSLoopClosing::DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, 
                                               int &nNumProjMatches, std::vector<MapPoint*> &vpMPs, 
                                               std::vector<MapPoint*> &vpMatchedMPs)
{
    const int nProjMatches = 30;  // Minimum projection matches required
    
    // Use existing matched points to avoid double counting
    set<MapPoint*> spAlreadyMatchedMPs(vpMatchedMPs.begin(), vpMatchedMPs.end());
    
    // Find new matches by projection
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    return (nNumProjMatches >= nProjMatches);
}

int MSLoopClosing::FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw, g2o::Sim3 &g2oScw,
                                         std::set<MapPoint*> &spMatchedMPinOrigin, std::vector<MapPoint*> &vpMapPoints,
                                         std::vector<MapPoint*> &vpMatchedMapPoints)
{
    const int nNumCovisibles = 10;
    
    // =============================================================================
    // STAGE 1: EXPAND COVISIBLE REGION
    // =============================================================================
    
    // Get covisible keyframes around matched keyframe
    vector<KeyFrame*> vpCovKFm = pMatchedKFw->GetBestCovisibilityKeyFrames(nNumCovisibles);
    int nInitialCov = vpCovKFm.size();
    vpCovKFm.push_back(pMatchedKFw);
    set<KeyFrame*> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end());
    
    // Get covisible keyframes of current keyframe to avoid conflicts
    set<KeyFrame*> spCurrentCovisibles = pCurrentKF->GetConnectedKeyFrames();
    
    // Expand the search region if insufficient covisible keyframes
    if(nInitialCov < nNumCovisibles)
    {
        for(int i = 0; i < nInitialCov; ++i)
        {
            vector<KeyFrame*> vpKFs = vpCovKFm[i]->GetBestCovisibilityKeyFrames(nNumCovisibles);
            int nInserted = 0;
            size_t j = 0;
            
            while(j < vpKFs.size() && nInserted < nNumCovisibles)
            {
                // Add keyframe if not already included and not connected to current
                if(spCheckKFs.find(vpKFs[j]) == spCheckKFs.end() && 
                   spCurrentCovisibles.find(vpKFs[j]) == spCurrentCovisibles.end())
                {
                    spCheckKFs.insert(vpKFs[j]);
                    ++nInserted;
                }
                ++j;
            }
            vpCovKFm.insert(vpCovKFm.end(), vpKFs.begin(), vpKFs.end());
        }
    }
    
    // =============================================================================
    // STAGE 2: COLLECT MAP POINTS FROM COVISIBLE REGION
    // =============================================================================
    
    set<MapPoint*> spMapPoints;
    vpMapPoints.clear();
    vpMatchedMapPoints.clear();
    
    // Gather all unique map points from covisible keyframes
    for(KeyFrame* pKFi : vpCovKFm)
    {
        for(MapPoint* pMPij : pKFi->GetMapPointMatches())
        {
            if(!pMPij || pMPij->isBad())
                continue;

            if(spMapPoints.find(pMPij) == spMapPoints.end())
            {
                spMapPoints.insert(pMPij);
                vpMapPoints.push_back(pMPij);
            }
        }
    }

    // =============================================================================
    // STAGE 3: PROJECT AND MATCH MAP POINTS
    // =============================================================================
    
    // Convert g2o Sim3 to Sophus format for projection
    Sim3f mScw = toSophus(g2oScw);
    
    // Initialize matcher with appropriate parameters
    Matcher matcher(mpMap->mpCamera, 0.9);

    // Prepare output container
    vpMatchedMapPoints.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(nullptr));
    
    // Perform projection matching with restrictive search radius
    int num_matches = matcher.SearchByProjection(pCurrentKF, mScw, vpMapPoints, vpMatchedMapPoints, 3, 1.5);

    return num_matches;
}

// =============================================================================
// LOOP CORRECTION IMPLEMENTATION
// =============================================================================

void MSLoopClosing::CorrectLoop()
{
    // =============================================================================
    // STAGE 1: SYSTEM PREPARATION FOR LOOP CORRECTION
    // =============================================================================
    
    std::cout << "LOOP: Executing loop closure correction..." << std::endl;

    // Stop local mapping to prevent conflicts during correction
    MSLocalMapping::get().RequestStop();
    MSLocalMapping::get().EmptyQueue();  // Process remaining keyframes in queue

    // Abort any running Global Bundle Adjustment
    if(isRunningGBA())
    {
        cout << "LOOP: Stopping Global Bundle Adjustment...";
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;
        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
            mpThreadGBA = nullptr;
        }
        cout << "LOOP:  Done!" << endl;
    }

    // Wait for local mapping to completely stop
    while(!MSLocalMapping::get().isStopped())
    {
        usleep(1000);
    }

    // =============================================================================
    // STAGE 2: COMPUTE POSE CORRECTIONS
    // =============================================================================
    
    // Update current keyframe connections
    mpCurrentKF->UpdateConnections();

    // Collect all keyframes that need correction (current + covisible)
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    std::cout << "LOOP: Correcting " << mvpCurrentConnectedKFs.size() << " connected keyframes" << std::endl;

    // Prepare transformation containers
    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    
    // Set correction for current keyframe (from loop detection)
    CorrectedSim3[mpCurrentKF] = mg2oLoopScw;
    
    // Store original pose for reference
    SE3f Twc = mpCurrentKF->GetPoseInverse();
    SE3f Tcw = mpCurrentKF->GetPose();
    g2o::Sim3 g2oScw(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>(), 1.0);
    NonCorrectedSim3[mpCurrentKF] = g2oScw;

    // Apply correction to current keyframe
    SE3d correctedTcw(mg2oLoopScw.rotation(), mg2oLoopScw.translation() / mg2oLoopScw.scale());
    mpCurrentKF->SetPose(correctedTcw.cast<float>());

    Map* pLoopMap = mpMap;

    // =============================================================================
    // STAGE 3: PROPAGATE CORRECTIONS TO CONNECTED KEYFRAMES
    // =============================================================================
    
    {
        // Acquire map mutex for thread-safe updates
        unique_lock<mutex> lock(pLoopMap->mMutexMapUpdate);
        const bool bImuInit = pLoopMap->isImuInitialized();

        // Propagate pose corrections to all connected keyframes
        for(KeyFrame* pKFi : mvpCurrentConnectedKFs)
        {
            if(pKFi != mpCurrentKF)
            {
                // Compute relative transformation and apply loop correction
                SE3f Tiw = pKFi->GetPose();
                SE3d Tic = (Tiw * Twc).cast<double>();
                g2o::Sim3 g2oSic(Tic.unit_quaternion(), Tic.translation(), 1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oLoopScw;
                
                // Store corrected pose
                CorrectedSim3[pKFi] = g2oCorrectedSiw;

                // Apply pose correction
                SE3d correctedTiw(g2oCorrectedSiw.rotation(), 
                                 g2oCorrectedSiw.translation() / g2oCorrectedSiw.scale());
                pKFi->SetPose(correctedTiw.cast<float>());

                // Store original pose for map point correction
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion().cast<double>(), Tiw.translation().cast<double>(), 1.0);
                NonCorrectedSim3[pKFi] = g2oSiw;
            }  
        }

        // =============================================================================
        // STAGE 4: CORRECT MAP POINT POSITIONS
        // =============================================================================
        
        // Transform all map points observed by corrected keyframes
        for(auto& kf_pose_pair : CorrectedSim3)
        {
            KeyFrame* pKFi = kf_pose_pair.first;
            g2o::Sim3 g2oCorrectedSiw = kf_pose_pair.second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();
            g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(MapPoint* pMPi : vpMPsi)
            {
                if(!pMPi || pMPi->isBad())
                    continue;
                    
                // Skip if already corrected by this loop closure
                if(pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                    continue;

                // Transform map point: old_world -> keyframe -> corrected_world
                Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(P3Dw));

                // Update map point position and tracking
                pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>());
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }
            
            // =============================================================================
            // STAGE 5: CORRECT VELOCITY FOR INERTIAL SYSTEMS
            // =============================================================================
            
            // Correct velocity based on orientation change for inertial SLAM
            if(bImuInit)
            {
                Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * g2oSiw.rotation()).cast<float>();
                pKFi->SetVelocity(Rcor * pKFi->GetVelocity());
            }

            // Update keyframe connections after pose correction
            pKFi->UpdateConnections();
        }
        
        // Notify map of structural changes
        mpMap->InfoMapChange();
    }
    
    // =============================================================================
    // STAGE 6: FUSE DUPLICATE MAP POINTS
    // =============================================================================
    
    std::cout << "LOOP: Fusing duplicate map points..." << std::endl;
    
    // Replace matched map points and fuse duplicates
    for(size_t i = 0; i < mvpLoopMatchedMPs.size(); i++)
    {
        if(mvpLoopMatchedMPs[i])
        {
            MapPoint* pLoopMP = mvpLoopMatchedMPs[i];
            MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
            
            if(pCurMP)
            {
                // Replace current map point with loop map point
                pCurMP->Replace(pLoopMP);
            }
            else
            {
                // Add new observation to current keyframe
                mpCurrentKF->AddMapPoint(pLoopMP, i);
                pLoopMP->AddObservation(mpCurrentKF, i);
                pLoopMP->ComputeDistinctiveDescriptors();
            }
        }
    }

    // Project and fuse map points from loop region into corrected keyframes
    SearchAndFuse(CorrectedSim3, mvpLoopMapPoints);

    // =============================================================================
    // STAGE 7: UPDATE COVISIBILITY GRAPH
    // =============================================================================
    
    // Detect new connections created by loop closure
    map<KeyFrame*, set<KeyFrame*>> LoopConnections;

    for(KeyFrame* pKFi : mvpCurrentConnectedKFs)
    {
        // Store previous connections
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections based on new map point observations
        pKFi->UpdateConnections();
        LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
        
        // Remove old connections to find only new loop connections
        for(KeyFrame* pPrevKF : vpPreviousNeighbors)
        {
            LoopConnections[pKFi].erase(pPrevKF);
        }
        
        // Remove connections within the current connected group
        for(KeyFrame* pConnKF : mvpCurrentConnectedKFs)
        {
            LoopConnections[pKFi].erase(pConnKF);
        }
    }

    // =============================================================================
    // STAGE 8: OPTIMIZE POSE GRAPH
    // =============================================================================
    
    bool bFixedScale = mpMap->GetInertialBA();  // Fixed scale for inertial systems
    
    std::cout << "LOOP: Optimizing essential graph..." << std::endl;
    
    // Choose optimization method based on system configuration
    if(pLoopMap->isImuInitialized())
    {
        // Use 4-DOF optimization for inertial systems (preserves gravity direction)
        Optimizer::OptimizeEssentialGraph4DoF(pLoopMap, mpLoopMatchedKF, mpCurrentKF, 
                                              NonCorrectedSim3, CorrectedSim3, LoopConnections);
    }
    else
    {
        // Use full 7-DOF optimization for visual-only systems
        std::cout << "LOOP: Loop -> Scale correction: " << mg2oLoopScw.scale() << std::endl;
        Optimizer::OptimizeEssentialGraph(pLoopMap, mpLoopMatchedKF, mpCurrentKF, 
                                         NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixedScale);
    }

    // =============================================================================
    // STAGE 9: ADD LOOP EDGE AND LAUNCH GLOBAL BA
    // =============================================================================
    
    // Add bidirectional loop edge to the pose graph
    mpLoopMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpLoopMatchedKF);

    // Launch Global Bundle Adjustment for final refinement (if map is small enough)
    if(!pLoopMap->isImuInitialized() || (pLoopMap->KeyFramesInMap() < 200))
    {
        mbRunningGBA = true;
        mbStopGBA = false;
        mpThreadGBA = new thread(&MSLoopClosing::RunGlobalBundleAdjustment, this, pLoopMap, mpCurrentKF->mnId);
    }

    // Release local mapping to resume normal operation
    MSLocalMapping::get().Release();
    
    std::cout << "LOOP: Loop closure correction completed successfully!" << std::endl;
}

void MSLoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, std::vector<MapPoint*> &vpMapPoints)
{
    Matcher matcher(mpMap->mpCamera, 0.8);
    int total_replaces = 0;

    std::cout << "LOOP: Fusing " << vpMapPoints.size() << " map points across " 
              << CorrectedPosesMap.size() << " keyframes" << std::endl;

    // Process each corrected keyframe
    for(const auto& kf_pose_pair : CorrectedPosesMap)
    {
        KeyFrame* pKFi = kf_pose_pair.first;
        Map* pMap = mpMap;
        g2o::Sim3 g2oScw = kf_pose_pair.second;
        Sim3f Scw = toSophus(g2oScw);

        // Find replacement candidates through projection
        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(), static_cast<MapPoint*>(nullptr));
        matcher.Fuse(pKFi, Scw, vpMapPoints, 4, vpReplacePoints);

        // Apply replacements under map mutex
        {
            unique_lock<mutex> lock(pMap->mMutexMapUpdate);
            int num_replaces = 0;
            
            for(size_t i = 0; i < vpMapPoints.size(); i++)
            {
                MapPoint* pRep = vpReplacePoints[i];
                if(pRep)
                {
                    pRep->Replace(vpMapPoints[i]);
                    num_replaces++;
                }
            }
            total_replaces += num_replaces;
        }
    }
    
    std::cout << "LOOP: Fused " << total_replaces << " duplicate map points" << std::endl;
}

// =============================================================================
// SYSTEM CONTROL FUNCTIONS
// =============================================================================

void MSLoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    // Wait for reset to complete
    while(true)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(5000);
    }
}

void MSLoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        cout << "LOOP: Loop closing system reset requested..." << endl;
        mlpLoopKeyFrameQueue.clear();
        mbResetRequested = false;
        cout << "LOOP: Loop closing system reset completed." << endl;
    }
}

// =============================================================================
// GLOBAL BUNDLE ADJUSTMENT IMPLEMENTATION
// =============================================================================

void MSLoopClosing::RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF)
{  
    std::cout << "LOOP: Starting Global Bundle Adjustment for loop " << nLoopKF << std::endl;

    const bool bImuInit = pActiveMap->isImuInitialized();
    int idx = mnFullBAIdx;

    // =============================================================================
    // STAGE 1: EXECUTE APPROPRIATE BUNDLE ADJUSTMENT
    // =============================================================================
    
    // Choose optimization method based on sensor configuration
    if(!bImuInit)
    {
        cout << "LOOP: Executing Visual-only Global Bundle Adjustment..." << endl;
        // Visual-only bundle adjustment
        Optimizer::GlobalBundleAdjustment(pActiveMap, 10, nLoopKF, &mbStopGBA);
    }
    else
    {
        cout << "LOOP: Executing Full Inertial Bundle Adjustment with IMU constraints..." << endl;
        // Full inertial bundle adjustment (includes IMU constraints)
        Optimizer::FullInertialBA(pActiveMap, 7, nLoopKF, &mbStopGBA);
    }

    // =============================================================================
    // STAGE 2: PROPAGATE OPTIMIZATION RESULTS
    // =============================================================================
    
    {
        unique_lock<mutex> lock(mMutexGBA);
        
        // Check if this GBA is still valid (no newer one started)
        if(idx != mnFullBAIdx)
            return;

        // Check if system configuration changed during optimization
        if(!bImuInit && pActiveMap->isImuInitialized())
            return;

        // Only proceed if optimization completed successfully
        if(!mbStopGBA)
        {
            std::cout << "LOOP: Global Bundle Adjustment completed successfully" << std::endl;
            std::cout << "LOOP: Propagating optimized poses and map points..." << std::endl;

            // Stop local mapping during result propagation
            MSLocalMapping::get().RequestStop();
            while(!MSLocalMapping::get().isStopped())
            {
                usleep(1000);
            }

            // =============================================================================
            // STAGE 3: UPDATE KEYFRAME POSES
            // =============================================================================
            
            {
                unique_lock<mutex> lock(pActiveMap->mMutexMapUpdate);

                std::vector<KeyFrame*> vpAllKFs = pActiveMap->GetAllKeyFrames();
                KeyFrame* pOriginKF = pActiveMap->GetOriginKF();
                
                cout << "LOOP: Updating " << vpAllKFs.size() << " keyframes and map structure..." << endl;
                
                for(KeyFrame* pKF : vpAllKFs)
                {
                    if(!pKF || pKF->isBad())
                        continue;
                        
                    // Update keyframes that weren't included in the global optimization
                    if(pKF->mnBAGlobalForKF != nLoopKF)
                    {
                        // Propagate correction through relative transformation
                        pKF->mTcwGBA = pKF->GetPose() * pOriginKF->GetPoseInverse() * pKF->mTcwGBA;
                        
                        if(pKF->isVelocitySet())
                        {
                            pKF->mVwbGBA = pKF->mTcwGBA.so3().inverse() * pKF->GetPose().so3() * pKF->GetVelocity();
                        }
                        else
                        {
                            std::cerr << "LOOP: Warning: GBA velocity empty for KF " << pKF->mnId << std::endl;
                        }
                        
                        pKF->mnBAGlobalForKF = nLoopKF;
                        pKF->mBiasGBA = pKF->GetImuBias();
                    }
                    
                    // Store pose before applying GBA correction
                    pKF->mTcwBefGBA = pKF->GetPose();
                    
                    // Apply optimized pose
                    pKF->SetPose(pKF->mTcwGBA);
                    
                    // Update inertial data if available
                    if(pKF->bImu)
                    {
                        pKF->mVwbBefGBA = pKF->GetVelocity();
                        pKF->SetVelocity(pKF->mVwbGBA);
                        pKF->SetNewBias(pKF->mBiasGBA);
                    }
                    else if(bImuInit)
                    {
                        std::cerr << "LOOP: Warning: Expected inertial data for KF " << pKF->mnId << std::endl;
                    }
                }

                // =============================================================================
                // STAGE 4: UPDATE MAP POINT POSITIONS
                // =============================================================================
                
                const vector<MapPoint*> vpMPs = pActiveMap->GetAllMapPoints();

                for(MapPoint* pMP : vpMPs)
                {
                    if(pMP->isBad())
                        continue;

                    if(pMP->mnBAGlobalForKF == nLoopKF)
                    {
                        // Map point was included in optimization - use optimized position
                        pMP->SetWorldPos(pMP->mPosGBA);
                    }
                    else
                    {
                        // Map point was not included - update based on reference keyframe correction
                        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                        if(pRefKF->mnBAGlobalForKF != nLoopKF)
                            continue;

                        // Transform map point using keyframe correction
                        Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();
                        pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
                    }
                }

                // Notify system of map changes
                pActiveMap->InfoMapChange();
            }

            // Resume local mapping
            MSLocalMapping::get().Release();
            std::cout << "LOOP: Map update completed successfully!" << std::endl;
        }

        // Mark GBA as completed
        mbRunningGBA = false;
    }
}

void MSLoopClosing::RequestFinish()
{
    mbFinishRequested = true;
    
    // Wait for graceful shutdown
    while(mbFinishRequested == true)
        usleep(3000);
        
    std::cout << "LOOP: Loop closing system finished gracefully." << std::endl;
}
