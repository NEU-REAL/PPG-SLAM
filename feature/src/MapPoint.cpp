/**
 * @file MapPoint.cpp
 * @brief Implementation of MapPoint class for PPG-SLAM system
 */

#include "MapPoint.h"
#include "KeyFrame.h"
#include "PPGGraph.h"

// ==================== SYSTEM INCLUDES ====================
#include <mutex>

using namespace std;

// ==================== STATIC MEMBER DEFINITIONS ====================
long unsigned int MapPoint::nNextId = 0;
std::mutex MapPoint::mGlobalMutex;
std::mutex MapPoint::mMutexPointCreation;

// ==================== UTILITY FUNCTIONS ====================

float DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    assert(a.cols == b.cols);
    assert(a.isContinuous() && b.isContinuous());
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> des1(a.ptr<float>(), a.rows, a.cols);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> des2(b.ptr<float>(), b.rows, b.cols);
    return (des1 - des2).norm();
}

// ==================== CONSTRUCTORS ====================

MapPoint::MapPoint(const Eigen::Vector3f &Pos, KeyFrame *pRefKF):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnTrackedbyFrame(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDepth(0), mfMaxDepth(0)
{
    SetWorldPos(Pos);
    mNormalVector.setZero();
    mbTrackInView = false;
    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(MapPoint::mMutexPointCreation);
    mnId = nNextId++;
    startTime = chrono::steady_clock::now();
}

// ==================== POSITION OPERATIONS ====================

void MapPoint::SetWorldPos(const Eigen::Vector3f &Pos) 
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    mWorldPos = Pos;
}

Eigen::Vector3f MapPoint::GetWorldPos() 
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos;
}

Eigen::Vector3f MapPoint::GetNormal() 
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector;
}

// ==================== OBSERVATION OPERATIONS ====================


KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapPoint::AddObservation(KeyFrame* pKF, int idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mObservations[pKF] = idx;
    nObs++;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int index = mObservations[pKF];
            if(index != -1)
                nObs--;
            mObservations.erase(pKF);
            if(mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;
            // If only 2 observations or less, discard point
            if(nObs <= 2)
                bBad = true;
        }
    }
    if(bBad)
        SetBadFlag();
}


std::map<KeyFrame*, int> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

// ==================== MAPPOINT MANAGEMENT ====================

void MapPoint::SetBadFlag()
{
    map<KeyFrame*, int> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        obs = mObservations;
        mObservations.clear();
    }
    for(map<KeyFrame*, int>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        int index = mit->second;
        if(index != -1)
            pKF->EraseMapPointMatch(index);
    }
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId == this->mnId)
        return;

    {
        std::unique_lock<std::mutex> lock(mMutexEdges);
        for(MapColine* pMC : mvColines)
        {
            auto obs = pMC->getObservations();
            for(auto ob : obs)
                pMP->addColine(pMC->mpMPs, pMC->mpMPe, ob.first, ob.second);
        }
        // Note: MapEdge replacement logic is commented out
        // This may need to be implemented based on specific requirements
    }
    
    int nvisible, nfound;
    map<KeyFrame*, int> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs = mObservations;
        mObservations.clear();
        mbBad = true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for(map<KeyFrame*, int>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;
        int index = mit->second;
        if(!pMP->IsInKeyFrame(pKF))
        {
            if(index != -1)
            {
                pKF->ReplaceMapPointMatch(index, pMP);
                pMP->AddObservation(pKF, index);
            }
        }
        else
        {
            if(index != -1)
                pKF->EraseMapPointMatch(index);
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();
    pMP->UpdateNormalAndDepth();
}

bool MapPoint::isBad()
{
    unique_lock<mutex> lock1(mMutexFeatures, std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos, std::defer_lock);
    lock(lock1, lock2);

    return mbBad;
}

// ==================== STATISTICS OPERATIONS ====================

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}

void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound += n;
}

float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound) / mnVisible;
}

// ==================== DESCRIPTOR OPERATIONS ====================

void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*, int> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if (mbBad)
            return;
        observations = mObservations;
    }

    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for (map<KeyFrame*, int>::iterator mit = observations.begin(), 
         mend = observations.end(); mit != mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        if (!pKF->isBad())
        {
            int index = mit->second;
            if (index != -1)
                vDescriptors.push_back(pKF->mDescriptors.row(index));
        }
    }

    if (vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for (size_t i = 0; i < N; i++)
    {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; j++)
        {
            float distij = DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    float BestMedian = 1.0f;
    int BestIdx = 0;
    for (size_t i = 0; i < N; i++)
    {
        vector<float> vDists(Distances[i], Distances[i] + N);
        sort(vDists.begin(), vDists.end());
        float median = vDists[0.5 * (N - 1)];
        if (median < BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

// ==================== KEYFRAME RELATION OPERATIONS ====================

int MapPoint::GetIndexInKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

int MapPoint::GetKeyFrameIdx(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

// ==================== GEOMETRIC OPERATIONS ====================

void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*, int> observations;
    Eigen::Vector3f Pos;
    
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if (mbBad)
            return;
        if (mObservations.empty())
            return;
        observations = mObservations;
        Pos = mWorldPos;
    }

    float minDepth(1e9), maxDepth(0);
    int n = 0;
    Eigen::Vector3f normal(0, 0, 0);
    
    for (auto mit : observations)
    {
        KeyFrame* pKF = mit.first;
        Eigen::Vector3f Owi = pKF->GetCameraCenter();
        Eigen::Vector3f normali = Pos - Owi;
        float dist = normali.norm();
        normal += normali / dist;
        minDepth = dist < minDepth ? dist : minDepth;
        maxDepth = dist > maxDepth ? dist : maxDepth;
        n++;
    }

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMinDepth = minDepth;
        mfMaxDepth = maxDepth;
        mNormalVector = normal / (float)n;
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.5f * mfMinDepth;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 2.0f * mfMaxDepth;
}

// ==================== EDGE OPERATIONS ====================
void MapPoint::addEdge(MapEdge* pME)
{
    std::unique_lock<std::mutex> lock(mMutexEdges);
    mvEdges.push_back(pME);
}

void MapPoint::removeEdge(MapEdge* pME)
{
    std::unique_lock<std::mutex> lock(mMutexEdges);
    for (unsigned int lid = 0; lid < mvEdges.size(); lid++)
    {
        if (mvEdges[lid] != pME)
            continue;
        mvEdges[lid] = mvEdges.back();
        mvEdges.pop_back();
        break;
    }
}

MapEdge* MapPoint::getEdge(MapPoint* pMP)
{
    std::unique_lock<std::mutex> lock(mMutexEdges);
    for (MapEdge* pME : mvEdges)
    {
        if (pME->theOtherPt(this) == pMP)
            return pME;
    }
    return nullptr;
}

std::vector<MapEdge*> MapPoint::getEdges()
{
    std::unique_lock<std::mutex> lock(mMutexEdges);
    return mvEdges;
}

// ==================== COLLINEARITY OPERATIONS ====================

std::vector<MapColine*> MapPoint::removeColineOutliers()
{
    std::vector<MapColine*> vCLs = getColinearity();
    std::vector<MapColine*> ret;
    ret.reserve(vCLs.size());
    auto iter = vCLs.begin();
    
    while (iter != vCLs.end())
    {
        MapPoint* pMPs = (*iter)->mpMPs;
        MapPoint* pMPe = (*iter)->mpMPe;
        Eigen::Vector3f v1_ = (pMPs->GetWorldPos() - this->GetWorldPos()).normalized();
        Eigen::Vector3f v2_ = (this->GetWorldPos() - pMPe->GetWorldPos()).normalized();
        float angleDiff = v1_.dot(v2_);
        
        if (angleDiff < 0.90) // TODO: 参数优化一下
        {
            (*iter)->mbBad = true;
            iter = vCLs.erase(iter);
            ret.push_back(*iter);
        }
        else
            iter++;
    }
    
    std::unique_lock<std::mutex> lock(mMutexEdges);
    mvColines = vCLs;
    return ret;
}

MapColine* MapPoint::addColine(MapPoint* pMPs, MapPoint* pMPe, KeyFrame* pKF, float weight)
{
    if (pMPs->mpReplaced)
        pMPs = pMPs->mpReplaced;
    if (pMPe->mpReplaced)
        pMPe = pMPe->mpReplaced;
        
    int idx_m = this->GetIndexInKeyFrame(pKF);
    int idx_s = pMPs->GetIndexInKeyFrame(pKF);
    int idx_e = pMPe->GetIndexInKeyFrame(pKF);
    
    if (idx_m < 0 || idx_s < 0 || idx_e < 0)
        return nullptr;
        
    Eigen::Vector3f v1_ = pMPs->GetWorldPos() - this->GetWorldPos();
    Eigen::Vector3f v2_ = this->GetWorldPos() - pMPe->GetWorldPos();
    Eigen::Vector3f n_ = pKF->GetCameraCenter() - this->GetWorldPos();
    
    float dist1 = v1_.norm();
    float dist2 = v2_.norm();
    float distn = n_.norm();
    float distRatio = dist1 / dist2;
    float viewDegen1 = n_.dot(v1_) / distn / dist1;
    float viewDegen2 = n_.dot(v2_) / distn / dist2;
    
    if (distRatio < 0.2 || distRatio > 5 || 
        std::fabs(viewDegen1) > 0.996 || std::fabs(viewDegen2) > 0.996)
        return nullptr;
        
    if (weight < 0)
    {
        Eigen::Vector2f pm, ps, pe;
        ps = pKF->mvKeysUn[idx_s].mPosUn;
        pm << pKF->mvKeysUn[idx_m].mPosUn;
        pe << pKF->mvKeysUn[idx_e].mPosUn;
        float l1 = (ps - pm).norm();
        float l2 = (pm - pe).norm();
        weight = 2 * l1 * l2 / (l1 + l2);
    }
    
    std::unique_lock<std::mutex> lock(mMutexEdges);
    for (MapColine* pMC : mvColines)
    {
        bool bFound = false;
        if (pMC->mpMPs == pMPs && pMC->mpMPe == pMPe)
            bFound = true;
        if (pMC->mpMPs == pMPe && pMC->mpMPe == pMPs)
            bFound = true;
            
        if (bFound)
        {
            pMC->addObservation(pKF, weight);
            return nullptr;
        }
    }
    
    MapColine* pNewMC = new MapColine(pMPs, this, pMPe);
    pNewMC->addObservation(pKF, weight);
    mvColines.push_back(pNewMC);
    return pNewMC;
}

std::vector<MapColine*> MapPoint::getColinearity()
{
    std::unique_lock<std::mutex> lock(mMutexEdges);
    return mvColines;
}
