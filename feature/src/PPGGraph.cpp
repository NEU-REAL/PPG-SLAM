/**
 * @file PPGGraph.cpp
 * @brief Implementation of graph structures for PPG-SLAM system
 */

#include "PPGGraph.h"
#include <chrono>

// ==================== STATIC MEMBER DEFINITIONS ====================
unsigned long int MapEdge::mnNextId = 0;
unsigned long int MapColine::mnNextId = 0;
double MapEdge::viewCosTh = 0.9;

// ==================== KEY EDGE IMPLEMENTATION ====================

KeyEdge::KeyEdge() : startIdx(0), endIdx(0), isBad(true)
{
}

KeyEdge::KeyEdge(const unsigned int& i0, const unsigned int& i1) : 
    startIdx(i0), endIdx(i1), isBad(false)
{
}

unsigned int KeyEdge::theOtherPid(const unsigned int pid) const
{
    assert(pid == startIdx || pid == endIdx);
    if (pid == startIdx) 
        return endIdx;
    else 
        return startIdx;
}

// ==================== MAP EDGE IMPLEMENTATION ====================

MapEdge::MapEdge(MapPoint* ps, MapPoint* pe) : 
    mpMPs(ps), mpMPe(pe), mbBad(false), mbValid(true)
{
    mnBALocalForKF = 0;
    trackedFrameId = 0;
    mnId = mnNextId++;
    ps->addEdge(this);
    pe->addEdge(this);
    startTime = std::chrono::steady_clock::now();
}

MapPoint* MapEdge::theOtherPt(MapPoint* pMP)
{
    if (mpMPs == pMP)
        return mpMPe;
    if (mpMPe == pMP)
        return mpMPs;
    return nullptr;
}

void MapEdge::addObservation(KeyFrame* pKF, unsigned int keyId)
{
    std::unique_lock<std::mutex> lock(mtxObs);
    mObservations[pKF] = keyId;
}

std::map<KeyFrame*, int> MapEdge::getObservations()
{
    std::unique_lock<std::mutex> lock(mtxObs);
    return mObservations; 
}

void MapEdge::checkValid()
{
    auto obs = getObservations();
    if (obs.size() < 2)
    {
        mbValid = false;
        return;
    }
    
    // Check line direction - ensure viewing angle constraints are satisfied
    Eigen::Vector3f n1_ = mpMPs->GetNormal().normalized();
    Eigen::Vector3f n2_ = mpMPe->GetNormal().normalized();
    Eigen::Vector3f v_ = (mpMPs->GetWorldPos() - mpMPe->GetWorldPos()).normalized();
    float cosView1 = v_.dot(n1_);
    float cosView2 = v_.dot(n2_);
    
    if (std::fabs(cosView1) > MapEdge::viewCosTh || std::fabs(cosView2) > MapEdge::viewCosTh)
        mbValid = false;
    else
        mbValid = true;
}

bool MapEdge::isBad()
{
    std::unique_lock<std::mutex> lock(mtxObs);
    return (mbBad || mpMPs->isBad() || mpMPe->isBad());
}

// ==================== MAP COLINE IMPLEMENTATION ====================

MapColine::MapColine(MapPoint* pMPs, MapPoint* pMPm, MapPoint* pMPe) : 
    mpMPs(pMPs), mpMPm(pMPm), mpMPe(pMPe), mbBad(false), mbValid(false), mpFirstKF(nullptr)
{
    mnId = mnNextId++;
}

void MapColine::addObservation(KeyFrame* pKF, float weight)
{
    std::unique_lock<std::mutex> lock(mtxObs);
    if (mObservations.count(pKF))
        return;
        
    if (mObservations.empty())
        mpFirstKF = pKF;
        
    mObservations[pKF] = weight;
    
    // Check validity when we have at least 2 observations
    if (mObservations.size() < 2 || mbValid)
        return;
        
    Eigen::Vector3f pts = mpMPs->GetWorldPos();
    Eigen::Vector3f pte = mpMPe->GetWorldPos();
    Eigen::Vector3f posKF_ini = mpFirstKF->GetCameraCenter();
    Eigen::Vector3f posKF_cur = pKF->GetCameraCenter();
    
    // Check triangulation - if normals are not parallel, we have valid triangulation
    Eigen::Vector3f n1_ = (pts - pte).cross(posKF_ini).normalized();
    Eigen::Vector3f n2_ = (pts - pte).cross(posKF_cur).normalized();
    
    if (std::fabs(n1_.dot(n2_)) < 1.0f)
        mbValid = true;
}

float MapColine::aveWeight()
{
    std::unique_lock<std::mutex> lock(mtxObs);
    float ret = 0.0f;
    for (auto mmC : mObservations)
        ret += mmC.second;
    return ret;
}

std::map<KeyFrame*, int> MapColine::getObservations()
{
    std::unique_lock<std::mutex> lock(mtxObs);
    return mObservations;
}

bool MapColine::isBad()
{
    if (mpMPs->mpReplaced)
        mpMPs = mpMPs->mpReplaced;
    if (mpMPe->mpReplaced)
        mpMPe = mpMPe->mpReplaced;
    return mbBad || mpMPs->isBad() || mpMPm->isBad() || mpMPe->isBad();
}