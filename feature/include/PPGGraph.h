#pragma once
#include <torch/torch.h>
#include <torch/script.h> 
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <list>
#include <mutex>

#include "MapPoint.h"
#include "KeyFrame.h"

class KeyFrame;
class MapPoint;
class Map;

class KeyEdge
{
public:
    KeyEdge() : startIdx(0),endIdx(0),isBad(true)
    {}
    KeyEdge(const unsigned int &i0, const unsigned int &i1) : startIdx(i0),endIdx(i1),isBad(false)
    {};

    unsigned int theOtherPid(const unsigned int pid) const
    {
        assert(pid==startIdx || pid==endIdx);
        if(pid==startIdx) 
            return endIdx;
        else 
            return startIdx;
    };

public:
    unsigned int startIdx,endIdx;
    bool isBad;
    float lscore; // 1.inlier check  2.refine point  2.erase redundant lines
    float length;
};

class MapEdge
{
public:
    // FIXME ::这样实例化：： createNewEdge();
    MapEdge(MapPoint* ps, MapPoint* pe, Map* pMap);
    MapPoint *theOtherPt(MapPoint* pMP);
    void addObservation(KeyFrame* pKF, unsigned int keyId);
    std::map<KeyFrame*, int> getObservations();
    void checkValid();
    bool isBad();
    Map* GetMap();
    static float viewCosTh;
    
public:
    unsigned long int mnId;
    static unsigned long int mnNextId;
    MapPoint *mpMPs, *mpMPe;
    Map* mpMap;
    bool mbBad, mbValid;

    unsigned long int mnBALocalForKF;
    // for visualizing
    unsigned int long trackedFrameId;

    std::chrono::steady_clock::time_point startTime;
private:
    std::map<KeyFrame*, int> mObservations;
    std::mutex mtxObs;
    cv::Mat mDesc;
};

class MapColine
{
public:
    MapColine(MapPoint* pMPs, MapPoint* pMPm, MapPoint* pMPe);
    void addObservation(KeyFrame* pKF, float weight);
    std::map<KeyFrame*, int> getObservations();
    float aveWeight();
    bool isBad();
public:
    unsigned long int mnId;
    static unsigned long int mnNextId;
    MapPoint *mpMPs, *mpMPm, *mpMPe; // start, end, and mid point
    bool mbBad, mbValid;
private:
    std::map<KeyFrame*, int> mObservations;
    std::mutex mtxObs;
    KeyFrame* mpFirstKF;
};