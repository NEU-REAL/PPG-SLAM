#pragma once

#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <mutex>
#include <map>

#include "KeyFrame.h"

class KeyFrame;
class MapEdge;
class MapColine;

float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

class MapPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapPoint(const Eigen::Vector3f &Pos, KeyFrame* pRefKF);

    void SetWorldPos(const Eigen::Vector3f &Pos);
    Eigen::Vector3f GetWorldPos();

    Eigen::Vector3f GetNormal();

    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*, int> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,int idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    int GetKeyFrameIdx(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();

    // byz
    void addEdge(MapEdge* pME);
    void removeEdge(MapEdge* pME);
    MapEdge* getEdge(MapPoint *pMP);
    std::vector<MapEdge*> getEdges();
    
    // void updateEdges();

    // void updateColines();

    MapColine* addColine(MapPoint* pMPs, MapPoint* pMPe, KeyFrame* pKF, float weight = -1);

    std::vector<MapColine*> removeColineOutliers();

    std::vector<MapColine*> getColinearity();

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Variables used by the tracking
    bool mbTrackInView;
    float mTrackProjX;
    float mTrackProjY;
    float mTrackDepth;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnTrackedbyFrame;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    Eigen::Vector3f mPosGBA;
    long unsigned int mnBAGlobalForKF;

    static std::mutex mGlobalMutex;
    static float COS_TH;
public:    

     // Position in absolute coordinates
     Eigen::Vector3f mWorldPos;

     // Keyframes observing the point and associated index in keyframe
     std::map<KeyFrame*, int> mObservations;

     // Mean viewing direction
     Eigen::Vector3f mNormalVector;

     // Best descriptor to fast matching
     cv::Mat mDescriptor;

     // Reference KeyFrame
     KeyFrame* mpRefKF;

     // Tracking counters
     int mnVisible;
     int mnFound;

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     MapPoint* mpReplaced;

     // Scale invariance distances
     float mfMinDepth, mfMaxDepth;

     // Mutex
     std::mutex mMutexPos;
     std::mutex mMutexFeatures;
     std::mutex mMutexMap;

    // byz
    std::mutex mMutexEdges;
    std::vector<MapEdge*> mvEdges;
    std::vector<MapColine*> mvColines;
    
    std::chrono::steady_clock::time_point startTime; // for visualizer

    static std::mutex mMutexPointCreation;
};
