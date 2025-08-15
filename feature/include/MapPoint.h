
/**
 * @file MapPoint.h
 * @brief 3D point in the world map for PPG-SLAM system
 */

#pragma once

// ==================== SYSTEM INCLUDES ====================
#include <mutex>
#include <map>
#include <vector>
#include <chrono>

// ==================== THIRD-PARTY INCLUDES ====================
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>

// ==================== FORWARD DECLARATIONS ====================
class KeyFrame;
class MapEdge;
class MapColine;

// ==================== UTILITY FUNCTIONS ====================
float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

/**
 * @class MapPoint
 * @brief 3D point in the world map observed by multiple keyframes
 */
class MapPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// Constructor with 3D position and reference keyframe
    MapPoint(const Eigen::Vector3f &Pos, KeyFrame* pRefKF);

    // ==================== POSITION AND GEOMETRY ====================
    void SetWorldPos(const Eigen::Vector3f &Pos); ///< Set 3D world position
    Eigen::Vector3f GetWorldPos();                ///< Get 3D world position
    Eigen::Vector3f GetNormal();                  ///< Get mean viewing direction

    // ==================== OBSERVATIONS AND KEYFRAMES ====================
    KeyFrame* GetReferenceKeyFrame();             ///< Get the reference keyframe
    std::map<KeyFrame*, int> GetObservations();   ///< Get all observing keyframes and their feature indices
    int Observations();                           ///< Get number of observations
    void AddObservation(KeyFrame* pKF, int idx);  ///< Add an observation from a keyframe
    void EraseObservation(KeyFrame* pKF);         ///< Remove an observation
    int GetIndexInKeyFrame(KeyFrame* pKF);        ///< Get feature index in a keyframe
    bool IsInKeyFrame(KeyFrame* pKF);             ///< Check if observed in a keyframe
    int GetKeyFrameIdx(KeyFrame* pKF);            ///< Get keyframe index

    // ==================== STATE AND TRACKING ====================
    void SetBadFlag();                            ///< Mark as bad (to be removed)
    bool isBad();                                 ///< Check if marked as bad
    void Replace(MapPoint* pMP);                  ///< Replace this point with another
    MapPoint* GetReplaced();                      ///< Get the point that replaced this one

    /// Tracking statistics
    void IncreaseVisible(int n=1);                ///< Increase visible count
    void IncreaseFound(int n=1);                  ///< Increase found count
    float GetFoundRatio();                        ///< Get found/visible ratio
    inline int GetFound() { return mnFound; }     ///< Get found count

    // ==================== DESCRIPTORS AND GEOMETRY ====================
    void ComputeDistinctiveDescriptors();         ///< Compute best descriptor
    cv::Mat GetDescriptor();                      ///< Get best descriptor
    void UpdateNormalAndDepth();                  ///< Update normal and depth statistics
    float GetMinDistanceInvariance();             ///< Get min scale-invariant distance
    float GetMaxDistanceInvariance();             ///< Get max scale-invariant distance

    // ==================== GRAPH STRUCTURE ====================
    /// Edge operations
    void addEdge(MapEdge* pME);                   ///< Add an edge
    void removeEdge(MapEdge* pME);                ///< Remove an edge
    MapEdge* getEdge(MapPoint *pMP);              ///< Get edge to another MapPoint
    std::vector<MapEdge*> getEdges();             ///< Get all edges
    
    /// Collinearity operations
    MapColine* addColine(MapPoint* pMPs, MapPoint* pMPe, KeyFrame* pKF, float weight = -1); ///< Add a coline
    std::vector<MapColine*> removeColineOutliers(); ///< Remove outlier colines
    std::vector<MapColine*> getColinearity();     ///< Get all colines

public:
    // ==================== IDENTIFICATION ====================
    long unsigned int mnId;       ///< Unique ID
    static long unsigned int nNextId; ///< Global ID counter
    long int mnFirstKFid;         ///< First keyframe observing this point
    long int mnFirstFrame;        ///< First frame observing this point
    int nObs;                     ///< Number of observations

    // ==================== TRACKING VARIABLES ====================
    bool mbTrackInView;           ///< Is this point currently in view?
    float mTrackProjX, mTrackProjY, mTrackDepth, mTrackViewCos; ///< Tracking projection data
    long unsigned int mnTrackReferenceForFrame; ///< Reference frame for tracking
    long unsigned int mnTrackedbyFrame;         ///< Frame that tracked this point

    // ==================== OPTIMIZATION VARIABLES ====================
    long unsigned int mnBALocalForKF;      ///< Local BA keyframe ID
    long unsigned int mnFuseCandidateForKF; ///< Fusion candidate keyframe ID

    // --- Loop closing variables ---
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    Eigen::Vector3f mPosGBA;
    long unsigned int mnBAGlobalForKF;

    // --- Global settings ---
    static std::mutex mGlobalMutex;

    // --- Core data ---
    Eigen::Vector3f mWorldPos;                    ///< 3D position in world
    std::map<KeyFrame*, int> mObservations;       ///< Observing keyframes and feature indices
    Eigen::Vector3f mNormalVector;                ///< Mean viewing direction
    cv::Mat mDescriptor;                          ///< Best descriptor for matching
    KeyFrame* mpRefKF;                            ///< Reference keyframe
    int mnVisible;                                ///< Times observed
    int mnFound;                                  ///< Times found
    bool mbBad;                                   ///< Bad flag
    MapPoint* mpReplaced;                         ///< Replacement pointer
    float mfMinDepth, mfMaxDepth;                 ///< Scale invariance distances

    // --- Thread safety ---
    std::mutex mMutexPos;                         ///< Mutex for position
    std::mutex mMutexFeatures;                    ///< Mutex for features
    std::mutex mMutexMap;                         ///< Mutex for map

    // --- Graph structure ---
    std::mutex mMutexEdges;                       ///< Mutex for edges
    std::vector<MapEdge*> mvEdges;                ///< Connected edges
    std::vector<MapColine*> mvColines;            ///< Connected colines

    // --- Visualization ---
    std::chrono::steady_clock::time_point startTime; ///< Creation time (for visualizer)

    // --- Static mutex for point creation ---
    static std::mutex mMutexPointCreation;
};
