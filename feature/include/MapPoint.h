
#pragma once

// Eigen for 3D vector and geometry operations
#include <Eigen/Geometry>
// OpenCV for matrix and descriptor operations
#include <opencv2/core/core.hpp>
#include <mutex>
#include <map>
#include <vector>
#include <chrono>

// Forward declarations to reduce compile dependencies
class KeyFrame;
class MapEdge;
class MapColine;

// Computes the distance between two feature descriptors
float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

/**
 * @brief MapPoint represents a 3D point in the world map observed by multiple keyframes.
 * It stores position, observations, descriptors, and connectivity (edges/colines) for SLAM.
 */
class MapPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Constructor
     * @param Pos 3D position in world coordinates
     * @param pRefKF Reference keyframe observing this point
     */
    MapPoint(const Eigen::Vector3f &Pos, KeyFrame* pRefKF);

    // --- Position and Geometry ---
    void SetWorldPos(const Eigen::Vector3f &Pos); ///< Set 3D world position
    Eigen::Vector3f GetWorldPos();                ///< Get 3D world position
    Eigen::Vector3f GetNormal();                  ///< Get mean viewing direction

    // --- Observations and KeyFrames ---
    KeyFrame* GetReferenceKeyFrame();             ///< Get the reference keyframe
    std::map<KeyFrame*, int> GetObservations();   ///< Get all observing keyframes and their feature indices
    int Observations();                           ///< Get number of observations
    void AddObservation(KeyFrame* pKF, int idx);  ///< Add an observation from a keyframe
    void EraseObservation(KeyFrame* pKF);         ///< Remove an observation
    int GetIndexInKeyFrame(KeyFrame* pKF);        ///< Get feature index in a keyframe
    bool IsInKeyFrame(KeyFrame* pKF);             ///< Check if observed in a keyframe
    int GetKeyFrameIdx(KeyFrame* pKF);            ///< Get keyframe index

    // --- State and Replacement ---
    void SetBadFlag();                            ///< Mark as bad (to be removed)
    bool isBad();                                 ///< Check if marked as bad
    void Replace(MapPoint* pMP);                  ///< Replace this point with another
    MapPoint* GetReplaced();                      ///< Get the point that replaced this one

    // --- Tracking statistics ---
    void IncreaseVisible(int n=1);                ///< Increase visible count
    void IncreaseFound(int n=1);                  ///< Increase found count
    float GetFoundRatio();                        ///< Get found/visible ratio
    inline int GetFound() { return mnFound; }     ///< Get found count

    // --- Descriptor and Normal update ---
    void ComputeDistinctiveDescriptors();         ///< Compute best descriptor
    cv::Mat GetDescriptor();                      ///< Get best descriptor
    void UpdateNormalAndDepth();                  ///< Update normal and depth statistics
    float GetMinDistanceInvariance();             ///< Get min scale-invariant distance
    float GetMaxDistanceInvariance();             ///< Get max scale-invariant distance

    // --- Graph structure: Edges and Colines ---
    void addEdge(MapEdge* pME);                   ///< Add an edge
    void removeEdge(MapEdge* pME);                ///< Remove an edge
    MapEdge* getEdge(MapPoint *pMP);              ///< Get edge to another MapPoint
    std::vector<MapEdge*> getEdges();             ///< Get all edges
    MapColine* addColine(MapPoint* pMPs, MapPoint* pMPe, KeyFrame* pKF, float weight = -1); ///< Add a coline
    std::vector<MapColine*> removeColineOutliers(); ///< Remove outlier colines
    std::vector<MapColine*> getColinearity();     ///< Get all colines

    // --- Member variables ---
public:
    // Unique ID for this MapPoint
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;         ///< First keyframe observing this point
    long int mnFirstFrame;        ///< First frame observing this point
    int nObs;                     ///< Number of observations

    // --- Tracking variables ---
    bool mbTrackInView;           ///< Is this point currently in view?
    float mTrackProjX, mTrackProjY, mTrackDepth, mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnTrackedbyFrame;

    // --- Local mapping variables ---
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

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
