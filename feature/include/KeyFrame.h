/**
 * @file KeyFrame.h
 * @brief KeyFrame class representing key frames in PPG-SLAM system
 * @details This class represents a key frame in the SLAM system, containing camera pose,
 *          IMU data, keypoints, descriptors, map point observations, and covisibility graph connections.
 *          Key frames are crucial nodes in the SLAM graph that maintain the map structure and
 *          enable loop closure detection and optimization.
 */

#pragma once

// ==================== SYSTEM INCLUDES ====================
#include <mutex>
#include <vector>
#include <map>
#include <set>

// ==================== THIRD-PARTY INCLUDES ====================
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include "DBoW3/DBoW3.h"

// ==================== LOCAL INCLUDES ====================
#include "MapPoint.h"
#include "PPGExtractor.h"
#include "IMU.h"
#include "PPGGraph.h"
#include "GeometricCamera.h"

// ==================== FORWARD DECLARATIONS ====================
class KeyPointEx;
class KeyEdge;
class MapPoint;
class MapEdge;

/**
 * @class KeyFrame
 * @brief Represents a key frame in the SLAM system
 * @details A KeyFrame is a selected frame that serves as a reference point in the SLAM system.
 *          It stores the camera pose, IMU measurements, extracted features, map point observations,
 *          and maintains connections with other key frames through the covisibility graph.
 *          Key frames are essential for:
 *          - Maintaining the sparse map representation
 *          - Enabling loop closure detection
 *          - Providing constraints for bundle adjustment
 *          - Supporting place recognition and relocalization
 */
class KeyFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // ==================== CONSTRUCTORS/DESTRUCTORS ====================
    
    /**
     * @brief Default constructor
     */
    KeyFrame();

    // ==================== POSE MANAGEMENT ====================
    
    /**
     * @brief Pose setter functions
     */
    void SetPose(const Sophus::SE3f &Tcw);          ///< Set camera pose (world to camera transformation)
    void SetVelocity(const Eigen::Vector3f &Vw_);   ///< Set camera velocity in world coordinates

    /**
     * @brief Camera pose getter functions
     */
    Sophus::SE3f GetPose();                         ///< Get camera pose (world to camera)
    Sophus::SE3f GetPoseInverse();                  ///< Get inverse camera pose (camera to world)
    Eigen::Vector3f GetCameraCenter();              ///< Get camera center position in world coordinates
    Eigen::Matrix3f GetRotation();                  ///< Get camera rotation matrix (world to camera)
    Eigen::Vector3f GetTranslation();               ///< Get camera translation vector (world to camera)

    /**
     * @brief IMU pose getter functions
     */
    Eigen::Vector3f GetImuPosition();               ///< Get IMU position in world coordinates
    Eigen::Matrix3f GetImuRotation();               ///< Get IMU rotation matrix
    Sophus::SE3f GetImuPose();                      ///< Get IMU pose transformation

    /**
     * @brief Velocity management functions
     */
    Eigen::Vector3f GetVelocity();                  ///< Get camera velocity in world coordinates
    bool isVelocitySet();                           ///< Check if velocity has been set

    // ==================== COVISIBILITY GRAPH ====================
    
    /**
     * @brief Connection management functions
     */
    void AddConnection(KeyFrame* pKF, const int &weight);       ///< Add covisibility connection
    void EraseConnection(KeyFrame* pKF);                        ///< Remove connection to keyframe
    void UpdateConnections(bool upParent=true);                 ///< Update connections based on shared map points
    void UpdateBestCovisibles();                                ///< Update best covisible keyframes list

    /**
     * @brief Covisible keyframe getter functions
     */
    std::set<KeyFrame *> GetConnectedKeyFrames();               ///< Get all connected keyframes
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();      ///< Get covisible keyframes as vector
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);     ///< Get N best covisible keyframes
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w); ///< Get keyframes with weight >= threshold

    /**
     * @brief Covisibility weight functions
     */
    int GetWeight(KeyFrame* pKF);                               ///< Get connection weight with keyframe

    // ==================== LOOP CLOSURE ====================
    
    /**
     * @brief Loop edge management functions
     */
    void AddLoopEdge(KeyFrame* pKF);                ///< Add loop edge connection
    std::set<KeyFrame*> GetLoopEdges();             ///< Get all loop edge connections

    // ==================== MAP POINT MANAGEMENT ====================
    
    /**
     * @brief Map point association functions
     */
    void AddMapPoint(MapPoint* pMP, const size_t &idx);         ///< Associate map point with keypoint
    void ReplaceMapPointMatch(const int &idx, MapPoint* pMP);   ///< Replace map point at keypoint index

    /**
     * @brief Map point removal functions
     */
    void EraseMapPointMatch(const int &idx);        ///< Remove map point association at index
    void EraseMapPointMatch(MapPoint* pMP);         ///< Remove all associations with map point

    /**
     * @brief Map point getter functions
     */
    std::set<MapPoint*> GetMapPoints();             ///< Get all associated map points
    std::vector<MapPoint*> GetMapPointMatches();    ///< Get map points as vector (with nullptrs)
    MapPoint* GetMapPoint(const size_t &idx);       ///< Get map point at keypoint index

    /**
     * @brief Map point statistics functions
     */
    int TrackedMapPoints(const int &minObs);        ///< Count tracked map points with min observations

    // ==================== FEATURE MANAGEMENT ====================
    
    /**
     * @brief Feature search and query functions
     */
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;  ///< Get keypoint indices within circular area

    // ==================== BAD FLAG MANAGEMENT ====================
    
    /**
     * @brief Keyframe lifecycle management functions
     */
    void SetNotErase();                             ///< Prevent keyframe from being erased
    void SetErase();                                ///< Allow keyframe to be erased
    void SetBadFlag();                              ///< Mark keyframe as bad
    bool isBad();                                   ///< Check if keyframe is marked as bad

    // ==================== IMU BIAS MANAGEMENT ====================
    
    /**
     * @brief IMU bias setter/getter functions
     */
    void SetNewBias(const IMU::Bias &b);            ///< Set new IMU bias values
    Eigen::Vector3f GetGyroBias();                  ///< Get gyroscope bias
    Eigen::Vector3f GetAccBias();                   ///< Get accelerometer bias  
    IMU::Bias GetImuBias();                         ///< Get complete IMU bias

    // ==================== GRAPH EDGE MANAGEMENT ====================
    
    /**
     * @brief Map edge management functions
     */
    void AddMapEdge(MapEdge* pME, const size_t &idx);           ///< Add map edge association
    MapEdge* GetMapEdge(int idx);                               ///< Get map edge at index
    int FineEdgeIdx(unsigned int p1_id, unsigned int p2_id);    ///< Find edge index by point IDs
    // ==================== PUBLIC MEMBER VARIABLES ====================
    // Note: The following variables are accessed from only 1 thread or never change (no mutex needed)

    // ==================== IDENTIFICATION ====================
    static long unsigned int nNextId;  ///< Global counter for generating unique key frame IDs
    long unsigned int mnId;            ///< Unique identifier for this key frame
    long unsigned int mnFrameId;       ///< Original frame ID from which this key frame was created

    double mTimeStamp;                 ///< Timestamp when this key frame was captured

    // ==================== TRACKING VARIABLES ====================
    long unsigned int mnTrackReferenceForFrame;  ///< Reference frame ID for tracking
    long unsigned int mnFuseTargetForKF;         ///< Target key frame ID for feature fusion

    // ==================== LOCAL MAPPING VARIABLES ====================
    long unsigned int mnBALocalForKF;  ///< Key frame ID for local bundle adjustment
    long unsigned int mnBAFixedForKF;   ///< Key frame ID when this was fixed in BA

    long unsigned int mnNumberOfOpt;    ///< Number of optimizations by bundle adjustment

    // ==================== KEYFRAME DATABASE VARIABLES ====================
    // Relocalization
    long unsigned int mnRelocQuery;     ///< Relocalization query ID
    int mnRelocWords;                  ///< Number of words for relocalization
    float mRelocScore;                 ///< Relocalization score

    // Place recognition
    long unsigned int mnPlaceRecognitionQuery;  ///< Place recognition query ID
    int mnPlaceRecognitionWords;              ///< Number of words for place recognition
    float mPlaceRecognitionScore;             ///< Place recognition score

    // ==================== GLOBAL BUNDLE ADJUSTMENT ====================
    Sophus::SE3f mTcwGBA;              ///< Pose after global bundle adjustment
    Sophus::SE3f mTcwBefGBA;           ///< Pose before global bundle adjustment
    Eigen::Vector3f mVwbGBA;           ///< Velocity after global bundle adjustment
    Eigen::Vector3f mVwbBefGBA;        ///< Velocity before global bundle adjustment
    IMU::Bias mBiasGBA;                ///< IMU bias after global bundle adjustment
    long unsigned int mnBAGlobalForKF;  ///< Global bundle adjustment key frame ID

    // ==================== FEATURE DATA ====================
    int N;                             ///< Number of keypoints in this key frame

    std::vector<KeyPointEx> mvKeys;    ///< Original keypoints (distorted)
    std::vector<KeyPointEx> mvKeysUn;  ///< Undistorted keypoints
    std::vector<KeyEdge> mvKeyEdges;   ///< Key edges for graph structure
    cv::Mat mDescriptors;              ///< Feature descriptors matrix

    // ==================== BAG-OF-WORDS ====================
    DBoW3::BowVector mBowVec;          ///< Bag-of-words vector representation
    DBoW3::FeatureVector mFeatVec;     ///< Feature vector for fast matching

    // ==================== RELATIVE POSE ====================
    Sophus::SE3f mTcp;                 ///< Pose relative to previous key frame

    // ==================== IMU INTEGRATION ====================
    KeyFrame* mPrevKF;                 ///< Pointer to previous key frame in temporal sequence
    KeyFrame* mNextKF;                 ///< Pointer to next key frame in temporal sequence

    IMU::Preintegrated* mpImuPreintegrated;  ///< Preintegrated IMU measurements from previous key frame
    
    // ==================== SENSORS ====================
    GeometricCamera *mpCamera;         ///< Pointer to camera calibration model
    IMU::Calib *mpImuCalib;           ///< Pointer to IMU calibration parameters
    bool bImu;                        ///< Flag indicating if IMU data is available

    // ==================== THREAD-SAFE VARIABLES ====================
    // Note: The following variables need to be accessed through a mutex to be thread safe
public:
    // ==================== POSE AND TRANSFORMATION ====================
    Sophus::SE3<float> mTcw;           ///< Camera pose: transformation from world to camera coordinates
    Eigen::Matrix3f mRcw;              ///< Rotation matrix from world to camera coordinates
    Sophus::SE3<float> mTwc;           ///< Inverse camera pose: transformation from camera to world coordinates
    Eigen::Matrix3f mRwc;              ///< Rotation matrix from camera to world coordinates

    // ==================== IMU STATE ====================
    Eigen::Vector3f mOwb;              ///< IMU position in world coordinates
    Eigen::Vector3f mVw;               ///< Camera velocity in world coordinates (for inertial SLAM)
    bool mbHasVelocity;                ///< Flag indicating if velocity has been set
    IMU::Bias mImuBias;                ///< Current IMU bias (accelerometer and gyroscope)

    // ==================== MAP ASSOCIATIONS ====================
    std::vector<MapPoint*> mvpMapPoints;  ///< Map points associated with keypoints (indexed by keypoint)
    std::vector<MapEdge*> mvpMapEdges;    ///< Map edges associated with key edges (indexed by key edge)

    // ==================== SPATIAL INDEXING ====================
    std::vector<std::vector<std::vector<std::size_t>>> mGrid;  ///< Grid structure for fast feature matching

    // ==================== COVISIBILITY GRAPH ====================
    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;       ///< Map of connected key frames and their weights
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;     ///< Connected key frames ordered by weight
    std::vector<int> mvOrderedWeights;                       ///< Weights corresponding to ordered connected key frames
    std::set<KeyFrame*> mspLoopEdges;                        ///< Set of key frames connected by loop edges

    // ==================== STATUS FLAGS ====================
    bool mbNotErase;                   ///< Flag preventing erasure of this key frame
    bool mbToBeErased;                 ///< Flag indicating this key frame is scheduled for erasure
    bool mbBad;                        ///< Flag indicating this key frame is marked as bad

    // ==================== THREAD SYNCHRONIZATION ====================
    std::mutex mMutexPose;             ///< Mutex for pose, velocity and bias data
    std::mutex mMutexConnections;      ///< Mutex for covisibility graph connections
    std::mutex mMutexFeatures;         ///< Mutex for feature-related data
    std::mutex mMutexMap;              ///< Mutex for map point associations

    // ==================== DEBUG DATA ====================
    cv::Mat srcMat;                    ///< Source image matrix for debugging purposes

};