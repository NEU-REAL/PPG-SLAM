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

// ==================== CONSTANTS ====================
#define FRAME_GRID_ROWS 48  ///< Number of grid rows for feature matching acceleration
#define FRAME_GRID_COLS 64  ///< Number of grid columns for feature matching acceleration

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
     * @brief Set the camera pose in world coordinates
     * @param Tcw Transformation matrix from world to camera coordinates
     */
    void SetPose(const Sophus::SE3f &Tcw);
    
    /**
     * @brief Set the velocity of the camera in world coordinates
     * @param Vw_ Velocity vector in world coordinates
     */
    void SetVelocity(const Eigen::Vector3f &Vw_);

    /**
     * @brief Get the camera pose (world to camera transformation)
     * @return SE3 transformation matrix from world to camera
     */
    Sophus::SE3f GetPose();

    /**
     * @brief Get the inverse camera pose (camera to world transformation)
     * @return SE3 transformation matrix from camera to world
     */
    Sophus::SE3f GetPoseInverse();
    
    /**
     * @brief Get the camera center position in world coordinates
     * @return 3D position vector of camera center
     */
    Eigen::Vector3f GetCameraCenter();

    /**
     * @brief Get the IMU position in world coordinates
     * @return 3D position vector of IMU sensor
     */
    Eigen::Vector3f GetImuPosition();
    
    /**
     * @brief Get the IMU rotation matrix
     * @return 3x3 rotation matrix of IMU sensor
     */
    Eigen::Matrix3f GetImuRotation();
    
    /**
     * @brief Get the IMU pose (transformation)
     * @return SE3 transformation of IMU sensor
     */
    Sophus::SE3f GetImuPose();
    
    /**
     * @brief Get the camera rotation matrix
     * @return 3x3 rotation matrix from world to camera
     */
    Eigen::Matrix3f GetRotation();
    
    /**
     * @brief Get the camera translation vector
     * @return 3D translation vector from world to camera
     */
    Eigen::Vector3f GetTranslation();
    
    /**
     * @brief Get the camera velocity in world coordinates
     * @return 3D velocity vector
     */
    Eigen::Vector3f GetVelocity();
    
    /**
     * @brief Check if velocity has been set
     * @return True if velocity is available, false otherwise
     */
    bool isVelocitySet();

    // ==================== COVISIBILITY GRAPH ====================
    
    /**
     * @brief Add a connection to another key frame with given weight
     * @param pKF Pointer to the connected key frame
     * @param weight Connection weight (number of shared map points)
     */
    void AddConnection(KeyFrame* pKF, const int &weight);
    
    /**
     * @brief Remove connection to another key frame
     * @param pKF Pointer to the key frame to disconnect
     */
    void EraseConnection(KeyFrame* pKF);

    /**
     * @brief Update connections based on shared map points
     * @param upParent Whether to update parent connections in spanning tree
     */
    void UpdateConnections(bool upParent=true);
    
    /**
     * @brief Update the list of best covisible key frames
     */
    void UpdateBestCovisibles();
    
    /**
     * @brief Get all connected key frames
     * @return Set of connected key frame pointers
     */
    std::set<KeyFrame *> GetConnectedKeyFrames();
    
    /**
     * @brief Get connected key frames as vector
     * @return Vector of connected key frame pointers
     */
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    
    /**
     * @brief Get N best covisible key frames
     * @param N Number of best key frames to return
     * @return Vector of N best covisible key frames
     */
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    
    /**
     * @brief Get key frames with covisibility weight above threshold
     * @param w Minimum weight threshold
     * @return Vector of key frames with weight >= w
     */
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    
    /**
     * @brief Get covisibility weight with another key frame
     * @param pKF Pointer to the other key frame
     * @return Connection weight (number of shared map points)
     */
    int GetWeight(KeyFrame* pKF);

    // ==================== LOOP CLOSURE ====================
    
    /**
     * @brief Add a loop edge connection
     * @param pKF Pointer to key frame forming loop closure
     */
    void AddLoopEdge(KeyFrame* pKF);
    
    /**
     * @brief Get all loop edge connections
     * @return Set of key frames connected by loop edges
     */
    std::set<KeyFrame*> GetLoopEdges();

    // ==================== MAP POINT MANAGEMENT ====================
    
    /**
     * @brief Associate a map point with a keypoint
     * @param pMP Pointer to the map point
     * @param idx Index of the corresponding keypoint
     */
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    
    /**
     * @brief Remove map point association at given index
     * @param idx Index of the keypoint to clear
     */
    void EraseMapPointMatch(const int &idx);
    
    /**
     * @brief Remove all associations with a specific map point
     * @param pMP Pointer to the map point to remove
     */
    void EraseMapPointMatch(MapPoint* pMP);
    
    /**
     * @brief Replace map point association at given index
     * @param idx Index of the keypoint
     * @param pMP Pointer to the new map point
     */
    void ReplaceMapPointMatch(const int &idx, MapPoint* pMP);
    
    /**
     * @brief Get all associated map points
     * @return Set of map point pointers
     */
    std::set<MapPoint*> GetMapPoints();
    
    /**
     * @brief Get map point associations as vector
     * @return Vector of map point pointers (with nullptrs for unmatched keypoints)
     */
    std::vector<MapPoint*> GetMapPointMatches();
    
    /**
     * @brief Count tracked map points with minimum observations
     * @param minObs Minimum number of observations required
     * @return Number of map points with at least minObs observations
     */
    int TrackedMapPoints(const int &minObs);
    
    /**
     * @brief Get map point at specific keypoint index
     * @param idx Index of the keypoint
     * @return Pointer to associated map point (nullptr if none)
     */
    MapPoint* GetMapPoint(const size_t &idx);

    // ==================== FEATURE MANAGEMENT ====================
    
    /**
     * @brief Get keypoint indices within circular area
     * @param x X coordinate of search center
     * @param y Y coordinate of search center
     * @param r Search radius in pixels
     * @return Vector of keypoint indices within the area
     */
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;

    // ==================== IMAGE BOUNDS ====================
    
    /**
     * @brief Check if point is within image bounds
     * @param x X coordinate in pixels
     * @param y Y coordinate in pixels
     * @return True if point is within image bounds
     */
    bool IsInImage(const float &x, const float &y) const;

    // ==================== BAD FLAG MANAGEMENT ====================
    
    /**
     * @brief Prevent this key frame from being erased
     */
    void SetNotErase();
    
    /**
     * @brief Allow this key frame to be erased
     */
    void SetErase();

    /**
     * @brief Mark this key frame as bad
     */
    void SetBadFlag();
    
    /**
     * @brief Check if this key frame is marked as bad
     * @return True if key frame is bad
     */
    bool isBad();

    // ==================== DEPTH COMPUTATION ====================
    
    /**
     * @brief Compute median scene depth for monocular SLAM
     * @param q Quantile parameter (typically 2 for median)
     * @return Median depth of tracked map points
     */
    float ComputeSceneMedianDepth(const int q);

    // ==================== STATIC UTILITY FUNCTIONS ====================
    
    /**
     * @brief Compare weights for sorting (descending order)
     * @param a First weight
     * @param b Second weight
     * @return True if a > b
     */
    static bool weightComp( int a, int b){
        return a>b;
    }

    /**
     * @brief Compare key frame IDs for sorting (ascending order)
     * @param pKF1 First key frame
     * @param pKF2 Second key frame
     * @return True if pKF1->mnId < pKF2->mnId
     */
    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }

    // ==================== IMU BIAS MANAGEMENT ====================
    
    /**
     * @brief Set new IMU bias values
     * @param b New bias values for accelerometer and gyroscope
     */
    void SetNewBias(const IMU::Bias &b);
    
    /**
     * @brief Get gyroscope bias
     * @return 3D gyroscope bias vector
     */
    Eigen::Vector3f GetGyroBias();

    /**
     * @brief Get accelerometer bias
     * @return 3D accelerometer bias vector
     */
    Eigen::Vector3f GetAccBias();

    /**
     * @brief Get complete IMU bias
     * @return IMU bias structure containing both gyro and accelerometer biases
     */
    IMU::Bias GetImuBias();

    // ==================== VOCABULARY MANAGEMENT ====================
    
    /**
     * @brief Set the vocabulary for bag-of-words representation
     * @param pVoc Pointer to the DBoW3 vocabulary
     */
    void SetVocabulary(DBoW3::Vocabulary* pVoc);

    // ==================== GRAPH EDGE MANAGEMENT ====================
    
    /**
     * @brief Add a map edge association
     * @param pME Pointer to the map edge
     * @param idx Index of the corresponding key edge
     */
    void AddMapEdge(MapEdge* pME, const size_t &idx);
    
    /**
     * @brief Get map edge at specific index
     * @param idx Index of the key edge
     * @return Pointer to associated map edge
     */
    MapEdge* GetMapEdge(int idx);

    /**
     * @brief Find edge index by point IDs
     * @param p1_id ID of first point
     * @param p2_id ID of second point
     * @return Index of the edge connecting the two points
     */
    int FineEdgeIdx(unsigned int p1_id, unsigned int p2_id);
    // ==================== PUBLIC MEMBER VARIABLES ====================
    // Note: The following variables are accessed from only 1 thread or never change (no mutex needed)

    // ==================== IDENTIFICATION ====================
    static long unsigned int nNextId;  ///< Global counter for generating unique key frame IDs
    long unsigned int mnId;            ///< Unique identifier for this key frame
    long unsigned int mnFrameId;       ///< Original frame ID from which this key frame was created

    double mTimeStamp;                 ///< Timestamp when this key frame was captured

    // ==================== GRID STRUCTURE ====================
    int mnGridCols;                    ///< Number of grid columns for feature matching acceleration
    int mnGridRows;                    ///< Number of grid rows for feature matching acceleration
    float mfGridElementWidthInv;       ///< Inverse width of each grid element
    float mfGridElementHeightInv;      ///< Inverse height of each grid element

    // ==================== TRACKING VARIABLES ====================
    long unsigned int mnTrackReferenceForFrame;  ///< Reference frame ID for tracking
    long unsigned int mnFuseTargetForKF;         ///< Target key frame ID for feature fusion

    // ==================== LOCAL MAPPING VARIABLES ====================
    long unsigned int mnBALocalForKF;  ///< Key frame ID for local bundle adjustment
    long unsigned int mnBAFixedForKF;   ///< Key frame ID when this was fixed in BA

    long unsigned int mnNumberOfOpt;    ///< Number of optimizations by bundle adjustment

    // ==================== KEYFRAME DATABASE VARIABLES ====================
    // Loop closure detection
    long unsigned int mnLoopQuery;      ///< Loop query ID for this key frame
    int mnLoopWords;                   ///< Number of words for loop detection
    float mLoopScore;                  ///< Loop closure score

    // Relocalization
    long unsigned int mnRelocQuery;     ///< Relocalization query ID
    int mnRelocWords;                  ///< Number of words for relocalization
    float mRelocScore;                 ///< Relocalization score

    // Map merging
    long unsigned int mnMergeQuery;     ///< Map merge query ID
    int mnMergeWords;                  ///< Number of words for map merging
    float mMergeScore;                 ///< Map merge score

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

    // ==================== IMAGE CALIBRATION ====================
    int mnMinX;                        ///< Minimum X coordinate of image bounds
    int mnMinY;                        ///< Minimum Y coordinate of image bounds
    int mnMaxX;                        ///< Maximum X coordinate of image bounds
    int mnMaxY;                        ///< Maximum Y coordinate of image bounds

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
    std::vector<long long int> mvBackupMapPointsId;  ///< Backup map point IDs for save/load functionality

    // ==================== VOCABULARY ====================
    DBoW3::Vocabulary* mpVocabulary;   ///< Pointer to bag-of-words vocabulary

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