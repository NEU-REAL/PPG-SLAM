#pragma once

/**
 * @file Frame.h
 * @brief Frame class representing individual frames in PPG-SLAM system
 */

// ==================== SYSTEM INCLUDES ====================
#include <vector>
#include <mutex>
#include <atomic>

// ==================== THIRD-PARTY INCLUDES ====================
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "SE3.h"
#include "DBoW3/DBoW3.h"

// ==================== LOCAL INCLUDES ====================
#include "IMU.h"
#include "PPGExtractor.h"
#include "PPGGraph.h"
#include "GeometricCamera.h"
#include "KeyFrame.h"
#include "Map.h"

// ==================== FORWARD DECLARATIONS ====================
class KeyPointEx;
class KeyEdge;
class KeyFrame;
class MapEdge;
class MapPoint;
class ConstraintPoseImu;
class GeometricCamera;
class PPGExtractor;
class Map;

/**
 * @class Frame
 * @brief Individual frame in visual-inertial SLAM with pose, features, and IMU data
 */
class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // ==================== CONSTRUCTORS/DESTRUCTORS ====================
    
    /**
     * @brief Default constructor
     */
    Frame();
    
    /**
     * @brief Copy constructor
     */
    Frame(const Frame &frame);
    
    /**
     * @brief Constructor with image and sensor data
     * @param imGray Input grayscale image
     * @param timeStamp Frame timestamp
     * @param pExt Feature extractor
     * @param pCam Camera model
     * @param pImu IMU calibration
     * @param pPrevF Previous frame
     */
    Frame(const cv::Mat &imGray, const double &timeStamp, PPGExtractor* pExt, GeometricCamera* pCam, IMU::Calib *pImu, Frame* pPrevF);

    // ==================== KEYFRAME CREATION ====================
    
    /**
     * @brief Create keyframe from current frame
     * @param pMap Map reference
     * @return Pointer to created keyframe
     */
    KeyFrame* buildKeyFrame(Map* pMap);
    
    /**
     * @brief Compute bag of words representation
     * @param pMap Map reference
     */
    void ComputeBoW(Map* pMap);

    // ==================== POSE MANAGEMENT ====================
    
    /**
     * @brief Set camera pose
     * @param Tcw Camera pose transformation
     */
    void SetPose(const SE3<float> &Tcw);
    
    /**
     * @brief Set IMU pose and velocity
     * @param Rwb IMU rotation
     * @param twb IMU translation
     * @param Vwb IMU velocity
     */
    void SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb);
    
    /**
     * @brief Set IMU velocity
     * @param Vw Velocity vector
     */
    void SetVelocity(Eigen::Vector3f Vw);
    
    /**
     * @brief Set new IMU bias
     * @param b IMU bias
     */
    void SetNewBias(const IMU::Bias &b);

    // ==================== POSE GETTERS ====================
    
    /// Camera pose getter functions
    inline SE3<float> GetPose() const { return mTcw; }               ///< Get camera pose
    inline Eigen::Vector3f GetCameraCenter() { return mOw; }                 ///< Get camera center
    inline Eigen::Matrix3f GetRotationInverse() { return mRwc; }             ///< Get inverse rotation
    inline Eigen::Matrix3f GetRwc() const { return mRwc; }                   ///< Get world to camera rotation
    inline Eigen::Vector3f GetOw() const { return mOw; }                     ///< Get camera center in world
    inline bool HasPose() const { return mbHasPose; }                        ///< Check if pose is valid
    inline bool HasVelocity() const { return mbHasVelocity; }                ///< Check if velocity is valid
    
    /// IMU state getter functions
    Eigen::Vector3f GetVelocity() const;                                     ///< Get IMU velocity
    Eigen::Vector3f GetImuPosition() const;                                  ///< Get IMU position
    Eigen::Matrix3f GetImuRotation();                                        ///< Get IMU rotation
    SE3f GetImuPose();                                               ///< Get IMU pose

    // ==================== FEATURE MANAGEMENT ====================
    
    /**
     * @brief Check if MapPoint is visible in frame
     * @param pMP Map point to check
     * @param viewingCosLimit Viewing angle limit
     */
    void CheckInFrustum(MapPoint* pMP, float viewingCosLimit);
    
    /**
     * @brief Get grid position of keypoint
     * @param kp Keypoint
     * @param posX Output x position
     * @param posY Output y position
     * @return True if position is valid
     */
    bool PosInGrid(const KeyPointEx &kp, int &posX, int &posY);
    
    /**
     * @brief Get features in circular area
     * @param x Center x coordinate
     * @param y Center y coordinate
     * @param r Radius
     * @return Vector of feature indices
     */
    std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r) const;
    
    /**
     * @brief Project map point to image
     * @param pMP Map point to project
     * @return Image coordinates
     */
    Eigen::Vector2f ProjectPoint(MapPoint* pMP);

private:
    // ==================== INTERNAL METHODS ====================
    
    /**
     * @brief Update pose matrices from mTcw
     */
    void UpdatePoseMatrices();
    
    /**
     * @brief Assign features to grid for fast matching
     */
    void AssignFeaturesToGrid();

public:
    // ==================== FRAME IDENTIFIERS ====================
    
    static long unsigned int nNextId;                           ///< Next frame ID counter
    long unsigned int mnId;                                     ///< Current frame ID
    double mTimeStamp;                                          ///< Frame timestamp

    // ==================== FEATURE DATA ====================
    
    int N;                                                      ///< Number of keypoints
    std::vector<KeyPointEx> mvKeys;                             ///< Original keypoints
    std::vector<KeyPointEx> mvKeysUn;                           ///< Undistorted keypoints
    std::vector<KeyEdge> mvKeyEdges;                            ///< Key edges
    cv::Mat mDescriptors;                                       ///< Feature descriptors

    // ==================== MAP ASSOCIATIONS ====================
    
    std::vector<MapPoint*> mvpMapPoints;                        ///< Associated map points
    std::vector<MapEdge*> mvpMapEdges;                          ///< Associated map edges
    std::vector<bool> mvbOutlier;                               ///< Outlier flags

    // ==================== BAG OF WORDS ====================
    
    DBoW3::BowVector mBowVec;                                   ///< Bag of words vector
    DBoW3::FeatureVector mFeatVec;                              ///< Feature vector

    // ==================== GRID FOR FAST MATCHING ====================
    
    /// Grid for efficient feature matching
    std::vector<std::size_t> mGrid[GeometricCamera::FRAME_GRID_COLS][GeometricCamera::FRAME_GRID_ROWS];

    // ==================== POSE STATE ====================
    
    SE3<float> mTcw;                                    ///< Camera pose transformation
    Eigen::Matrix3f mRwc;                                       ///< World to camera rotation
    Eigen::Vector3f mOw;                                        ///< Camera center in world coordinates
    Eigen::Matrix3f mRcw;                                       ///< Camera to world rotation
    Eigen::Vector3f mtcw;                                       ///< Camera to world translation
    bool mbHasPose;                                             ///< Valid pose flag

    // ==================== IMU STATE ====================
    
    Eigen::Vector3f mVw;                                        ///< IMU velocity in world frame
    bool mbHasVelocity;                                         ///< Valid velocity flag
    IMU::Bias mImuBias;                                         ///< IMU bias estimate
    bool mbImuPreintegrated;                                    ///< IMU preintegration status

    // ==================== POINTERS ====================
    
    ConstraintPoseImu* mpcpi;                                   ///< Pose constraint
    PPGExtractor* mpExtractor;                                  ///< Feature extractor
    IMU::Preintegrated *mpImuPreintegrated, *mpImuPreintegratedFrame;  ///< IMU preintegration data
    Frame* mpPrevFrame;                                         ///< Previous frame reference
    KeyFrame *mpLastKeyFrame, *mpReferenceKF;                   ///< Keyframe references
    GeometricCamera* mpCamera;                                  ///< Camera model
    IMU::Calib *mpImuCalib;                                     ///< IMU calibration

public:
    cv::Mat srcMat;                                             ///< Source image for debugging
};
