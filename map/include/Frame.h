#pragma once

/**
 * @file Frame.h
 * @brief Frame class for visual-inertial SLAM
 */

#include <vector>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sophus/geometry.hpp>
#include "DBoW3/DBoW3.h"
#include "IMU.h"
#include "PPGExtractor.h"
#include "PPGGraph.h"
#include "GeometricCamera.h"
#include "KeyFrame.h"
#include "Map.h"

// Forward declarations
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
 * @brief Frame class for visual-inertial SLAM
 */
class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // ==================== CONSTRUCTORS ====================
    
    Frame();
    Frame(const Frame &frame);
    Frame(const cv::Mat &imGray, const double &timeStamp, PPGExtractor* pExt, GeometricCamera* pCam, IMU::Calib *pImu, Frame* pPrevF);

    // ==================== KEYFRAME CREATION ====================
    
    KeyFrame* buildKeyFrame(Map* pMap);
    void ComputeBoW(Map* pMap);

    // ==================== POSE MANAGEMENT ====================
    
    /// Set camera pose (IMU pose is not modified)
    void SetPose(const Sophus::SE3<float> &Tcw);
    
    /// Set IMU pose and velocity (implicitly changes camera pose)
    void SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb);
    
    /// Set IMU velocity
    void SetVelocity(Eigen::Vector3f Vw);
    
    /// Set new IMU bias
    void SetNewBias(const IMU::Bias &b);

    // ==================== POSE GETTERS ====================
    
    inline Sophus::SE3<float> GetPose() const { return mTcw; }
    inline Eigen::Vector3f GetCameraCenter() { return mOw; }
    inline Eigen::Matrix3f GetRotationInverse() { return mRwc; }
    inline Eigen::Matrix3f GetRwc() const { return mRwc; }
    inline Eigen::Vector3f GetOw() const { return mOw; }
    inline bool HasPose() const { return mbHasPose; }
    inline bool HasVelocity() const { return mbHasVelocity; }
    
    Eigen::Vector3f GetVelocity() const;
    Eigen::Vector3f GetImuPosition() const;
    Eigen::Matrix3f GetImuRotation();
    Sophus::SE3f GetImuPose();

    // ==================== FEATURE MANAGEMENT ====================
    
    /// Check if MapPoint is in camera frustum
    void CheckInFrustum(MapPoint* pMP, float viewingCosLimit);
    
    /// Compute grid cell of a keypoint
    bool PosInGrid(const KeyPointEx &kp, int &posX, int &posY);
    
    /// Get features in area
    std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r) const;
    
    /// Project MapPoint to image
    Eigen::Vector2f ProjectPoint(MapPoint* pMP);

private:
    // ==================== INTERNAL METHODS ====================
    
    /// Update pose matrices from mTcw
    void UpdatePoseMatrices();
    
    /// Assign features to grid for fast matching
    void AssignFeaturesToGrid();

public:
    // ==================== FRAME IDENTIFIERS ====================
    
    static long unsigned int nNextId;      ///< Next frame ID
    long unsigned int mnId;                ///< Current frame ID
    double mTimeStamp;                     ///< Frame timestamp

    // ==================== FEATURE DATA ====================
    
    int N;                                 ///< Number of keypoints
    std::vector<KeyPointEx> mvKeys;        ///< Original keypoints
    std::vector<KeyPointEx> mvKeysUn;      ///< Undistorted keypoints
    std::vector<KeyEdge> mvKeyEdges;       ///< Key edges
    cv::Mat mDescriptors;                  ///< Feature descriptors

    // ==================== MAP ASSOCIATIONS ====================
    
    std::vector<MapPoint*> mvpMapPoints;   ///< Associated map points
    std::vector<MapEdge*> mvpMapEdges;     ///< Associated map edges
    std::vector<bool> mvbOutlier;          ///< Outlier flags

    // ==================== BAG OF WORDS ====================
    
    DBoW3::BowVector mBowVec;              ///< Bag of words vector
    DBoW3::FeatureVector mFeatVec;         ///< Feature vector

    // ==================== GRID FOR FAST MATCHING ====================
    
    std::vector<std::size_t> mGrid[GeometricCamera::FRAME_GRID_COLS][GeometricCamera::FRAME_GRID_ROWS];

    // ==================== POSE STATE ====================
    
    Sophus::SE3<float> mTcw;               ///< Camera pose
    Eigen::Matrix3f mRwc;                  ///< Rotation world to camera
    Eigen::Vector3f mOw;                   ///< Camera center in world
    Eigen::Matrix3f mRcw;                  ///< Rotation camera to world
    Eigen::Vector3f mtcw;                  ///< Translation camera to world
    bool mbHasPose;                        ///< Has valid pose

    // ==================== IMU STATE ====================
    
    Eigen::Vector3f mVw;                   ///< IMU velocity
    bool mbHasVelocity;                    ///< Has valid velocity
    IMU::Bias mImuBias;                    ///< IMU bias
    bool mbImuPreintegrated;               ///< IMU preintegration flag

    // ==================== POINTERS ====================
    
    ConstraintPoseImu* mpcpi;              ///< Pose constraint
    PPGExtractor* mpExtractor;             ///< Feature extractor
    IMU::Preintegrated *mpImuPreintegrated, *mpImuPreintegratedFrame;  ///< IMU preintegration
    Frame* mpPrevFrame;                    ///< Previous frame
    KeyFrame *mpLastKeyFrame, *mpReferenceKF;  ///< Reference keyframes
    GeometricCamera* mpCamera;             ///< Camera model
    IMU::Calib *mpImuCalib;                ///< IMU calibration

public:
    cv::Mat srcMat;                        ///< Debug: source image
};
