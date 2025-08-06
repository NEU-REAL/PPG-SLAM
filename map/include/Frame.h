#pragma once

#include <vector>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "DBoW3/DBoW3.h"
#include "sophus/geometry.hpp"

#include "IMU.h"
#include "Eigen/Core"
#include "sophus/se3.hpp"
#include "PPGExtractor.h"
#include "PPGGraph.h"
#include "GeometricCamera.h"
#include "KeyFrame.h"

class KeyPointEx;
class KeyEdge;
class KeyFrame;

class MapEdge;
class MapPoint;
class ConstraintPoseImu;
class GeometricCamera;
class PPGExtractor;

using namespace std;

class Frame
{
public:
    Frame();
    Frame(const Frame &frame);
    Frame(const cv::Mat &imGray, const double &timeStamp, PPGExtractor* pExt, GeometricCamera* pCam, IMU::Calib *pImu, Frame* pPrevF);
    KeyFrame* buildKeyFrame(Map* pMap);
    void ComputeBoW(Map* pMap);
    // Set the camera pose. (Imu pose is not modified!)
    void SetPose(const Sophus::SE3<float> &Tcw);
    // Set IMU velocity
    void SetVelocity(Eigen::Vector3f Vw);
    Eigen::Vector3f GetVelocity() const;
    // Set IMU pose and velocity (implicitly changes camera pose)
    void SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb);
    Eigen::Matrix<float,3,1> GetImuPosition() const;
    Eigen::Matrix<float,3,3> GetImuRotation();
    Sophus::SE3<float> GetImuPose();

    void SetNewBias(const IMU::Bias &b);
    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    void CheckInFrustum(MapPoint* pMP, float viewingCosLimit);
    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const KeyPointEx &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();
    // Returns the camera center.
    inline Eigen::Vector3f GetCameraCenter(){ return mOw; }
    // Returns inverse of rotation
    inline Eigen::Matrix3f GetRotationInverse(){ return mRwc; }
    //TODO: can the Frame pose be accsessed from several threads? should this be protected somehow?
    inline Sophus::SE3<float> GetPose() const { return mTcw; }
    inline Eigen::Matrix3f GetRwc() const { return mRwc; }
    inline Eigen::Vector3f GetOw() const { return mOw; }
    inline bool HasPose() const { return mbHasPose; }
    inline bool HasVelocity() const { return mbHasVelocity; }

    Eigen::Vector2f ProjectPoint(MapPoint* pMP);

public:
    ConstraintPoseImu* mpcpi;
    // Vocabulary used for relocalization.
    DBoW3::Vocabulary* mpVocabulary;
    // Feature extractor. The right is used only in the stereo case.
    PPGExtractor* mpExtractor;
    // PPGExtractor* mpExtractor;
    // Imu preintegration from last keyframe
    IMU::Preintegrated *mpImuPreintegrated, *mpImuPreintegratedFrame;
    // Pointer to previous frame
    Frame* mpPrevFrame;
    // Pointer to last frame
    KeyFrame *mpLastKeyFrame, *mpReferenceKF;

    // IMU bias
    IMU::Bias mImuBias;

    // calibration
    GeometricCamera* mpCamera;
    IMU::Calib *mpImuCalib;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;
    // Frame timestamp.
    double mTimeStamp;

    // Number of KeyPoints.
    int N;
    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<KeyPointEx> mvKeys;
    std::vector<KeyPointEx> mvKeysUn;
    std::vector<KeyEdge> mvKeyEdges;
    // Corresponding stereo coordinate and depth for each keypoint.
    std::vector<MapPoint*> mvpMapPoints;
    std::vector<MapEdge*> mvpMapEdges;
    // Bag of Words Vector structures.
    DBoW3::BowVector mBowVec;
    DBoW3::FeatureVector mFeatVec;
    // descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors;
    // MapPoints associated to keypoints, NULL pointer if no association.
    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;
    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    // Scale pyramid info.

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;
    static bool mbInitialComputations;

public:
    //Sophus/Eigen migration
    Sophus::SE3<float> mTcw;
    Eigen::Matrix<float,3,3> mRwc;
    Eigen::Matrix<float,3,1> mOw;
    Eigen::Matrix<float,3,3> mRcw;
    Eigen::Matrix<float,3,1> mtcw;
    bool mbHasPose;

    //Rcw_ not necessary as Sophus has a method for extracting the rotation matrix: Tcw_.rotationMatrix()
    //tcw_ not necessary as Sophus has a method for extracting the translation vector: Tcw_.translation()
    //Twc_ not necessary as Sophus has a method for easily computing the inverse pose: Tcw_.inverse()

    // IMU linear velocity
    Eigen::Vector3f mVw;
    bool mbHasVelocity;


public:
    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);
    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    bool mbImuPreintegrated;
public:
    cv::Mat srcMat; // debug
};
