#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <atomic>
#include <mutex>
#include <list>
#include <vector>
#include <unordered_set>

#include "Frame.h"
#include "PPGExtractor.h"
#include "IMU.h"
#include "SE3.h"
#include "GeometricCamera.h"

class LocalMapping;
class LoopClosing;
class System;
class Map;
class KeyFrame;
class MapPoint;

// Tracking state enumeration
enum eTrackingState{
    NO_IMAGES_YET=0,
    NOT_INITIALIZED=1,
    OK=2,
    RECENTLY_LOST=3,
    LOST=4,
};


/**
 * @brief Visual-Inertial SLAM Tracking Module
 * Implements camera pose estimation and feature tracking using PPG features
 * Handles monocular visual-inertial initialization and frame-to-frame tracking
 */
class MSTracking
{  
public:
    // Singleton pattern
    static MSTracking& get()
    {
        static MSTracking single_instance;
        return single_instance;
    }

private:
    MSTracking() = default;
    ~MSTracking() {}
    MSTracking(const MSTracking& other) = delete;
    MSTracking& operator=(const MSTracking& other) = delete;
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // System interface
    void Launch(Map* pMap, const string &strNet);  // Initialize tracking system
    void Reset();  // Reset tracking state
    
    // Main tracking functions
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);  // Process monocular image
    void GrabImuData(const IMU::Point &imuMeasurement);  // Add IMU measurement
    
    // Frame and IMU processing
    void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);  // Update frame with IMU data
    
    // State accessors
    KeyFrame* GetLastKeyFrame();  // Get most recent keyframe
    int GetMatchesInliers();  // Get current tracking quality
    
public:
    // Tracking state
    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Current processing
    Frame mCurrentFrame;
    Frame mLastFrame;
    cv::Mat mImGray;

    // Monocular initialization
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Trajectory recovery data
    std::list<SE3f> mlRelativeFramePoses;
    std::list<KeyFrame*> mlpReferences;
    std::list<double> mlFrameTimes;
    std::list<bool> mlbLost;

protected:
    // Core tracking pipeline
    void Track();  // Main tracking algorithm
    void MonocularInitialization();  // Initialize map from monocular sequence
    void CreateInitialMapMonocular();  // Create initial map points
    
    // Frame processing
    void CheckReplacedInLastFrame();  // Update replaced map points
    bool TrackReferenceKeyFrame();  // Track against reference keyframe
    bool TrackWithMotionModel();  // Track using constant velocity model
    bool PredictStateIMU();  // Predict pose using IMU
    bool Relocalization();  // Relocalize when tracking lost
    
    // Local mapping interface
    void UpdateLocalMap();  // Update local map for tracking
    void UpdateLocalPoints();  // Update local map points
    void UpdateLocalKeyFrames();  // Update local keyframes
    bool TrackLocalMap();  // Track against local map
    void SearchLocalPoints();  // Find matches with local points
    
    // Keyframe management
    bool NeedNewKeyFrame();  // Decide if new keyframe needed
    void CreateNewKeyFrame();  // Create and insert new keyframe
    
    // IMU processing
    void PreintegrateIMU();  // Preintegrate IMU from last frame
    void ResetFrameIMU();  // Reset IMU biases and compute velocity
    void InitializeIMU(float priorG = 1e2, float priorA = 1e6, bool bFirst = false);  // Initialize IMU parameters
    void ScaleRefinement();  // Refine scale estimation

private:
    // System components
    System* mpSystem;
    PPGExtractor* mpExtractor;  // Feature extractor
    Map* mpMap;  // Global map
    
    // Local mapping
    KeyFrame* mpReferenceKF;  // Reference keyframe
    std::vector<KeyFrame*> mvpLocalKeyFrames;  // Local keyframes
    std::vector<MapPoint*> mvpLocalMapPoints;  // Local map points
    
    // Tracking state
    KeyFrame* mpLastKeyFrame;  // Last keyframe
    unsigned int mnLastRelocFrameId;  // Last relocalization frame ID
    double mTimeStampLost;  // Time when tracking was lost
    bool mbMapUpdated;  // Map update flag
    bool mbReadyToInitializate;  // Initialization ready flag
    
    // IMU data and processing
    IMU::Preintegrated *mpImuPreintegratedFromLastKF;  // IMU preintegration
    std::list<IMU::Point> mlQueueImuData;  // IMU measurement queue
    std::vector<IMU::Point> mvImuFromLastFrame;  // IMU data between frames
    std::mutex mMutexImuQueue;  // IMU queue mutex
    IMU::Bias mLastBias;  // Last bias estimation
    
    // Motion model
    SE3f mVelocity;  // Frame velocity
    bool mInsertKFsLost;  // Insert keyframes when lost
    int mnMatchesInliers;  // Current inlier matches
    
    // IMU initialization parameters
    float mTinit;  // Initialization time
    
    // Calibration
    IMU::Calib *mpImuCalib;  // IMU calibration
    GeometricCamera* mpCamera;  // Camera model
};