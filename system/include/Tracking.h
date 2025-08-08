#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <atomic>

#include "Viewer.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "PPGExtractor.h"
#include "System.h"
#include "IMU.h"

#include "GeometricCamera.h"

#include <mutex>
#include <unordered_set>

class LocalMapping;
class LoopClosing;
class System;

// Tracking states
enum eTrackingState{
    NO_IMAGES_YET=0,
    NOT_INITIALIZED=1,
    OK=2,
    RECENTLY_LOST=3,
    LOST=4,
};


class MSTracking
{  
public:
    static MSTracking& get()
    {
        static MSTracking single_instance;
        return single_instance;
    }
private:
    MSTracking() = default;
    ~MSTracking()
    {}
    MSTracking(const MSTracking& other) = delete;
    MSTracking& operator=(const MSTracking& other) = delete;
    static MSTracking* singleViewing;
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void Launch(Map* pMap, const string &strNet);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    Sophus::SE3f GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);
    void GrabImuData(const IMU::Point &imuMeasurement);

    void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);

    KeyFrame* GetLastKeyFrame();

    int GetMatchesInliers();

    void Reset();
    
public:

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Current Frame
    Frame mCurrentFrame;
    Frame mLastFrame;

    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<Sophus::SE3f> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

protected:
    // Main tracking function.
    void Track();
    // Map initialization for monocular
    void MonocularInitialization();
    //void CreateNewMapPoints();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    bool TrackWithMotionModel();
    bool PredictStateIMU();
    // Map
    bool Relocalization();
    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();
    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();
    // Perform preintegration from last frame
    void PreintegrateIMU();
    // Reset IMU biases and compute frame velocity
    void ResetFrameIMU();
private:
    // System
    System* mpSystem;
    //ext
    PPGExtractor* mpExtractor;
    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    //Map
    Map* mpMap;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    unsigned int mnLastRelocFrameId;
    double mTimeStampLost;
    // map flags
    bool mbMapUpdated;

    // Initalization (only for monocular)
    bool mbReadyToInitializate;

    // IMU
    // Imu preintegration from last frame
    IMU::Preintegrated *mpImuPreintegratedFromLastKF;
    // Queue of IMU measurements between frames
    std::list<IMU::Point> mlQueueImuData;
    // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
    std::vector<IMU::Point> mvImuFromLastFrame;
    std::mutex mMutexImuQueue;
    // Last Bias Estimation (at keyframe creation)
    IMU::Bias mLastBias;

    // Threshold close/far points
    // Points seen as close by the stereo are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    bool mInsertKFsLost;

    //Current matches in frame
    int mnMatchesInliers;
    
    //Motion Model
    Sophus::SE3f mVelocity;

    //calibration 
    IMU::Calib *mpImuCalib;
    GeometricCamera* mpCamera;
};