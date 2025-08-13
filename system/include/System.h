#pragma once

#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "Tracking.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Viewer.h"
#include "IMU.h"
#include "KannalaBrandt8.h"
#include "Pinhole.h"
#include "GeometricCamera.h"
#include "Map.h"

class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

class System
{
public:

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const string &strVocFile, const string &strSettingsFile,const string &strNet, const bool bUseViewer = true);

    // Proccess the given monocular frame and optionally imu data
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // This stops local mapping thread (map building) and performs only camera tracking.
    // This resumes local mapping thread and performs SLAM again.

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();

public:

    // vocabulary used for place recognition and feature matching.
    DBoW3::Vocabulary* mpVocabulary;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    Map* mpMap;

    // Reset flag
    std::mutex mMutexReset;

    // Shutdown flag
    bool mbShutDown;
};