/**
 * @file System.h
 * @brief Main SLAM system class for PPG-SLAM
 * @author PPG-SLAM Team
 */

#pragma once

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <mutex>
#include <opencv2/core/core.hpp>

#include "Tracking.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Viewer.h"
#include "IMU.h"
#include "KannalaBrandt8.h"
#include "Pinhole.h"
#include "GeometricCamera.h"
#include "SE3.h"
#include "Map.h"

using namespace std;

// Forward declarations
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

/**
 * @brief Main SLAM system class that coordinates all threads and components
 * 
 * This class serves as the main interface for the PPG-SLAM system.
 * It initializes and manages all major components including tracking,
 * local mapping, loop closing, and visualization threads.
 */
class System
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    /**
     * @brief Initialize the SLAM system and launch all threads
     * @param strVocFile Path to vocabulary file for place recognition
     * @param strSettingsFile Path to configuration file with camera/IMU parameters
     * @param strNet Path to neural network models for feature extraction
     * @param bUseViewer Enable/disable visualization thread (default: true)
     */
    System(const string &strVocFile, const string &strSettingsFile, const string &strNet, const bool bUseViewer = true);

    /**
     * @brief Process monocular frame with optional IMU data
     * @param im Input image (RGB CV_8UC3 or grayscale CV_8U, RGB will be converted)
     * @param timestamp Image timestamp in seconds
     * @param vImuMeas Vector of IMU measurements between frames (optional)
     * @param filename Debug filename for logging (optional)
     * @return Camera pose SE3f (empty if tracking fails)
     */
    SE3f TrackMonocular(const cv::Mat &im, const double &timestamp, 
                       const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), 
                       string filename = "");

    /**
     * @brief Shutdown all threads and save trajectory
     * 
     * Requests all threads to finish and waits for completion.
     * Must be called before program termination to save trajectory data.
     */
    void Shutdown();

private:
    // Core SLAM components
    DBoW3::Vocabulary* mpVocabulary;    ///< Vocabulary for place recognition and feature matching
    Map* mpMap;                         ///< Map structure storing KeyFrames and MapPoints

    // Thread synchronization
    std::mutex mMutexReset;             ///< Mutex for reset operations
    bool mbShutDown;                    ///< Shutdown flag to coordinate thread termination
};