/**
 * @file LocalMapping.h
 * @brief Local mapping module for PPG-SLAM system
 * @details Processes keyframes, performs bundle adjustment, and maintains 
 *          local map structure for visual-only and visual-inertial SLAM.
 */

#pragma once

#include "KeyFrame.h"
#include "Map.h"

#include <mutex>
#include <atomic>
#include <thread>
#include <list>

// Forward declarations
class System;
class Tracking;
class LoopClosing;
class MapEdge;

/**
 * @class MSLocalMapping
 * @brief Local mapping thread for real-time SLAM
 * @details Singleton class that handles keyframe processing, bundle adjustment,
 *          neighbor matching, and covisibility graph maintenance.
 */
class MSLocalMapping
{
public:
    /// Singleton access
    static MSLocalMapping& get() {
        static MSLocalMapping single_instance;
        return single_instance;
    }
    
private:
    MSLocalMapping() = default;
    ~MSLocalMapping() { mbFinishRequested.store(true); }
    MSLocalMapping(const MSLocalMapping&) = delete;
    MSLocalMapping& operator=(const MSLocalMapping&) = delete;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Core Functions
    void Launch(Map *pMap);              ///< Initialize and launch thread
    void Run();                          ///< Main processing loop
    void InsertKeyFrame(KeyFrame *pKF);  ///< Insert keyframe for processing
    void EmptyQueue();                   ///< Process all pending keyframes
    bool CheckNewKeyFrames();            ///< Check if keyframes are waiting

    // Thread Control
    void RequestStop();                  ///< Request thread stop
    void RequestReset();                 ///< Request system reset
    bool Stop();                         ///< Perform stop operation
    void Release();                      ///< Release from stopped state
    bool isStopped();                    ///< Check if stopped
    bool stopRequested();                ///< Check if stop requested
    bool SetNotStop(bool flag);          ///< Prevent stopping during critical ops
    void InterruptBA();                  ///< Interrupt bundle adjustment
    void RequestFinish();                ///< Request thread termination
    
    bool mbLocalMappingIdle;             ///< Idle state flag

private:
    // Internal Functions
    void ProcessNewKeyFrame();           ///< Process next keyframe from queue
    void SearchInNeighbors();            ///< Search matches in neighboring keyframes
    void ResetIfRequested();             ///< Handle reset requests

    // Member Variables
    Map* mpMap;                          ///< Map being processed
    KeyFrame *mpCurrentKeyFrame;         ///< Current keyframe
    std::list<KeyFrame*> mlNewKeyFrames; ///< Keyframe processing queue

    // Thread Synchronization
    std::mutex mMutexNewKFs;             ///< Keyframe queue mutex
    std::mutex mMutexReset;              ///< Reset operation mutex
    std::mutex mMutexStop;               ///< Stop operation mutex
    
    // Control Flags
    std::atomic<bool> mbFinishRequested; ///< Thread termination request
    bool mbResetRequested;               ///< Reset request flag
    bool mbAbortBA;                      ///< Bundle adjustment abort flag
    bool mbStopped;                      ///< Thread stopped flag
    bool mbStopRequested;                ///< Stop request flag
    bool mbNotStop;                      ///< Prevent stop flag

    std::thread* mptLocalMapping;        ///< Local mapping thread pointer
};