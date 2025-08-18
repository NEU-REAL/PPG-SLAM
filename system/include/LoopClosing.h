#pragma once

// Standard library includes
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <list>
#include <map>
#include <set>

// Third-party includes
#include <boost/algorithm/string.hpp>
#include "g2o/types/sim3/types_seven_dof_expmap.h"

// Project includes
#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Optimizer.h"

// Forward declarations
class LocalMapping;
class Map;

/**
 * @brief Loop Closing Module - Singleton class for SLAM loop detection and correction
 * 
 * This class implements a robust loop detection and closure system for visual-inertial SLAM.
 * It uses a combination of Bag-of-Words (BoW) place recognition and geometric verification
 * to detect when the robot revisits previously mapped areas, then corrects accumulated
 * drift through pose graph optimization.
 * 
 * Key features:
 * - Singleton pattern ensures only one loop closing instance
 * - Multi-threaded operation for real-time performance
 * - Handles both monocular and stereo configurations
 * - Supports inertial sensor integration for improved robustness
 */
class MSLoopClosing
{
public:
    // Type definitions for cleaner code
    typedef std::map<KeyFrame*, g2o::Sim3, std::less<KeyFrame*>,
                     Eigen::aligned_allocator<std::pair<KeyFrame* const, g2o::Sim3>>> KeyFrameAndPose;

    /**
     * @brief Get singleton instance of MSLoopClosing
     * @return Reference to the single MSLoopClosing instance
     */
    static MSLoopClosing& get()
    {
        static MSLoopClosing single_instance;
        return single_instance;
    }

private:
    // Singleton pattern implementation
    MSLoopClosing() = default;
    ~MSLoopClosing()
    {
        mbFinishRequested = true;
    }
    MSLoopClosing(const MSLoopClosing& other) = delete;
    MSLoopClosing& operator=(const MSLoopClosing& other) = delete;
    
public:
    // =============================================================================
    // PUBLIC INTERFACE METHODS
    // =============================================================================

    /**
     * @brief Initialize and launch the loop closing thread
     * @param pMap Pointer to the active map
     * @param bActiveLC Flag to enable/disable loop closing functionality
     */
    void Launch(Map* pMap, const bool bActiveLC);

    /**
     * @brief Main loop closing thread function
     * 
     * Continuously processes keyframes from the queue, detects loops using
     * place recognition algorithms, and performs loop closure when detected.
     * Runs until finish is requested.
     */
    void Run();

    /**
     * @brief Insert a new keyframe into the loop closing queue
     * @param pKF Pointer to the keyframe to be processed
     * 
     * Thread-safe insertion of keyframes for loop detection processing.
     * Keyframes with ID 0 (typically the first frame) are ignored.
     */
    void InsertKeyFrame(KeyFrame *pKF);

    /**
     * @brief Request system reset
     * 
     * Clears the keyframe processing queue and resets internal state.
     * Blocks until reset is completed.
     */
    void RequestReset();

    /**
     * @brief Execute Global Bundle Adjustment in separate thread
     * @param pActiveMap Map to optimize
     * @param nLoopKF ID of the loop keyframe that triggered GBA
     * 
     * Performs full map optimization after loop closure to minimize
     * reprojection errors across the entire trajectory.
     */
    void RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF);

    /**
     * @brief Check if Global Bundle Adjustment is currently running
     * @return true if GBA is active, false otherwise
     */
    bool isRunningGBA()
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }

    /**
     * @brief Request graceful shutdown of loop closing thread
     * 
     * Sets finish flag and waits for thread to complete current operations.
     */
    void RequestFinish();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    // =============================================================================
    // PROTECTED CORE ALGORITHMS
    // =============================================================================

    /**
     * @brief Check if new keyframes are available for processing
     * @return true if keyframes exist in queue, false otherwise
     */
    bool CheckNewKeyFrames();

    /**
     * @brief Main place recognition algorithm for loop detection
     * @return true if a loop candidate region is detected
     * 
     * Implements a multi-stage approach:
     * 1. Geometric validation of previous candidates
     * 2. Bag-of-Words based place recognition
     * 3. Spatial consistency checking
     */
    bool NewDetectCommonRegions();

    /**
     * @brief Refine Sim3 transformation from previous detection
     * @param pCurrentKF Current keyframe being processed
     * @param pMatchedKF Previously matched keyframe
     * @param gScw Sim3 transformation (scale, rotation, translation)
     * @param nNumProjMatches Number of projection matches found
     * @param vpMPs Vector of map points from matched keyframe
     * @param vpMatchedMPs Vector of matched map points
     * @return true if refinement succeeds with sufficient matches
     * 
     * Uses iterative optimization to refine the Sim3 transformation
     * between current and matched keyframes based on map point projections.
     */
    bool DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, 
                                        int &nNumProjMatches, std::vector<MapPoint*> &vpMPs, 
                                        std::vector<MapPoint*> &vpMatchedMPs);

    /**
     * @brief Detect loop candidates using Bag-of-Words approach
     * @param vpBowCand Vector of BoW candidate keyframes
     * @param pMatchedKF Output: best matched keyframe
     * @param pLastCurrentKF Output: last current keyframe in sequence
     * @param g2oScw Output: computed Sim3 transformation
     * @param nNumCoincidences Output: number of sequential detections
     * @param vpMPs Output: map points from matched region
     * @param vpMatchedMPs Output: corresponding matched map points
     * @return true if a valid loop is detected
     * 
     * Implements robust place recognition using:
     * - BoW similarity scoring
     * - RANSAC-based geometric verification
     * - Covisibility analysis for spatial consistency
     */
    bool DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand, KeyFrame* &pMatchedKF, 
                                    KeyFrame* &pLastCurrentKF, g2o::Sim3 &g2oScw,
                                    int &nNumCoincidences, std::vector<MapPoint*> &vpMPs, 
                                    std::vector<MapPoint*> &vpMatchedMPs);

    /**
     * @brief Detect common regions using previous keyframe information
     * @param pCurrentKF Current keyframe
     * @param pMatchedKF Matched keyframe from previous detection
     * @param gScw Sim3 transformation to verify
     * @param nNumProjMatches Output: number of projection matches
     * @param vpMPs Map points to project
     * @param vpMatchedMPs Output: successfully matched map points
     * @return true if sufficient matches are found
     * 
     * Validates loop hypothesis by projecting map points from the matched
     * region into the current keyframe using the estimated transformation.
     */
    bool DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, 
                                       int &nNumProjMatches, std::vector<MapPoint*> &vpMPs, 
                                       std::vector<MapPoint*> &vpMatchedMPs);

    /**
     * @brief Find matches by projecting map points into keyframe
     * @param pCurrentKF Current keyframe
     * @param pMatchedKFw Matched keyframe (world frame)
     * @param g2oScw Sim3 transformation
     * @param spMatchedMPinOrigin Set of already matched map points
     * @param vpMapPoints Vector of map points to project
     * @param vpMatchedMapPoints Output: vector of matched map points
     * @return Number of successful matches
     * 
     * Projects 3D map points into the current keyframe using the estimated
     * transformation and finds correspondences with observed features.
     */
    int FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw, g2o::Sim3 &g2oScw,
                                std::set<MapPoint*> &spMatchedMPinOrigin, std::vector<MapPoint*> &vpMapPoints,
                                std::vector<MapPoint*> &vpMatchedMapPoints);

    /**
     * @brief Fuse duplicate map points after loop closure
     * @param CorrectedPosesMap Map of corrected keyframe poses
     * @param vpMapPoints Vector of map points to fuse
     * 
     * Identifies and merges duplicate map points that represent the same
     * 3D feature but were created separately before loop closure.
     */
    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, std::vector<MapPoint*> &vpMapPoints);

    /**
     * @brief Execute complete loop correction procedure
     * 
     * Comprehensive loop closure implementation:
     * 1. Stop local mapping to prevent conflicts
     * 2. Compute pose corrections using Sim3 optimization
     * 3. Update keyframe poses and map point positions
     * 4. Fuse duplicate map points
     * 5. Optimize pose graph structure
     * 6. Launch Global Bundle Adjustment if needed
     */
    void CorrectLoop();

    /**
     * @brief Handle reset requests in thread-safe manner
     * 
     * Processes pending reset requests and clears internal state.
     */
    void ResetIfRequested();

private:
    // =============================================================================
    // MEMBER VARIABLES - SYSTEM STATE
    // =============================================================================

    // Reset mechanism
    bool mbResetRequested;                    ///< Flag indicating reset is requested
    Map* mpMapToReset;                        ///< Map to reset (currently unused)
    std::mutex mMutexReset;                   ///< Mutex for thread-safe reset operations

    std::atomic<bool> mbFinishRequested;      ///< Atomic flag for graceful shutdown

    // Core system components
    Map* mpMap;                               ///< Pointer to the active map

    // Keyframe processing queue
    std::list<KeyFrame*> mlpLoopKeyFrameQueue; ///< Queue of keyframes awaiting loop detection
    std::mutex mMutexLoopQueue;               ///< Mutex protecting keyframe queue

    // =============================================================================
    // LOOP DETECTION STATE VARIABLES
    // =============================================================================

    // Current detection state
    KeyFrame* mpCurrentKF;                    ///< Currently processed keyframe
    KeyFrame* mpLastCurrentKF;                ///< Previous keyframe in detection sequence
    std::vector<KeyFrame*> mvpCurrentConnectedKFs; ///< Covisible keyframes of current KF
    std::vector<MapPoint*> mvpLoopMapPoints;  ///< Map points involved in loop closure

    // Transformation matrices
    cv::Mat mScw;                             ///< Similarity transformation (deprecated)
    g2o::Sim3 mg2oScw;                        ///< Current Sim3 transformation

    // Loop detection flags and counters
    bool mbLoopDetected;                      ///< Flag indicating successful loop detection
    int mnLoopNumCoincidences;                ///< Number of sequential loop detections
    int mnLoopNumNotFound;                    ///< Counter for failed detections

    // Loop closure data
    KeyFrame* mpLoopLastCurrentKF;            ///< Last current keyframe in loop sequence
    g2o::Sim3 mg2oLoopSlw;                    ///< Sim3 from loop last to world
    g2o::Sim3 mg2oLoopScw;                    ///< Sim3 from current to world (corrected)
    KeyFrame* mpLoopMatchedKF;                ///< Keyframe matched in loop detection
    std::vector<MapPoint*> mvpLoopMPs;        ///< Map points from loop region
    std::vector<MapPoint*> mvpLoopMatchedMPs; ///< Matched map points for fusion

    // =============================================================================
    // GLOBAL BUNDLE ADJUSTMENT STATE
    // =============================================================================

    bool mbRunningGBA;                        ///< Flag indicating GBA is running
    bool mbStopGBA;                           ///< Flag to request GBA termination
    std::mutex mMutexGBA;                     ///< Mutex for GBA state synchronization
    std::thread* mpThreadGBA;                 ///< Thread handle for GBA execution

    // Optimization tracking
    int mnFullBAIdx;                          ///< Index for full bundle adjustment tracking

    // =============================================================================
    // CONFIGURATION FLAGS
    // =============================================================================

    bool mbActiveLC = true;                   ///< Enable/disable loop closing functionality

    // Threading
    std::thread* mptLoopClosing;              ///< Main loop closing thread handle
};