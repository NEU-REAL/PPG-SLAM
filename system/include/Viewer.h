#pragma once

// Standard libraries
#include <mutex>
#include <thread>
#include <atomic>
#include <unordered_set>
#include <vector>
#include <string>

// OpenCV libraries
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

// Third-party libraries
#include <pangolin/pangolin.h>

// PPG-SLAM headers
#include "MapPoint.h"
#include "Frame.h"
#include "SE3.h"

// Forward declarations
class Tracking;
class System;
class Map;
class KeyFrame;
class MapPoint;
class MapEdge;
class MapColine;
class GeometricCamera;
struct KeyPointEx;
struct KeyEdge;

/**
 * @brief Viewer class for PPG-SLAM visualization system
 * 
 * This class implements a singleton pattern for the map viewer that displays:
 * - Real-time trajectory visualization
 * - Map points, colines, and edges
 * - Current frame with feature tracking
 * - Interactive 3D visualization using Pangolin
 * 
 * Key algorithms:
 * - Implements time-based fading for map elements
 * - Provides multiple view modes (camera view, top view)
 * - Supports step-by-step debugging
 * - Handles IMU-based coordinate system orientation
 */
class MSViewing
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Get singleton instance using Meyer's singleton pattern
     * @return Reference to the single MSViewing instance
     */
    static MSViewing& get()
    {
        static MSViewing single_instance;
        return single_instance;
    }

private:
    // Singleton pattern enforcement
    MSViewing() = default;
    ~MSViewing() { mbFinishRequested = true; }
    MSViewing(const MSViewing& other) = delete;
    MSViewing& operator=(const MSViewing& other) = delete;
    static MSViewing* singleViewing;

public:
    // === Core Viewer Control Functions ===
    
    /**
     * @brief Initialize and launch the viewer in a separate thread
     * @param pMap Pointer to the map to be visualized
     * 
     * Algorithm: Creates visualization thread and initializes OpenGL context
     */
    void Launch(Map* pMap);

    /**
     * @brief Main viewer loop running in separate thread
     * 
     * Algorithm: 
     * - Sets up Pangolin GUI with interactive controls
     * - Implements main rendering loop with frame rate control
     * - Handles view switching and camera following
     */
    void Run();

    /**
     * @brief Request viewer termination
     * 
     * Algorithm: Sets termination flag and waits for thread completion
     */
    void RequestFinish();

    // === Frame and Display Functions ===

    /**
     * @brief Update current frame data for visualization
     * @param F Current frame containing tracking results
     * 
     * Algorithm: Thread-safely copies frame data including keypoints,
     * edges, map points, and outlier flags
     */
    void UpdateFrame(Frame &F);

    /**
     * @brief Generate 2D visualization of current frame
     * @return OpenCV Mat containing annotated frame image
     * 
     * Algorithm:
     * - Overlays tracked features with color coding
     * - Draws colines and edges based on visibility flags
     * - Adds tracking statistics and state information
     */
    cv::Mat DrawFrame();

    /**
     * @brief Add text overlay with tracking information
     * @param im Input image
     * @param nState Current tracking state
     * @param imText Output image with text overlay
     * 
     * Algorithm: Displays keyframe count, map point count, and tracking status
     */
    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    // === 3D Map Visualization Functions ===

    /**
     * @brief Render all map points in 3D view
     * 
     * Algorithm:
     * - Uses time-based alpha blending for fading effect
     * - Filters out bad map points
     * - Implements temporal visibility decay
     */
    void DrawMapPoints();

    /**
     * @brief Render colinear point relationships
     * 
     * Algorithm:
     * - Draws lines connecting colinear map points
     * - Highlights middle points of colinear triplets
     * - Uses red color scheme for distinction
     */
    void DrawMapColines();

    /**
     * @brief Render map edges (line features)
     * 
     * Algorithm:
     * - Different rendering for current vs. historical edges
     * - Time-based fading for older edges
     * - Green highlight for currently tracked edges
     */
    void DrawMapEdges();

    /**
     * @brief Render keyframe poses and connectivity graph
     * @param bDrawKF Whether to draw keyframe coordinate frames
     * @param bDrawGraph Whether to draw covisibility graph
     * @param bDrawInertialGraph Whether to draw IMU connectivity
     * 
     * Algorithm:
     * - Draws coordinate axes for each keyframe
     * - Renders covisibility connections with weight thresholding
     * - Shows IMU sequential connections when available
     */
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph);

    /**
     * @brief Render current camera pose as wireframe pyramid
     * @param Twc OpenGL transformation matrix for camera pose
     * 
     * Algorithm: Draws camera frustum using OpenGL lines
     */
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);

    // === Camera Pose Management ===

    /**
     * @brief Update current camera pose for visualization
     * @param Tcw Camera-to-world transformation (SE3)
     * 
     * Algorithm: Thread-safely updates pose with inverse transformation
     */
    void SetCurrentCameraPose(const SE3f &Tcw);

    /**
     * @brief Convert current pose to OpenGL matrices
     * @param M Output camera transformation matrix
     * @param MOw Output world origin transformation
     * 
     * Algorithm: Converts SE3 to column-major OpenGL format
     */
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw);

    // === Trajectory Export Functions ===

    /**
     * @brief Save complete trajectory to file
     * @param filename Output file path
     * 
     * Algorithm:
     * - Exports all frame poses in TUM RGB-D format
     * - Handles IMU body frame transformations
     * - Sorts by frame ID for temporal consistency
     */
    void SaveTrajectory(const std::string &filename);

    /**
     * @brief Save keyframe trajectory to file
     * @param filename Output file path
     * 
     * Algorithm: Exports only keyframe poses with timestamps
     */
    void SaveKeyFrameTrajectory(const std::string &filename);

    // === Member Variables ===

    // Thread control
    std::atomic<bool> mbFinishRequested{false};
    std::atomic<bool> mbFinish{false};
    std::atomic<bool> mbStepByStep{false};
    std::atomic<bool> mbStep{false};

    // Visualization control flags
    std::atomic<bool> mbShowPoint{true};
    std::atomic<bool> mbShowColine{true};
    std::atomic<bool> mbShowEdge{true};
    std::atomic<bool> mbunFaded{false};

    // Core components
    Map* mpMap{nullptr};
    std::thread* mptViewer{nullptr};

    // Current frame visualization data
    cv::Mat mIm;                                    ///< Current frame image
    std::vector<KeyPointEx> mvIniKeys;              ///< Initial keypoints for initialization
    std::vector<int> mvIniMatches;                  ///< Initial matches for initialization
    std::vector<KeyPointEx> mvCurrentKeys;          ///< Current frame keypoints
    std::vector<KeyEdge> mvCurrentEdges;            ///< Current frame edges
    std::vector<bool> mvbOutliers;                  ///< Outlier flags for features
    std::vector<MapPoint*> mvpMapPoints;            ///< Associated map points
    std::vector<MapEdge*> mvpMapEdges;              ///< Associated map edges
    std::vector<MapPoint*> mvpLocalMap;             ///< Local map points

    // State information
    int mState{0};                                  ///< Current tracking state
    int mnTracked{0};                               ///< Number of tracked features
    unsigned long mnCurFrameID{0};                  ///< Current frame ID

    // Camera and pose
    SE3<float> mTcw;                               ///< Camera pose in world frame
    SE3f mCameraPose;                              ///< Current camera pose
    GeometricCamera* mpCamera{nullptr};             ///< Camera model

    // Thread synchronization
    std::mutex mMutex;                             ///< Mutex for thread-safe data access
};