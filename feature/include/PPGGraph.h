/**
 * @file PPGGraph.h
 * @brief Graph structures for PPG-SLAM system including KeyEdge, MapEdge and MapColine
 */

#pragma once

// ==================== SYSTEM INCLUDES ====================
#include <map>
#include <mutex>
#include <chrono>
#include <cassert>

// ==================== THIRD-PARTY INCLUDES ====================
#include <torch/torch.h>
#include <torch/script.h> 
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

// ==================== LOCAL INCLUDES ====================
#include "MapPoint.h"
#include "KeyFrame.h"

// ==================== FORWARD DECLARATIONS ====================
class KeyFrame;
class MapPoint;

/**
 * @class KeyEdge
 * @brief Edge connection between two keypoints within a single frame
 */
class KeyEdge
{
public:
    // ==================== CONSTRUCTORS ====================
    
    /// Default constructor - creates an invalid edge
    KeyEdge();
    
    /// Constructor with keypoint indices
    KeyEdge(const unsigned int& i0, const unsigned int& i1);

    // ==================== UTILITY FUNCTIONS ====================
    
    /// Get the other endpoint given one endpoint
    unsigned int theOtherPid(const unsigned int pid) const;

public:
    // ==================== MEMBER VARIABLES ====================
    
    unsigned int startIdx, endIdx;  ///< Indices of connected keypoints
    bool isBad;                     ///< Invalid edge flag
    float lscore;                   ///< Line score for validation and refinement
    float length;                   ///< Geometric length
};

/**
 * @class MapEdge
 * @brief 3D edge connection between two MapPoints across multiple frames
 */
class MapEdge
{
public:
    // ==================== CONSTRUCTORS ====================
    
    /// Constructor with start and end MapPoints
    MapEdge(MapPoint* ps, MapPoint* pe);

    // ==================== OPERATIONS ====================
    
    /// Get the other MapPoint given one MapPoint
    MapPoint* theOtherPt(MapPoint* pMP);
    
    /// Add observation of this edge in a keyframe
    void addObservation(KeyFrame* pKF, unsigned int keyId);
    
    /// Get all observations of this edge
    std::map<KeyFrame*, int> getObservations();

    /// Check and update validity based on geometric constraints
    void checkValid();
    
    /// Check if edge is marked as bad
    bool isBad();
    
public:
    // ==================== IDENTIFICATION ====================
    
    unsigned long int mnId;             ///< Unique identifier
    static unsigned long int mnNextId;  ///< Global ID counter

    // ==================== ENDPOINTS AND STATUS ====================
    
    MapPoint* mpMPs, * mpMPe;           ///< Start and end MapPoints
    bool mbBad, mbValid;                ///< Status flags

    // ==================== OPTIMIZATION AND VISUALIZATION ====================
    
    unsigned long int mnBALocalForKF;   ///< Local BA keyframe ID
    unsigned long int trackedFrameId;   ///< Tracking visualization frame ID
    static double viewCosTh;            ///< Viewing angle threshold
    std::chrono::steady_clock::time_point startTime;  ///< Creation timestamp

private:
    // ==================== THREAD-SAFE DATA ====================
    
    std::map<KeyFrame*, int> mObservations;  ///< Keyframe observations
    std::mutex mtxObs;                       ///< Observation mutex
    cv::Mat mDesc;                           ///< Edge descriptor
};

/**
 * @class MapColine
 * @brief Collinearity constraints between three MapPoints
 */
class MapColine
{
public:
    // ==================== CONSTRUCTORS ====================
    
    /// Constructor with three collinear MapPoints
    MapColine(MapPoint* pMPs, MapPoint* pMPm, MapPoint* pMPe);

    // ==================== OPERATIONS ====================
    
    /// Add weighted observation in a keyframe
    void addObservation(KeyFrame* pKF, float weight);
    
    /// Get all observations with weights
    std::map<KeyFrame*, int> getObservations();

    /// Calculate average weight of all observations
    float aveWeight();

    /// Check if constraint is marked as bad
    bool isBad();

public:
    // ==================== IDENTIFICATION ====================
    
    unsigned long int mnId;             ///< Unique identifier
    static unsigned long int mnNextId;  ///< Global ID counter

    // ==================== MAPPOINTS AND STATUS ====================
    
    MapPoint* mpMPs, * mpMPm, * mpMPe;  ///< Start, middle, and end MapPoints
    bool mbBad, mbValid;                ///< Status flags

private:
    // ==================== THREAD-SAFE DATA ====================
    
    std::map<KeyFrame*, int> mObservations;  ///< Keyframe observations with weights
    std::mutex mtxObs;                       ///< Observation mutex
    KeyFrame* mpFirstKF;                     ///< First observing keyframe
};