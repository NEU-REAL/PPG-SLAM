/**
 * @file GeometricCamera.h
 * @brief Abstract geometric camera interface for SLAM
 */

#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Geometry>

// Forward declarations
class TwoViewReconstruction;

/**
 * @brief Extended keypoint with SLAM-specific information
 */

class KeyPointEx
{
public:
    KeyPointEx() {}
    KeyPointEx(float x_, float y_, float sc) : mPos(x_, y_), mfScore(sc), mbOut(true) {}
    
    /// Update keypoint position if new score is better
    void updatePreturb(float deltX, float deltY, float sc);

public:
    Eigen::Vector2f mPos, mPosUn;      ///< Distorted and undistorted positions
    float mfScore;                     ///< Detection score
    std::vector<unsigned int> mvConnected;  ///< Connected line indices
    std::vector<std::pair<unsigned int, unsigned int>> mvColine;  ///< Collinear point pairs
    bool mbOut;                        ///< Whether point is outside image bounds
};

/**
 * @brief Abstract geometric camera model interface
 */
class GeometricCamera
{
public:
    GeometricCamera(const std::vector<float>& parameters, int width, int height, float fps);

    // ==================== PROJECTION FUNCTIONS ====================
    virtual Eigen::Vector2d project(const Eigen::Vector3d& v3D) = 0;
    virtual Eigen::Vector2f project(const Eigen::Vector3f& v3D) = 0;
    virtual Eigen::Vector3f unproject(const Eigen::Vector2f& p2D) = 0;
    virtual Eigen::Matrix<double, 2, 3> projectJac(const Eigen::Vector3d& v3D) = 0;

    // ==================== CAMERA PARAMETERS ====================
    virtual cv::Mat toK() = 0;
    virtual cv::Mat toD() = 0;
    virtual Eigen::Matrix3f toK_() = 0;
    virtual int imWidth() = 0;
    virtual int imHeight() = 0;

    // ==================== SLAM FUNCTIONS ====================
    virtual bool ReconstructWithTwoViews(const std::vector<KeyPointEx>& vKeys1, 
                                        const std::vector<KeyPointEx>& vKeys2,
                                        const std::vector<int>& vMatches12, 
                                        Sophus::SE3f& T21, 
                                        std::vector<cv::Point3f>& vP3D,
                                        std::vector<bool>& vbTriangulated) = 0;

    virtual bool epipolarConstrain(const KeyPointEx& kp1, const KeyPointEx& kp2, 
                                 const Eigen::Matrix3f& R12, const Eigen::Vector3f& t12) = 0;

    /// Check if point is within image bounds
    bool IsInImage(const float& x, const float& y) const;

    // ==================== CAMERA TYPES ====================
    static const unsigned int CAM_PINHOLE = 0;
    static const unsigned int CAM_FISHEYE = 1;

    // ==================== GRID PARAMETERS ====================
    static const int FRAME_GRID_ROWS = 48;
    static const int FRAME_GRID_COLS = 64;

protected:
    /// Initialize image bounds and grid parameters
    void InitializeImageBounds();

public:
    std::vector<float> mvParameters;   ///< Camera parameters
    int mnWidth, mnHeight;             ///< Image dimensions  
    float mfFps;                       ///< Frame rate
    unsigned int mnId;                 ///< Camera ID
    unsigned int mnType;               ///< Camera type

    // Grid parameters for feature matching acceleration
    int mnGridCols, mnGridRows;
    float mfGridElementWidthInv, mfGridElementHeightInv;
    int mnMinX, mnMinY, mnMaxX, mnMaxY;
};