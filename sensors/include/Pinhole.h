/**
 * @file Pinhole.h
 * @brief Standard pinhole camera model implementation
 * @details Classic perspective projection camera model with radial distortion
 */

#pragma once

#include <assert.h>
#include "GeometricCamera.h"

// Forward declaration
class TwoViewReconstruction;

/**
 * @brief Standard pinhole camera model with perspective projection
 * @details Implements classic pinhole camera model with optional radial distortion
 *          for standard perspective cameras
 */
class Pinhole : public GeometricCamera
{
public:
    /// Constructor with camera parameters
    Pinhole(const std::vector<float> &_vParameters, int width, int height, float fps);

    // ==================== PROJECTION FUNCTIONS ====================
    
    /// Project 3D point to image plane (double precision)
    Eigen::Vector2d project(const Eigen::Vector3d &v3D) override;
    
    /// Project 3D point to image plane (float precision)
    Eigen::Vector2f project(const Eigen::Vector3f &v3D) override;

    /// Unproject image point to 3D ray
    Eigen::Vector3f unproject(const Eigen::Vector2f &p2D) override;

    /// Compute projection Jacobian
    Eigen::Matrix<double, 2, 3> projectJac(const Eigen::Vector3d &v3D) override;

    // ==================== CAMERA PARAMETERS ====================
    
    /// Get camera intrinsic matrix as OpenCV Mat
    cv::Mat toK() override;
    
    /// Get distortion parameters as OpenCV Mat
    cv::Mat toD() override;
    
    /// Get camera intrinsic matrix as Eigen matrix
    Eigen::Matrix3f toK_() override;
    
    /// Get image width
    int imWidth() override;
    
    /// Get image height
    int imHeight() override;

    // ==================== SLAM FUNCTIONS ====================
    
    /// Two-view reconstruction from matched keypoints
    bool ReconstructWithTwoViews(const std::vector<KeyPointEx> &vKeys1, 
                               const std::vector<KeyPointEx> &vKeys2, 
                               const std::vector<int> &vMatches12,
                               Sophus::SE3f &T21, 
                               std::vector<cv::Point3f> &vP3D, 
                               std::vector<bool> &vbTriangulated) override;

    /// Check epipolar constraint between keypoints
    bool epipolarConstrain(const KeyPointEx &kp1, const KeyPointEx &kp2, 
                         const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12) override;

private:
    TwoViewReconstruction *mpTvr;  ///< Two-view reconstruction handler
};