#pragma once

/**
 * @file TwoViewReconstruction.h
 * @brief Two-view structure from motion reconstruction
 * @details Based on ORB-SLAM3 implementation
 */

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <unordered_set>
#include <sophus/se3.hpp>
#include "GeometricCamera.h"

class KeyPointEx;

/**
 * @brief Two-view structure from motion reconstruction
 */
class TwoViewReconstruction
{
    typedef std::pair<int, int> Match;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // ==================== CONSTRUCTOR ====================
    
    TwoViewReconstruction(const Eigen::Matrix3f &k, float sigma = 1.0, int iterations = 200);

    // ==================== MAIN INTERFACE ====================

    /// Reconstruct motion and structure from two views
    bool Reconstruct(const std::vector<KeyPointEx> &vKeys1, const std::vector<KeyPointEx> &vKeys2, const std::vector<int> &vMatches12,
                     Sophus::SE3f &T21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

private:
    // ==================== RANSAC ESTIMATION ====================

    void FindHomography(std::vector<bool> &vbMatchesInliers, float &score, Eigen::Matrix3f &H21);
    void FindFundamental(std::vector<bool> &vbInliers, float &score, Eigen::Matrix3f &F21);

    // ==================== MATRIX COMPUTATION ====================

    Eigen::Matrix3f ComputeH21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
    Eigen::Matrix3f ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);

    // ==================== MODEL VALIDATION ====================

    float CheckHomography(const Eigen::Matrix3f &H21, const Eigen::Matrix3f &H12, std::vector<bool> &vbMatchesInliers, float sigma);
    float CheckFundamental(const Eigen::Matrix3f &F21, std::vector<bool> &vbMatchesInliers, float sigma);

    // ==================== RECONSTRUCTION METHODS ====================

    bool ReconstructF(std::vector<bool> &vbMatchesInliers, Eigen::Matrix3f &F21, Eigen::Matrix3f &K,
                      Sophus::SE3f &T21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    bool ReconstructH(std::vector<bool> &vbMatchesInliers, Eigen::Matrix3f &H21, Eigen::Matrix3f &K,
                      Sophus::SE3f &T21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    // ==================== UTILITY FUNCTIONS ====================

    void Normalize(const std::vector<KeyPointEx> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, Eigen::Matrix3f &T);

    int CheckRT(const Eigen::Matrix3f &R, const Eigen::Vector3f &t, const std::vector<KeyPointEx> &vKeys1, const std::vector<KeyPointEx> &vKeys2,
                const std::vector<Match> &vMatches12, std::vector<bool> &vbMatchesInliers,
                const Eigen::Matrix3f &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax);

    void DecomposeE(const Eigen::Matrix3f &E, Eigen::Matrix3f &R1, Eigen::Matrix3f &R2, Eigen::Vector3f &t);

private:
    // ==================== MEMBER VARIABLES ====================

    std::vector<KeyPointEx> mvKeys1;   ///< Keypoints from reference frame
    std::vector<KeyPointEx> mvKeys2;   ///< Keypoints from current frame

    std::vector<Match> mvMatches12;    ///< Matches between frames
    std::vector<bool> mvbMatched1;     ///< Match flags

    Eigen::Matrix3f mK;                ///< Camera intrinsic matrix
    float mSigma, mSigma2;             ///< RANSAC parameters
    int mMaxIterations;                ///< Maximum RANSAC iterations
    std::vector<std::vector<size_t>> mvSets;  ///< RANSAC point sets
};