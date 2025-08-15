/**
 * @file Sim3Solver.h
 * @brief Sim3 transformation solver for loop closure
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "GeometricCamera.h"
#include "KeyFrame.h"

/**
 * @brief RANSAC-based Sim3 solver for keyframe alignment
 */
class Sim3Solver
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    /** Constructor with keyframe correspondences */
    Sim3Solver(KeyFrame* pKF1, KeyFrame* pKF2, GeometricCamera* pCam, 
               const std::vector<MapPoint*> &vpMatched12, const bool bFixScale = true,
               const std::vector<KeyFrame*> vpKeyFrameMatchedMP = std::vector<KeyFrame*>());

    // RANSAC configuration
    /** Set RANSAC parameters */
    void SetRansacParameters(double probability = 0.99, int minInliers = 6, int maxIterations = 300);

    // Main estimation
    /** Find best Sim3 transformation */
    Eigen::Matrix4f find(std::vector<bool> &vbInliers12, int &nInliers);
    
    /** RANSAC iteration */
    Eigen::Matrix4f iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);
    Eigen::Matrix4f iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, 
                           int &nInliers, bool &bConverge);

    // Result accessors
    /** Get results */
    Eigen::Matrix4f GetEstimatedTransformation();
    
    /**
     * @brief Get estimated rotation component
     * @return 3x3 rotation matrix
     */
    Eigen::Matrix3f GetEstimatedRotation();
    
    /**
     * @brief Get estimated translation component
     * @return 3D translation vector
     */
    Eigen::Vector3f GetEstimatedTranslation();
    
    /**
     * @brief Get estimated scale factor
     * @return Scale factor (1.0 if fixed scale mode)
     */
    float GetEstimatedScale();

protected:
    // ==================== CORE COMPUTATION METHODS ====================
    
    /**
     * @brief Compute centroid of 3D point sets
     * @param P Original point set
     * @param Pr Recentered point set (output)
     * @param C Computed centroid (output)
     */
    void ComputeCentroid(Eigen::Matrix3f &P, Eigen::Matrix3f &Pr, Eigen::Vector3f &C);

    /**
     * @brief Compute Sim3 transformation between point sets
     * @param P1 First point set (centered)
     * @param P2 Second point set (centered)
     * 
     * Uses Horn's method for closed-form similarity transformation computation.
     */
    void ComputeSim3(Eigen::Matrix3f &P1, Eigen::Matrix3f &P2);

    /**
     * @brief Check inliers for current transformation hypothesis
     * 
     * Projects points using current transformation and counts inliers
     * based on reprojection error threshold.
     */
    void CheckInliers();

    /**
     * @brief Project 3D world points to image coordinates
     * @param vP3Dw 3D points in world coordinates
     * @param vP2D Output 2D image coordinates
     * @param Tcw Camera pose transformation
     * @param pCamera Camera model for projection
     */
    void Project(const std::vector<Eigen::Vector3f> &vP3Dw, std::vector<Eigen::Vector2f> &vP2D, 
                 Eigen::Matrix4f Tcw, GeometricCamera* pCamera);
    
    /**
     * @brief Convert camera coordinates to image coordinates
     * @param vP3Dc 3D points in camera coordinates
     * @param vP2D Output 2D image coordinates
     * @param pCamera Camera model for projection
     */
    void FromCameraToImage(const std::vector<Eigen::Vector3f> &vP3Dc, 
                          std::vector<Eigen::Vector2f> &vP2D, GeometricCamera* pCamera);

protected:
    // ==================== INPUT DATA ====================
    
    KeyFrame* mpKF1;                          ///< First keyframe
    KeyFrame* mpKF2;                          ///< Second keyframe

    std::vector<Eigen::Vector3f> mvX3Dc1;     ///< 3D points in KF1 camera coords
    std::vector<Eigen::Vector3f> mvX3Dc2;     ///< 3D points in KF2 camera coords
    std::vector<MapPoint*> mvpMapPoints1;     ///< Map points from KF1
    std::vector<MapPoint*> mvpMapPoints2;     ///< Map points from KF2
    std::vector<MapPoint*> mvpMatches12;      ///< Matched map points KF1->KF2
    std::vector<size_t> mvnIndices1;          ///< Valid correspondence indices
    
    // Error thresholds per correspondence
    std::vector<size_t> mvSigmaSquare1;       ///< Squared sigma values for KF1
    std::vector<size_t> mvSigmaSquare2;       ///< Squared sigma values for KF2
    std::vector<size_t> mvnMaxError1;         ///< Max error thresholds for KF1
    std::vector<size_t> mvnMaxError2;         ///< Max error thresholds for KF2

    int N;                                    ///< Total correspondences
    int mN1;                                  ///< Valid correspondences count

    // ==================== ESTIMATION STATE ====================
    
    // Current hypothesis
    Eigen::Matrix3f mR12i;                    ///< Current rotation estimate
    Eigen::Vector3f mt12i;                    ///< Current translation estimate
    float ms12i;                              ///< Current scale estimate
    Eigen::Matrix4f mT12i;                    ///< Current transformation matrix
    Eigen::Matrix4f mT21i;                    ///< Inverse transformation
    std::vector<bool> mvbInliersi;            ///< Current inlier mask
    int mnInliersi;

    // RANSAC state
    int mnIterations;
    std::vector<bool> mvbBestInliers;
    int mnBestInliers;
    Eigen::Matrix4f mBestT12;
    Eigen::Matrix3f mBestRotation;
    Eigen::Vector3f mBestTranslation;
    float mBestScale;

    // Configuration
    bool mbFixScale;
    std::vector<size_t> mvAllIndices;

    // Reprojection data
    std::vector<Eigen::Vector2f> mvP1im1;
    std::vector<Eigen::Vector2f> mvP2im2;

    // RANSAC parameters
    double mRansacProb;
    int mRansacMinInliers;
    int mRansacMaxIts;
    float mTh;
    float mSigma2;

    GeometricCamera* mpCamera;
};