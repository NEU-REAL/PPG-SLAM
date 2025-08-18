/**
 * @file MLPnPsolver.h
 * @brief ML-PnP solver for camera pose estimation
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "MapPoint.h"
#include "Frame.h"
#include "GeometricCamera.h"

/**
 * @brief Maximum Likelihood PnP solver with RANSAC
 */
class MLPnPsolver
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Type definitions
    typedef Eigen::Vector3d bearingVector_t;
    typedef std::vector<bearingVector_t, Eigen::aligned_allocator<bearingVector_t>> bearingVectors_t;
    typedef Eigen::Matrix2d cov2_mat_t;
    typedef Eigen::Matrix3d cov3_mat_t;
    
    /** @brief Array of 3D covariance matrices */
    typedef std::vector<cov3_mat_t, Eigen::aligned_allocator<cov3_mat_t>> cov3_mats_t;
    
    /** @brief 3D point in world coordinates */
    typedef Eigen::Vector3d point_t;
    
    /** @brief Array of 3D points */
    typedef std::vector<point_t, Eigen::aligned_allocator<point_t>> points_t;
    
    /** @brief Homogeneous 3D point representation */
    typedef Eigen::Vector4d point4_t;
    
    /** @brief Array of homogeneous 3D points */
    typedef std::vector<point4_t, Eigen::aligned_allocator<point4_t>> points4_t;
    
    /** @brief Rodrigues parameters for rotation representation */
    typedef Eigen::Vector3d rodrigues_t;
    
    /** @brief 3x3 rotation matrix */
    typedef Eigen::Matrix3d rotation_t;
    
    /** @brief 3x4 transformation matrix [R|t] */
    typedef Eigen::Matrix<double, 3, 4> transformation_t;
    typedef Eigen::Vector3d translation_t;

    // Constructor & Destructor
    /** Constructor with frame and correspondences */
    MLPnPsolver(const Frame &F, GeometricCamera* pCam, const vector<MapPoint *> &vpMapPointMatches);
    ~MLPnPsolver();

    // RANSAC configuration
    /** Set RANSAC parameters */
    void SetRansacParameters(double probability = 0.99, int minInliers = 8, int maxIterations = 300, 
                           int minSet = 6, float epsilon = 0.4, float th2 = 5.991);

    // Pose estimation
    /** Main RANSAC iteration */
    bool iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers, Eigen::Matrix4f &Tout);

private:
    // ==================== CORE ML-PNP ALGORITHM ====================
    
    /**
     * @brief Validate current pose hypothesis by checking inliers
     */
    void CheckInliers();
    
    /**
     * @brief Refine pose estimate using all inliers
     * @return True if refinement successful
     */
    bool Refine();
    
    /**
     * @brief Core ML-PnP pose computation
     * @param f Bearing vectors (normalized image rays)
     * @param p 3D points in world coordinates
     * @param covMats Covariance matrices for uncertainty modeling
     * @param indices Subset of correspondences to use
     * @param result Output transformation matrix [R|t]
     * 
     * Implements the maximum likelihood PnP algorithm with iterative optimization.
     */
    void computePose(const bearingVectors_t &f, const points_t &p, const cov3_mats_t &covMats,
                     const std::vector<int> &indices, transformation_t &result);

    // Core ML-PnP algorithms
    /** Gauss-Newton optimization */
    void mlpnp_gn(Eigen::VectorXd &x, const points_t &pts, 
                  const std::vector<Eigen::MatrixXd> &nullspaces,
                  const Eigen::SparseMatrix<double> Kll, bool use_cov);
    
    /** Compute residuals and Jacobians */
    void mlpnp_residuals_and_jacs(const Eigen::VectorXd &x, const points_t &pts,
                                  const std::vector<Eigen::MatrixXd> &nullspaces,
                                  Eigen::VectorXd &r, Eigen::MatrixXd &fjac, bool getJacs);
    
    /** Jacobian computation */
    void mlpnpJacs(const point_t &pt, const Eigen::Vector3d &nullspace_r,
                   const Eigen::Vector3d &nullspace_s, const rodrigues_t &w,
                   const translation_t &t, Eigen::MatrixXd &jacs);

    // Utility functions
    /** Rodrigues conversions */
    Eigen::Matrix3d rodrigues2rot(const Eigen::Vector3d &omega);
    Eigen::Vector3d rot2rodrigues(const Eigen::Matrix3d &R);

    // Data members
    
    // Input correspondences
    vector<MapPoint *> mvpMapPointMatches;    ///< 3D map points
    vector<cv::Point2f> mvP2D;               ///< 2D image points
    bearingVectors_t mvBearingVecs;          ///< Normalized bearing vectors
    points_t mvP3Dw;                         ///< 3D world coordinates
    vector<size_t> mvKeyPointIndices;        ///< Original feature indices
    
    // Current estimation state
    Eigen::Matrix3d mRi;                     ///< Current rotation estimate
    Eigen::Vector3d mti;                     ///< Current translation estimate  
    Eigen::Matrix4f mTcwi;                   ///< Current pose matrix
    vector<bool> mvbInliersi;                ///< Current inlier mask
    int mnInliersi;                          ///< Number of current inliers
    
    // RANSAC state management
    int mnIterations;                        ///< Current iteration count
    vector<bool> mvbBestInliers;             ///< Best inlier configuration
    int mnBestInliers;                       ///< Best inlier count
    Eigen::Matrix4f mBestTcw;                ///< Best pose estimate
    
    // Refined solution
    Eigen::Matrix4f mRefinedTcw;             ///< Refined pose after optimization
    vector<bool> mvbRefinedInliers;          ///< Refined inlier mask
    int mnRefinedInliers;                    ///< Number of refined inliers
    
    // Problem size
    int N;                                   ///< Total number of correspondences
    vector<size_t> mvAllIndices;             ///< Index array for random sampling
    
    // RANSAC parameters
    double mRansacProb;                      ///< Success probability
    int mRansacMinInliers;                   ///< Minimum inliers required
    int mRansacMaxIts;                       ///< Maximum iterations
    float mRansacEpsilon;                    ///< Expected inlier ratio
    float mRansacTh;                         ///< Pixel error threshold
    int mRansacMinSet;                       ///< Minimum set size
    float mfMaxError;                        ///< Maximum squared error
    
    GeometricCamera *mpCamera;               ///< Camera model
};

