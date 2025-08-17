/**
 * @file Optimizer.h
 * @brief Optimization framework for PPG-SLAM visual-inertial SLAM system
 * @details This file contains the Optimizer class which provides various optimization
 *          methods including bundle adjustment, pose optimization, and graph optimization
 *          for both visual and inertial measurements.
 */

#pragma once

// PPG-SLAM headers
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"

// Standard library
#include <math.h>
#include <map>
#include <set>
#include <vector>

// g2o optimization library headers
#include "g2o/core/sparse_block_matrix.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"

// Eigen
#include <Eigen/Core>

/**
 * @typedef KeyFrameAndPose
 * @brief Container for keyframe-pose associations used in loop closure
 * @details Maps KeyFrame pointers to g2o::Sim3 transformations with proper memory alignment
 */
typedef std::map<KeyFrame*, g2o::Sim3, std::less<KeyFrame*>, 
                 Eigen::aligned_allocator<std::pair<KeyFrame* const, g2o::Sim3>>> KeyFrameAndPose;

/**
 * @class Optimizer
 * @brief Static optimization methods for PPG-SLAM system
 * @details Provides comprehensive optimization functionality including:
 *          - Bundle adjustment (local and global)
 *          - Pose optimization for single frames
 *          - Inertial optimization for IMU integration
 *          - Essential graph optimization for loop closure
 *          - Sim3 optimization for scale estimation
 */
class Optimizer
{
public:

    // ==============================
    // BUNDLE ADJUSTMENT METHODS
    // ==============================

    /**
     * @brief Global Bundle Adjustment for the entire map
     * @param pMap Pointer to the map containing all keyframes and map points
     * @param nIterations Number of optimization iterations (default: 5)
     * @param nLoopKF Loop keyframe identifier for loop closure handling (default: 0)
     * @param pbStopFlag Pointer to stop flag for early termination (default: NULL)
     * @details Optimizes all keyframe poses and map point positions simultaneously.
     *          Uses robust kernels to handle outliers in feature observations.
     */
    static void GlobalBundleAdjustment(Map* pMap, int nIterations = 5, const unsigned long nLoopKF = 0,
                                       bool* pbStopFlag = NULL);

    /**
     * @brief Full Inertial Bundle Adjustment with IMU constraints
     * @param pMap Pointer to the map
     * @param its Number of optimization iterations
     * @param nLoopKF Loop keyframe identifier (default: 0)
     * @param pbStopFlag Stop flag for early termination (default: NULL)
     * @param bInit Flag indicating initialization phase (default: false)
     * @param priorG Prior weight for gyroscope bias (default: 1e2)
     * @param priorA Prior weight for accelerometer bias (default: 1e6)
     * @details Performs bundle adjustment including IMU measurements, optimizing
     *          poses, velocities, and sensor biases with inertial constraints.
     */
    static void FullInertialBA(Map* pMap, int its, const unsigned long nLoopKF = 0, 
                               bool* pbStopFlag = NULL, bool bInit = false, 
                               float priorG = 1e2, float priorA = 1e6);

    /**
     * @brief Local Bundle Adjustment around a keyframe
     * @param pKF Central keyframe for local optimization
     * @param pbStopFlag Stop flag for early termination
     * @param pMap Pointer to the map
     * @details Optimizes a local window of keyframes and map points around
     *          the specified keyframe for computational efficiency.
     */
    static void LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap);

    /**
     * @brief Local Inertial Bundle Adjustment with IMU constraints
     * @param pKF Central keyframe for local optimization
     * @param pbStopFlag Stop flag for early termination
     * @param pMap Pointer to the map
     * @param bLarge Flag for large-scale optimization (default: false)
     * @param bRecInit Flag for reinitialization (default: false)
     * @details Local bundle adjustment including IMU measurements and constraints.
     */
    static void LocalInertialBA(KeyFrame* pKF, bool* pbStopFlag, Map* pMap, 
                                bool bLarge = false, bool bRecInit = false);

    // ==============================
    // POSE OPTIMIZATION METHODS
    // ==============================

    /**
     * @brief Optimize pose of a single frame using visual observations
     * @param pFrame Pointer to the frame to optimize
     * @return Number of inlier observations after optimization
     * @details Optimizes the 6DoF pose of a frame given its map point observations.
     *          Uses robust estimation to handle outliers.
     */
    static int PoseOptimization(Frame* pFrame);

    /**
     * @brief Pose optimization for last keyframe with inertial constraints
     * @param pFrame Pointer to the frame to optimize
     * @param pMap Pointer to the map
     * @param bRecInit Flag for reinitialization (default: false)
     * @return Number of inlier observations after optimization
     * @details Optimizes pose including IMU preintegration constraints
     *          from the last keyframe.
     */
    static int PoseInertialOptimizationLastKeyFrame(Frame* pFrame, Map* pMap, bool bRecInit = false);

    /**
     * @brief Pose optimization for last frame with inertial constraints
     * @param pFrame Pointer to the frame to optimize
     * @param pMap Pointer to the map
     * @param bRecInit Flag for reinitialization (default: false)
     * @return Number of inlier observations after optimization
     * @details Optimizes pose including IMU preintegration constraints
     *          from the previous frame.
     */
    static int PoseInertialOptimizationLastFrame(Frame* pFrame, Map* pMap, bool bRecInit = false);

    // ==============================
    // GRAPH OPTIMIZATION METHODS
    // ==============================

    /**
     * @brief Essential graph optimization for loop closure
     * @param pMap Pointer to the map
     * @param pLoopKF Loop closure keyframe
     * @param pCurKF Current keyframe
     * @param NonCorrectedSim3 Non-corrected Sim3 poses before loop closure
     * @param CorrectedSim3 Corrected Sim3 poses after loop closure
     * @param LoopConnections Map of loop closure connections
     * @param bFixScale Fix scale flag (true for stereo: 6DoF, false for mono: 7DoF)
     * @details Optimizes the essential graph (pose graph) after loop closure detection.
     *          Propagates loop closure corrections throughout the pose graph.
     */
    static void OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const KeyFrameAndPose& NonCorrectedSim3,
                                       const KeyFrameAndPose& CorrectedSim3,
                                       const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections,
                                       const bool& bFixScale);

    /**
     * @brief Essential graph optimization for inertial loop closing (4DoF)
     * @param pMap Pointer to the map
     * @param pLoopKF Loop closure keyframe
     * @param pCurKF Current keyframe
     * @param NonCorrectedSim3 Non-corrected poses before loop closure
     * @param CorrectedSim3 Corrected poses after loop closure
     * @param LoopConnections Map of loop closure connections
     * @details 4DoF optimization for inertial systems where gravity direction
     *          and scale are constrained by IMU measurements.
     */
    static void OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                           const KeyFrameAndPose& NonCorrectedSim3,
                                           const KeyFrameAndPose& CorrectedSim3,
                                           const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections);

    // ==============================
    // SIM3 AND SCALE OPTIMIZATION
    // ==============================

    /**
     * @brief Sim3 optimization between two keyframes
     * @param pMap Pointer to the map
     * @param pKF1 First keyframe
     * @param pKF2 Second keyframe
     * @param vpMatches1 Vector of matched map points from first keyframe
     * @param g2oS12 Input/output Sim3 transformation from KF1 to KF2
     * @param th2 Threshold for outlier rejection (squared)
     * @param bFixScale Fix scale flag (true for stereo: SE3, false for mono: Sim3)
     * @param mAcumHessian Output accumulated Hessian matrix (7x7)
     * @param bAllPoints Use all points flag (default: false)
     * @return Number of inlier correspondences
     * @details Optimizes Sim3 transformation between two keyframes for loop closure.
     *          Can fix scale (SE3) for stereo or optimize scale (Sim3) for monocular.
     */
    static int OptimizeSim3(Map* pMap, KeyFrame* pKF1, KeyFrame* pKF2, 
                            std::vector<MapPoint*>& vpMatches1,
                            g2o::Sim3& g2oS12, const float th2, const bool bFixScale,
                            Eigen::Matrix<double, 7, 7>& mAcumHessian, const bool bAllPoints = false);

    // ==============================
    // INERTIAL OPTIMIZATION METHODS
    // ==============================

    /**
     * @brief Inertial-only optimization for scale and gravity alignment
     * @param pMap Pointer to the map
     * @param Rwg Output rotation from world to gravity-aligned frame
     * @param scale Output scale factor
     * @param bg Output gyroscope bias
     * @param ba Output accelerometer bias
     * @param covInertial Output covariance matrix of inertial parameters
     * @param bFixedVel Fix velocities during optimization (default: false)
     * @param bGauss Use Gauss-Newton instead of Levenberg-Marquardt (default: false)
     * @param priorG Prior weight for gyroscope bias (default: 1e2)
     * @param priorA Prior weight for accelerometer bias (default: 1e6)
     * @details Optimizes inertial parameters including scale, gravity direction,
     *          and sensor biases using only IMU measurements.
     */
    static void InertialOptimization(Map* pMap, Eigen::Matrix3d& Rwg, double& scale, 
                                     Eigen::Vector3d& bg, Eigen::Vector3d& ba, 
                                     bool bFixedVel = false, float priorG = 1e2, float priorA = 1e6);

    /**
     * @brief Simplified inertial optimization for scale and gravity
     * @param pMap Pointer to the map
     * @param Rwg Output rotation from world to gravity-aligned frame
     * @param scale Output scale factor
     * @details Simplified version that only estimates scale and gravity direction.
     */
    static void InertialOptimization(Map* pMap, Eigen::Matrix3d& Rwg, double& scale);

    // ==============================
    // UTILITY METHODS
    // ==============================

    /**
     * @brief Marginalize block elements using Schur complement
     * @param H Input Hessian matrix
     * @param start Starting index of block to marginalize
     * @param end Ending index of block to marginalize
     * @return Marginalized Hessian matrix
     * @details Performs Schur complement to marginalize specified block elements.
     *          Marginalized elements are filled with zeros.
     */
    static Eigen::MatrixXd Marginalize(const Eigen::MatrixXd& H, const int& start, const int& end);

    // Eigen memory alignment macro for proper vectorization
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
