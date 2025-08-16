#pragma once

// ==================================================================================
// G2O VERTEX DEFINITIONS FOR PPG-SLAM OPTIMIZATION
// ==================================================================================
// This file contains various vertex types used in the pose graph optimization.
// Vertices represent optimizable parameters in the graph optimization framework.
//
// Vertex categories:
// 1. POSE VERTICES: Camera and IMU pose representations (6DoF and 4DoF)
// 2. VELOCITY VERTICES: Velocity parameters for inertial optimization
// 3. BIAS VERTICES: IMU bias parameters (gyroscope and accelerometer)
// 4. GEOMETRY VERTICES: Gravity direction and scale parameters
// 5. SIM3 VERTICES: Similarity transformation vertices for loop closure
// ==================================================================================

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_multi_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sim3/sim3.h"
#include "g2o/types/slam3d/se3quat.h"

#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Frame.h>
#include <KeyFrame.h>

#include <math.h>

// Forward declarations
class KeyFrame;
class Frame;
class GeometricCamera;

// ==================================================================================
// COMMON TYPE DEFINITIONS
// ==================================================================================

// Commonly used Eigen matrix and vector types
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;

// ==================================================================================
// UTILITY FUNCTIONS
// ==================================================================================

/**
 * @brief Normalizes a rotation matrix using SVD decomposition
 * @param R Input rotation matrix
 * @return Normalized rotation matrix
 */
template<typename T = double>
Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T,3,3> &R) 
{
    Eigen::JacobiSVD<Eigen::Matrix<T,3,3>> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

/**
 * @brief Exponential map for SO(3) - converts axis-angle to rotation matrix
 * @param x, y, z Axis-angle representation components
 * @return Rotation matrix
 */
Eigen::Matrix3d ExpSO3(const double x, const double y, const double z);

/**
 * @brief Exponential map for SO(3) - converts axis-angle vector to rotation matrix
 * @param w Axis-angle vector
 * @return Rotation matrix
 */
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);

/**
 * @brief Logarithm map for SO(3) - converts rotation matrix to axis-angle
 * @param R Rotation matrix
 * @return Axis-angle vector
 */
Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);

/**
 * @brief Inverse right Jacobian for SO(3)
 * @param x, y, z Axis-angle components
 * @return Inverse right Jacobian matrix
 */
Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z);

/**
 * @brief Inverse right Jacobian for SO(3)
 * @param v Axis-angle vector
 * @return Inverse right Jacobian matrix
 */
Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v);

/**
 * @brief Right Jacobian for SO(3)
 * @param x, y, z Axis-angle components
 * @return Right Jacobian matrix
 */
Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z);

/**
 * @brief Right Jacobian for SO(3)
 * @param v Axis-angle vector
 * @return Right Jacobian matrix
 */
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v);

/**
 * @brief Skew-symmetric matrix operator
 * @param w 3D vector
 * @return 3x3 skew-symmetric matrix
 */
Eigen::Matrix3d Skew(const Eigen::Vector3d &w);

// ==================================================================================
// IMU-CAMERA POSE CLASS
// ==================================================================================

/**
 * @brief Combined IMU-Camera pose representation
 * Handles the relationship between IMU and camera coordinate frames
 */

class ImuCamPose
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructors
    ImuCamPose() {}
    ImuCamPose(KeyFrame* pKF, GeometricCamera* pCam);
    ImuCamPose(Frame* pF, GeometricCamera* pCam);
    ImuCamPose(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, KeyFrame* pKF, GeometricCamera* pCam);

    // Update functions
    void Update(const double *pu);  // Update in the IMU reference frame
    void UpdateW(const double *pu); // Update in the world reference frame
    
    // Projection functions
    Eigen::Vector2d Project(const Eigen::Vector3d &Xw) const; // Monocular projection
    bool isDepthPositive(const Eigen::Vector3d &Xw) const;     // Check if point is in front of camera

public:
    // IMU pose in world frame
    Eigen::Matrix3d Rwb;  // Rotation from body to world
    Eigen::Vector3d twb;  // Translation from body to world

    // Camera pose in world frame
    Eigen::Matrix3d Rcw;  // Rotation from world to camera
    Eigen::Vector3d tcw;  // Translation from world to camera
    
    // IMU-Camera calibration
    Eigen::Matrix3d Rcb, Rbc;  // Rotation between camera and body frames
    Eigen::Vector3d tcb, tbc;  // Translation between camera and body frames
    
    double bf;                 // Baseline times focal length (for stereo)
    GeometricCamera* pCamera;  // Camera model pointer

    // For pose graph 4DoF optimization
    Eigen::Matrix3d Rwb0;  // Initial body rotation
    Eigen::Matrix3d DR;    // Incremental rotation

    int its;  // Iteration counter for normalization
};

// ==================================================================================
// VERTEX CLASSES FOR G2O OPTIMIZATION
// ==================================================================================

/**
 * @brief 6DoF pose vertex for full IMU pose optimization
 * Optimizable parameters: 3D rotation + 3D translation in IMU frame
 */

class VertexPose : public g2o::BaseVertex<6, ImuCamPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructors
    VertexPose() {}
    VertexPose(KeyFrame* pKF, GeometricCamera* pCam) {
        setEstimate(ImuCamPose(pKF, pCam));
    }
    VertexPose(Frame* pF, GeometricCamera* pCam) {
        setEstimate(ImuCamPose(pF, pCam));
    }

    // g2o interface (not implemented for this application)
    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_) {
        _estimate.Update(update_);
        updateCache();
    }
};

/**
 * @brief 4DoF pose vertex for pose graph optimization
 * Optimizable parameters: yaw rotation + 3D translation only
 */
class VertexPose4DoF : public g2o::BaseVertex<4, ImuCamPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructors
    VertexPose4DoF() {}
    VertexPose4DoF(KeyFrame* pKF, GeometricCamera* pCam) {
        setEstimate(ImuCamPose(pKF, pCam));
    }
    VertexPose4DoF(Frame* pF, GeometricCamera* pCam) {
        setEstimate(ImuCamPose(pF, pCam));
    }
    VertexPose4DoF(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, KeyFrame* pKF, GeometricCamera* pCam) {
        setEstimate(ImuCamPose(_Rwc, _twc, pKF, pCam));
    }

    // g2o interface
    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_) {
        double update6DoF[6];
        update6DoF[0] = 0;           // No roll update
        update6DoF[1] = 0;           // No pitch update  
        update6DoF[2] = update_[0];  // Yaw update
        update6DoF[3] = update_[1];  // X translation
        update6DoF[4] = update_[2];  // Y translation
        update6DoF[5] = update_[3];  // Z translation
        _estimate.UpdateW(update6DoF);
        updateCache();
    }
};

/**
 * @brief Velocity vertex for inertial optimization
 * Optimizable parameters: 3D velocity vector
 */

class VertexVelocity : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructors
    VertexVelocity() {}
    VertexVelocity(KeyFrame* pKF) {
        setEstimate(pKF->GetVelocity().cast<double>());
    }
    VertexVelocity(Frame* pF) {
        setEstimate(pF->GetVelocity().cast<double>());
    }

    // g2o interface
    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_) {
        Eigen::Vector3d uv;
        uv << update_[0], update_[1], update_[2];
        setEstimate(estimate() + uv);
    }
};

/**
 * @brief Gyroscope bias vertex for inertial optimization  
 * Optimizable parameters: 3D gyroscope bias vector
 */
class VertexGyroBias : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructors
    VertexGyroBias() {}
    VertexGyroBias(KeyFrame* pKF) {
        setEstimate(pKF->GetGyroBias().cast<double>());
    }
    VertexGyroBias(Frame* pF) {
        Eigen::Vector3d bg;
        bg << pF->mImuBias.bwx, pF->mImuBias.bwy, pF->mImuBias.bwz;
        setEstimate(bg);
    }

    // g2o interface
    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_) {
        Eigen::Vector3d ubg;
        ubg << update_[0], update_[1], update_[2];
        setEstimate(estimate() + ubg);
    }
};

/**
 * @brief Accelerometer bias vertex for inertial optimization
 * Optimizable parameters: 3D accelerometer bias vector  
 */


class VertexAccBias : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructors
    VertexAccBias() {}
    VertexAccBias(KeyFrame* pKF) {
        setEstimate(pKF->GetAccBias().cast<double>());
    }
    VertexAccBias(Frame* pF) {
        Eigen::Vector3d ba;
        ba << pF->mImuBias.bax, pF->mImuBias.bay, pF->mImuBias.baz;
        setEstimate(ba);
    }

    // g2o interface
    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_) {
        Eigen::Vector3d uba;
        uba << update_[0], update_[1], update_[2];
        setEstimate(estimate() + uba);
    }
};

// ==================================================================================
// GRAVITY DIRECTION OPTIMIZATION
// ==================================================================================

/**
 * @brief Gravity direction representation for optimization
 * Parameterizes gravity direction using rotation matrix
 */

class GDirection
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructors
    GDirection() {}
    GDirection(Eigen::Matrix3d pRwg) : Rwg(pRwg) {}
    
    /**
     * @brief Update gravity direction with incremental rotation
     * @param pu Update parameters [roll, pitch] (yaw is fixed to 0)
     */
    void Update(const double *pu) {
        Rwg = Rwg * ExpSO3(pu[0], pu[1], 0.0);
    }

public:
    Eigen::Matrix3d Rwg, Rgw;  // Rotation matrices between world and gravity frames
    int its;                   // Iteration counter
};

/**
 * @brief Gravity direction vertex (2DoF optimization)
 * Optimizable parameters: gravity direction (roll and pitch only)
 */
class VertexGDir : public g2o::BaseVertex<2, GDirection>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructors
    VertexGDir() {}
    VertexGDir(Eigen::Matrix3d pRwg) {
        setEstimate(GDirection(pRwg));
    }

    // g2o interface
    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_) {
        _estimate.Update(update_);
        updateCache();
    }
};

/**
 * @brief Scale factor vertex for similarity transformation
 * Optimizable parameters: single scale factor (logarithmic parameterization)
 */

class VertexScale : public g2o::BaseVertex<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructors
    VertexScale() {
        setEstimate(1.0);
    }
    VertexScale(double ps) {
        setEstimate(ps);
    }

    // g2o interface
    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() {
        setEstimate(1.0);
    }

    virtual void oplusImpl(const double *update_) {
        setEstimate(estimate() * exp(*update_));
    }
};

// ==================================================================================
// SIM3 TRANSFORMATION VERTEX
// ==================================================================================

/**
 * @brief Sim3 transformation vertex for loop closure optimization
 * Optimizable parameters: 7DoF similarity transformation (rotation + translation + scale)
 */
class VertexSim3Expmap : public g2o::BaseVertex<7, g2o::Sim3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructor
    VertexSim3Expmap();
    
    // g2o interface
    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() {
        _estimate = g2o::Sim3();
    }

    virtual void oplusImpl(const double* update_) {
        Eigen::Map<g2o::Vector7> update(const_cast<double*>(update_));

        if (_fix_scale)
            update[6] = 0;

        g2o::Sim3 s(update);
        setEstimate(s * estimate());
    }

public:
    GeometricCamera* pCamera1, *pCamera2;  // Camera models for stereo/multi-camera systems
    bool _fix_scale;                       // Flag to fix scale during optimization
};

