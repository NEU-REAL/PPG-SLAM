/**
 * @file IMU.h
 * @brief IMU sensor processing and preintegration for visual-inertial SLAM
 * @details Based on ORB-SLAM3 IMU implementation
 */

#pragma once

#include <vector>
#include <utility>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "SE3.h"

namespace IMU
{

// ==================== CONSTANTS ====================
const float GRAVITY_VALUE = 9.81;

// ==================== IMU DATA STRUCTURES ====================

/**
 * @brief IMU measurement containing accelerometer, gyroscope data and timestamp
 */
class Point
{
public:
    Point(const float &acc_x, const float &acc_y, const float &acc_z,
          const float &ang_vel_x, const float &ang_vel_y, const float &ang_vel_z,
          const double &timestamp) 
        : a(acc_x, acc_y, acc_z), w(ang_vel_x, ang_vel_y, ang_vel_z), t(timestamp) {}
    
    Point(const cv::Point3f Acc, const cv::Point3f Gyro, const double &timestamp) 
        : a(Acc.x, Acc.y, Acc.z), w(Gyro.x, Gyro.y, Gyro.z), t(timestamp) {}

public:
    Eigen::Vector3f a;  ///< Accelerometer measurement
    Eigen::Vector3f w;  ///< Gyroscope measurement  
    double t;           ///< Timestamp
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief IMU bias parameters for accelerometer and gyroscope
 */
class Bias
{
public:
    Bias() : bax(0), bay(0), baz(0), bwx(0), bwy(0), bwz(0) {}
    
    Bias(const float &b_acc_x, const float &b_acc_y, const float &b_acc_z,
         const float &b_ang_vel_x, const float &b_ang_vel_y, const float &b_ang_vel_z) 
        : bax(b_acc_x), bay(b_acc_y), baz(b_acc_z), 
          bwx(b_ang_vel_x), bwy(b_ang_vel_y), bwz(b_ang_vel_z) {}
    
    /// Copy bias values from another bias object
    void CopyFrom(Bias &b);

public:
    float bax, bay, baz;  ///< Accelerometer biases
    float bwx, bwy, bwz;  ///< Gyroscope biases
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief IMU calibration parameters including camera-IMU transformation and noise parameters
 */
class Calib
{
public:
    Calib(const SE3<float> &Tbc, const float &ng, const float &na, 
          const float &ngw, const float &naw, const float &freq);
    
    Calib(const Calib &calib);
    Calib() : mbIsSet(false) {}

    /// Set calibration parameters
    void Set(const SE3<float> &sophTbc, const float &ng, const float &na, 
             const float &ngw, const float &naw);

public:
    SE3<float> mTcb;           ///< Transformation from camera to body (IMU)
    SE3<float> mTbc;           ///< Transformation from body (IMU) to camera
    Eigen::DiagonalMatrix<float, 6> Cov, CovWalk;  ///< Noise covariance matrices
    bool mbIsSet;                      ///< Whether calibration is set
    float mfFreq;                      ///< IMU frequency
    float mImuPer;                     ///< IMU period (1/frequency)
};

// ==================== INTEGRATION CLASSES ====================

/**
 * @brief Integration of single gyroscope measurement
 */
class IntegratedRotation
{
public:
    IntegratedRotation() {}
    IntegratedRotation(const Eigen::Vector3f &angVel, const Bias &imuBias, const float &time);

public:
    float deltaT;           ///< Integration time interval
    Eigen::Matrix3f deltaR; ///< Rotation increment
    Eigen::Matrix3f rightJ; ///< Right Jacobian for SO(3)
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief IMU preintegration for visual-inertial optimization
 */
class Preintegrated
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Preintegrated(const Bias &b_, const Calib *calib);
    Preintegrated(Preintegrated *pImuPre);
    Preintegrated() {}
    ~Preintegrated() {}

    // ==================== INITIALIZATION & MANAGEMENT ====================
    
    /// Copy data from another preintegration object
    void CopyFrom(Preintegrated *pImuPre);
    
    /// Initialize with given bias
    void Initialize(const Bias &b_);
    
    /// Reintegrate all measurements with updated bias
    void Reintegrate();
    
    /// Merge with previous preintegration measurements
    void MergePrevious(Preintegrated *pPrev);

    // ==================== MEASUREMENT INTEGRATION ====================
    
    /// Integrate new IMU measurement
    void IntegrateNewMeasurement(const Eigen::Vector3f &acceleration, 
                               const Eigen::Vector3f &angVel, const float &dt);
    
    /// Set new bias and compute bias difference
    void SetNewBias(const Bias &bu_);
    
    /// Get bias difference
    IMU::Bias GetDeltaBias(const Bias &b_);

    // ==================== DELTA COMPUTATION (WITH BIAS CORRECTION) ====================
    
    /// Get rotation increment with bias correction
    Eigen::Matrix3f GetDeltaRotation(const Bias &b_);
    
    /// Get velocity increment with bias correction  
    Eigen::Vector3f GetDeltaVelocity(const Bias &b_);
    
    /// Get position increment with bias correction
    Eigen::Vector3f GetDeltaPosition(const Bias &b_);

    // ==================== UPDATED VALUES (WITH CURRENT BIAS) ====================
    
    /// Get rotation increment with updated bias
    Eigen::Matrix3f GetUpdatedDeltaRotation();
    
    /// Get velocity increment with updated bias
    Eigen::Vector3f GetUpdatedDeltaVelocity();
    
    /// Get position increment with updated bias
    Eigen::Vector3f GetUpdatedDeltaPosition();

    // ==================== ORIGINAL VALUES (WITH ORIGINAL BIAS) ====================
    
    /// Get original rotation increment
    Eigen::Matrix3f GetOriginalDeltaRotation();
    
    /// Get original velocity increment
    Eigen::Vector3f GetOriginalDeltaVelocity();
    
    /// Get original position increment
    Eigen::Vector3f GetOriginalDeltaPosition();

    // ==================== BIAS ACCESS ====================
    
    /// Get bias difference vector
    Eigen::Matrix<float, 6, 1> GetDeltaBias();
    
    /// Get original bias
    Bias GetOriginalBias();
    
    /// Get updated bias
    Bias GetUpdatedBias();

    // ==================== DEBUG FUNCTIONS ====================
    
    /// Print all measurements for debugging
    void printMeasurements() const;

public:
    // ==================== INTEGRATION RESULTS ====================
    
    float dT;                           ///< Total integration time
    Eigen::Matrix<float, 15, 15> C;     ///< Covariance matrix
    Eigen::Matrix<float, 15, 15> Info;  ///< Information matrix
    Eigen::DiagonalMatrix<float, 6> Nga, NgaWalk;  ///< Noise matrices

    // Values for the original bias (when integration was computed)
    Bias b;                             ///< Original bias
    Eigen::Matrix3f dR;                 ///< Rotation increment
    Eigen::Vector3f dV, dP;             ///< Velocity and position increments
    Eigen::Matrix3f JRg, JVg, JVa, JPg, JPa;  ///< Jacobians w.r.t. bias
    Eigen::Vector3f avgA, avgW;         ///< Average acceleration and angular velocity

private:
    // ==================== INTERNAL STATE ====================
    
    Bias bu;                            ///< Updated bias
    Eigen::Matrix<float, 6, 1> db;      ///< Bias difference

    /// Internal measurement storage
    struct integrable
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        integrable() {}
        integrable(const Eigen::Vector3f &a_, const Eigen::Vector3f &w_, const float &t_) 
            : a(a_), w(w_), t(t_) {}
        
        Eigen::Vector3f a, w;  ///< Acceleration and angular velocity
        float t;               ///< Time interval
    };

    std::vector<integrable> mvMeasurements;  ///< Stored measurements
    std::mutex mMutex;                       ///< Thread safety
};

// ==================== LIE ALGEBRA UTILITY FUNCTIONS ====================

/// Compute right Jacobian for SO(3) group
Eigen::Matrix3f RightJacobianSO3(const float &x, const float &y, const float &z);
Eigen::Matrix3f RightJacobianSO3(const Eigen::Vector3f &v);

/// Compute inverse right Jacobian for SO(3) group  
Eigen::Matrix3f InverseRightJacobianSO3(const float &x, const float &y, const float &z);
Eigen::Matrix3f InverseRightJacobianSO3(const Eigen::Vector3f &v);

/// Normalize rotation matrix using SVD
Eigen::Matrix3f NormalizeRotation(const Eigen::Matrix3f &R);

} // namespace IMU