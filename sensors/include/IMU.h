#pragma once

#include <vector>
#include <utility>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <mutex>

namespace IMU
{

const float GRAVITY_VALUE = 9.81;

// IMU measurement (gyro, accelerometer and timestamp)
class Point
{
public:
    Point(const float &acc_x, const float &acc_y, const float &acc_z,
            const float &ang_vel_x, const float &ang_vel_y, const float &ang_vel_z,
            const double &timestamp) : a(acc_x, acc_y, acc_z), w(ang_vel_x, ang_vel_y, ang_vel_z), t(timestamp) {}
    Point(const cv::Point3f Acc, const cv::Point3f Gyro, const double &timestamp) : a(Acc.x, Acc.y, Acc.z), w(Gyro.x, Gyro.y, Gyro.z), t(timestamp) {}

public:
    Eigen::Vector3f a;
    Eigen::Vector3f w;
    double t;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// IMU biases (gyro and accelerometer)
class Bias
{

public:
    Bias() : bax(0), bay(0), baz(0), bwx(0), bwy(0), bwz(0) {}
    Bias(const float &b_acc_x, const float &b_acc_y, const float &b_acc_z,
            const float &b_ang_vel_x, const float &b_ang_vel_y, const float &b_ang_vel_z) : bax(b_acc_x), bay(b_acc_y), baz(b_acc_z), bwx(b_ang_vel_x), bwy(b_ang_vel_y), bwz(b_ang_vel_z) {}
    void CopyFrom(Bias &b);

public:
    float bax, bay, baz;
    float bwx, bwy, bwz;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// IMU calibration (Tbc, Tcb, noise)
class Calib
{
public:
    Calib(const Sophus::SE3<float> &Tbc, const float &ng, const float &na, const float &ngw, const float &naw, const float &freq)
    {
        Set(Tbc, ng, na, ngw, naw);
        mfFreq = freq;
        mImuPer = 1./mfFreq;
    }

    Calib(const Calib &calib);
    Calib() { mbIsSet = false; }

    // void Set(const cv::Mat &cvTbc, const float &ng, const float &na, const float &ngw, const float &naw);
    void Set(const Sophus::SE3<float> &sophTbc, const float &ng, const float &na, const float &ngw, const float &naw);

public:
    // Sophus/Eigen implementation
    Sophus::SE3<float> mTcb;
    Sophus::SE3<float> mTbc;
    Eigen::DiagonalMatrix<float, 6> Cov, CovWalk;
    bool mbIsSet;
    float mfFreq;
    float mImuPer;
};

// Integration of 1 gyro measurement
class IntegratedRotation
{
public:
    IntegratedRotation() {}
    IntegratedRotation(const Eigen::Vector3f &angVel, const Bias &imuBias, const float &time);

public:
    float deltaT; // integration time
    Eigen::Matrix3f deltaR;
    Eigen::Matrix3f rightJ; // right jacobian
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Preintegration of Imu Measurements
class Preintegrated
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Preintegrated(const Bias &b_, const Calib *calib);
    Preintegrated(Preintegrated *pImuPre);
    Preintegrated() {}
    ~Preintegrated() {}
    void CopyFrom(Preintegrated *pImuPre);
    void Initialize(const Bias &b_);
    void IntegrateNewMeasurement(const Eigen::Vector3f &acceleration, const Eigen::Vector3f &angVel, const float &dt);
    void Reintegrate();
    void MergePrevious(Preintegrated *pPrev);
    void SetNewBias(const Bias &bu_);
    IMU::Bias GetDeltaBias(const Bias &b_);

    Eigen::Matrix3f GetDeltaRotation(const Bias &b_);
    Eigen::Vector3f GetDeltaVelocity(const Bias &b_);
    Eigen::Vector3f GetDeltaPosition(const Bias &b_);

    Eigen::Matrix3f GetUpdatedDeltaRotation();
    Eigen::Vector3f GetUpdatedDeltaVelocity();
    Eigen::Vector3f GetUpdatedDeltaPosition();

    Eigen::Matrix3f GetOriginalDeltaRotation();
    Eigen::Vector3f GetOriginalDeltaVelocity();
    Eigen::Vector3f GetOriginalDeltaPosition();

    Eigen::Matrix<float, 6, 1> GetDeltaBias();

    Bias GetOriginalBias();
    Bias GetUpdatedBias();

    void printMeasurements() const
    {
        std::cout << "pint meas:\n";
        for (unsigned int i = 0; i < mvMeasurements.size(); i++)
        {
            std::cout << "meas " << mvMeasurements[i].t << std::endl;
        }
        std::cout << "end pint meas:\n";
    }

public:
    float dT;
    Eigen::Matrix<float, 15, 15> C;
    Eigen::Matrix<float, 15, 15> Info;
    Eigen::DiagonalMatrix<float, 6> Nga, NgaWalk;

    // Values for the original bias (when integration was computed)
    Bias b;
    Eigen::Matrix3f dR;
    Eigen::Vector3f dV, dP;
    Eigen::Matrix3f JRg, JVg, JVa, JPg, JPa;
    Eigen::Vector3f avgA, avgW;

private:
    // Updated bias
    Bias bu;
    // Dif between original and updated bias
    // This is used to compute the updated values of the preintegration
    Eigen::Matrix<float, 6, 1> db;

    struct integrable
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        integrable() {}
        integrable(const Eigen::Vector3f &a_, const Eigen::Vector3f &w_, const float &t_) : a(a_), w(w_), t(t_) {}
        Eigen::Vector3f a, w;
        float t;
    };

    std::vector<integrable> mvMeasurements;

    std::mutex mMutex;
};

// Lie Algebra Functions
Eigen::Matrix3f RightJacobianSO3(const float &x, const float &y, const float &z);
Eigen::Matrix3f RightJacobianSO3(const Eigen::Vector3f &v);

Eigen::Matrix3f InverseRightJacobianSO3(const float &x, const float &y, const float &z);
Eigen::Matrix3f InverseRightJacobianSO3(const Eigen::Vector3f &v);

Eigen::Matrix3f NormalizeRotation(const Eigen::Matrix3f &R);

}