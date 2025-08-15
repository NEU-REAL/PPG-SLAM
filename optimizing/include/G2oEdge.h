#pragma once

// G2O includes
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_multi_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sim3/sim3.h"

// System includes
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

// Local includes
#include "Frame.h"
#include "KeyFrame.h"
#include "G2oVertex.h"

// Forward declarations
class KeyFrame;
class Frame;
class GeometricCamera;

// ===============================================
// VISUAL REPROJECTION EDGES
// ===============================================

/**
 * @brief Monocular reprojection edge for visual-inertial optimization
 * Connects 3D point and camera pose vertices
 */
class EdgeMono : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeMono(int cam_idx = 0) : cam_idx(cam_idx) {}

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;
    
    bool isDepthPositive();
    Eigen::Matrix<double, 2, 9> GetJacobian();
    Eigen::Matrix<double, 9, 9> GetHessian();

private:
    const int cam_idx;
};

/**
 * @brief Monocular reprojection edge for pose-only optimization
 * Used when 3D points are fixed
 */
class EdgeMonoOnlyPose : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeMonoOnlyPose(const Eigen::Vector3f& Xw, int cam_idx = 0)
        : Xw(Xw.cast<double>()), cam_idx(cam_idx) {}

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;
    
    bool isDepthPositive();
    Eigen::Matrix<double, 6, 6> GetHessian();

private:
    const Eigen::Vector3d Xw;
    const int cam_idx;
};

// ===============================================
// INERTIAL MEASUREMENT EDGES
// ===============================================

/**
 * @brief Inertial edge for visual-inertial optimization
 * Connects consecutive pose and bias vertices using IMU preintegration
 */
class EdgeInertial : public g2o::BaseMultiEdge<9, Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeInertial(IMU::Preintegrated* pInt);

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;

    // Hessian computation methods
    Eigen::Matrix<double, 24, 24> GetHessian() {
        linearizeOplus();
        Eigen::Matrix<double, 9, 24> J;
        J.block<9, 6>(0, 0) = _jacobianOplus[0];
        J.block<9, 3>(0, 6) = _jacobianOplus[1];
        J.block<9, 3>(0, 9) = _jacobianOplus[2];
        J.block<9, 3>(0, 12) = _jacobianOplus[3];
        J.block<9, 6>(0, 15) = _jacobianOplus[4];
        J.block<9, 3>(0, 21) = _jacobianOplus[5];
        return J.transpose() * information() * J;
    }

    Eigen::Matrix<double, 18, 18> GetHessianNoPose1() {
        linearizeOplus();
        Eigen::Matrix<double, 9, 18> J;
        J.block<9, 3>(0, 0) = _jacobianOplus[1];
        J.block<9, 3>(0, 3) = _jacobianOplus[2];
        J.block<9, 3>(0, 6) = _jacobianOplus[3];
        J.block<9, 6>(0, 9) = _jacobianOplus[4];
        J.block<9, 3>(0, 15) = _jacobianOplus[5];
        return J.transpose() * information() * J;
    }

    Eigen::Matrix<double, 9, 9> GetHessian2() {
        linearizeOplus();
        Eigen::Matrix<double, 9, 9> J;
        J.block<9, 6>(0, 0) = _jacobianOplus[4];
        J.block<9, 3>(0, 6) = _jacobianOplus[5];
        return J.transpose() * information() * J;
    }

private:
    const Eigen::Matrix3d JRg, JVg, JPg;  ///< Jacobians w.r.t gyroscope bias
    const Eigen::Matrix3d JVa, JPa;       ///< Jacobians w.r.t accelerometer bias
    IMU::Preintegrated* mpInt;             ///< Preintegrated measurements
    const double dt;                       ///< Time interval
    Eigen::Vector3d g;                     ///< Gravity vector
};

/**
 * @brief Inertial edge with gravity and scale optimization
 * Extended version for monocular SLAM initialization
 */
class EdgeInertialGS : public g2o::BaseMultiEdge<9, Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeInertialGS(IMU::Preintegrated* pInt);

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;

    // Specialized Hessian methods
    Eigen::Matrix<double, 27, 27> GetHessian();
    Eigen::Matrix<double, 27, 27> GetHessian2();
    Eigen::Matrix<double, 9, 9> GetHessian3();
    Eigen::Matrix<double, 1, 1> GetHessianScale();
    Eigen::Matrix<double, 3, 3> GetHessianBiasGyro();
    Eigen::Matrix<double, 3, 3> GetHessianBiasAcc();
    Eigen::Matrix<double, 2, 2> GetHessianGDir();

private:
    const Eigen::Matrix3d JRg, JVg, JPg;  ///< Jacobians w.r.t gyroscope bias
    const Eigen::Matrix3d JVa, JPa;       ///< Jacobians w.r.t accelerometer bias
    IMU::Preintegrated* mpInt;             ///< Preintegrated measurements
    const double dt;                       ///< Time interval
    Eigen::Vector3d g, gI;                 ///< Gravity vectors
};

// ===============================================
// BIAS RANDOM WALK EDGES
// ===============================================

/**
 * @brief Gyroscope bias random walk edge
 * Enforces smooth evolution of gyroscope bias
 */
class EdgeGyroRW : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, VertexGyroBias, VertexGyroBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeGyroRW() = default;

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;

    Eigen::Matrix<double, 6, 6> GetHessian();
    Eigen::Matrix3d GetHessian2();
};

/**
 * @brief Accelerometer bias random walk edge
 * Enforces smooth evolution of accelerometer bias
 */
class EdgeAccRW : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, VertexAccBias, VertexAccBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeAccRW() = default;

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;

    Eigen::Matrix<double, 6, 6> GetHessian();
    Eigen::Matrix3d GetHessian2();
};

// ===============================================
// PRIOR CONSTRAINTS
// ===============================================

/**
 * @brief Pose-IMU constraint data structure
 * Stores prior information for pose and IMU state
 */
class ConstraintPoseImu
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ConstraintPoseImu(const Eigen::Matrix3d& Rwb, const Eigen::Vector3d& twb, 
                      const Eigen::Vector3d& vwb, const Eigen::Vector3d& bg, 
                      const Eigen::Vector3d& ba, const Matrix15d& H)
        : Rwb(Rwb), twb(twb), vwb(vwb), bg(bg), ba(ba), H(H)
    {
        // Ensure Hessian is symmetric and positive semi-definite
        this->H = (H + H.transpose()) / 2.0;
        Eigen::SelfAdjointEigenSolver<Matrix15d> es(this->H);
        Eigen::Matrix<double, 15, 1> eigs = es.eigenvalues();
        
        // Clamp negative eigenvalues to zero
        for (int i = 0; i < 15; ++i) {
            if (eigs[i] < 1e-12) {
                eigs[i] = 0.0;
            }
        }
        
        this->H = es.eigenvectors() * eigs.asDiagonal() * es.eigenvectors().transpose();
    }

    Eigen::Matrix3d Rwb;     ///< Rotation from body to world
    Eigen::Vector3d twb;     ///< Translation from body to world
    Eigen::Vector3d vwb;     ///< Velocity in world frame
    Eigen::Vector3d bg;      ///< Gyroscope bias
    Eigen::Vector3d ba;      ///< Accelerometer bias
    Matrix15d H;             ///< Information matrix
};

/**
 * @brief Prior edge for pose and IMU state
 * Applies prior constraints on full pose-IMU state
 */
class EdgePriorPoseImu : public g2o::BaseMultiEdge<15, Vector15d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    explicit EdgePriorPoseImu(ConstraintPoseImu* constraint);

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;

    Eigen::Matrix<double, 15, 15> GetHessian();
    Eigen::Matrix<double, 9, 9> GetHessianNoPose();

private:
    Eigen::Matrix3d Rwb;           ///< Prior rotation
    Eigen::Vector3d twb, vwb;      ///< Prior translation and velocity
    Eigen::Vector3d bg, ba;        ///< Prior biases
};

/**
 * @brief Prior edge for accelerometer bias
 */
class EdgePriorAcc : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexAccBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgePriorAcc(const Eigen::Vector3f& prior) : bprior(prior.cast<double>()) {}

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;

    Eigen::Matrix<double, 3, 3> GetHessian();

private:
    const Eigen::Vector3d bprior;  ///< Prior bias value
};

/**
 * @brief Prior edge for gyroscope bias
 */
class EdgePriorGyro : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexGyroBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgePriorGyro(const Eigen::Vector3f& prior) : bprior(prior.cast<double>()) {}

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;

    Eigen::Matrix<double, 3, 3> GetHessian();

private:
    const Eigen::Vector3d bprior;  ///< Prior bias value
};

// ===============================================
// POSE CONSTRAINT EDGES
// ===============================================

// ===============================================
// POSE CONSTRAINT EDGES
// ===============================================

/**
 * @brief 4-DOF pose constraint edge
 * For pose-graph optimization with reduced degrees of freedom
 */
class Edge4DoF : public g2o::BaseBinaryEdge<6, Vector6d, VertexPose4DoF, VertexPose4DoF>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit Edge4DoF(const Eigen::Matrix4d& deltaT);

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;

private:
    Eigen::Matrix4d dTij;    ///< Relative transformation
    Eigen::Matrix3d dRij;    ///< Relative rotation
    Eigen::Vector3d dtij;    ///< Relative translation
};

/**
 * @brief Collinearity constraint edge
 * Enforces collinearity between multiple points
 */
class EdgeColine : public g2o::BaseMultiEdge<3, Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EdgeColine();

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;
};

// ===============================================
// SE3 PROJECTION EDGES
// ===============================================

/**
 * @brief SE3 projection edge for pose-only optimization
 * Projects 3D point to image using SE3 pose
 */
class EdgeSE3ProjectXYZOnlyPose : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZOnlyPose() = default;

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;
    
    bool isDepthPositive();

    Eigen::Vector3d Xw;              ///< 3D point in world coordinates
    GeometricCamera* pCamera;        ///< Camera model
};

/**
 * @brief SE3 projection edge for joint optimization
 * Projects 3D point to image using SE3 pose (point and pose optimization)
 */
class EdgeSE3ProjectXYZ : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZ();

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;
    
    bool isDepthPositive();

    GeometricCamera* pCamera;        ///< Camera model
};

// ===============================================
// SIM3 PROJECTION EDGES
// ===============================================

/**
 * @brief Sim3 projection edge for loop closure
 * Projects 3D point using Sim3 transformation
 */
class EdgeSim3ProjectXYZ : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EdgeSim3ProjectXYZ();

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
};

/**
 * @brief Inverse Sim3 projection edge
 * Projects 3D point using inverse Sim3 transformation
 */
class EdgeInverseSim3ProjectXYZ : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EdgeInverseSim3ProjectXYZ();

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
};
    