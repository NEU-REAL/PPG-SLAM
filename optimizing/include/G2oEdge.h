#pragma once

// ==================================================================================
// G2O EDGE DEFINITIONS FOR PPG-SLAM OPTIMIZATION
// ==================================================================================
// This file contains various edge types used in the pose graph optimization.
// Edges are organized into three main categories:
//
// 1. VISUAL EDGES: For camera reprojection constraints
//    - EdgeMono: Monocular projection with pose and point optimization
//    - EdgeMonoOnlyPose: Monocular projection with pose-only optimization
//    - EdgeSE3ProjectXYZ: SE3 pose projection edges
//    - EdgeSim3ProjectXYZ: Sim3 transformation projection edges
//
// 2. INERTIAL EDGES: For IMU and inertial constraints
//    - EdgeInertial: IMU preintegration constraints
//    - EdgeInertialGS: Inertial constraints with gravity and scale
//    - EdgeGyroRW/EdgeAccRW: Bias random walk constraints
//    - EdgePrior*: Prior constraints for pose, velocity, and biases
//
// 3. GEOMETRIC EDGES: For geometric constraints
//    - Edge4DoF: 4-degree-of-freedom pose constraints
//    - EdgeColine: Collinearity constraints for three points
// ==================================================================================

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/core/base_multi_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sim3/sim3.h"

#include<opencv2/core/core.hpp>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <Frame.h>
#include <KeyFrame.h>
#include "G2oVertex.h"

// Forward declarations
class KeyFrame;
class Frame;
class GeometricCamera;

// ==================================================================================
// VISUAL EDGES - Camera Reprojection Constraints
// ==================================================================================

/**
 * @brief Monocular camera projection edge
 * Edge for visual feature reprojection error with pose and 3D point optimization
 */
class EdgeMono : public g2o::BaseBinaryEdge<2,Eigen::Vector2d,g2o::VertexPointXYZ,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeMono()
    {}

    // I/O functions (not implemented)
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    // Core functions
    void computeError();                               // Compute reprojection error
    virtual void linearizeOplus();                     // Compute Jacobians
    bool isDepthPositive();                           // Check if 3D point is in front of camera
    
    // Optimization helpers
    Eigen::Matrix<double,2,9> GetJacobian();         // Get combined Jacobian matrix
    Eigen::Matrix<double,9,9> GetHessian();          // Get Hessian matrix
};

/**
 * @brief Monocular projection edge with fixed 3D point
 * Edge for visual feature reprojection error with only pose optimization
 */
class EdgeMonoOnlyPose : public g2o::BaseUnaryEdge<2,Eigen::Vector2d,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeMonoOnlyPose(const Eigen::Vector3f &Xw_):Xw(Xw_.cast<double>())
    {}

    // I/O functions (not implemented)
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    // Core functions
    void computeError();                               // Compute reprojection error
    virtual void linearizeOplus();                     // Compute Jacobians w.r.t. pose only
    bool isDepthPositive();                           // Check if 3D point is in front of camera
    
    // Optimization helpers
    Eigen::Matrix<double,6,6> GetHessian();          // Get Hessian matrix

public:
    const Eigen::Vector3d Xw;                         // Fixed 3D world point
};


/**
 * @brief SE3 projection edge with pose optimization only
 * Edge for visual feature reprojection with fixed 3D point and pose optimization
 */
class  EdgeSE3ProjectXYZOnlyPose: public  g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
        EdgeSE3ProjectXYZOnlyPose(){}
    
        // I/O functions (not implemented)
        bool read(std::istream& is){ return false; };
        bool write(std::ostream& os) const{ return false; };
    
        // Core functions
        void computeError();                           // Compute reprojection error
        bool isDepthPositive();                       // Check if point is in front of camera
        virtual void linearizeOplus();                // Compute Jacobians w.r.t. pose only
    
        Eigen::Vector3d Xw;                           // 3D world point
        GeometricCamera* pCamera;                     // Camera model
    };

/**
 * @brief SE3 projection edge with point and pose optimization
 * Edge for visual feature reprojection with both 3D point and pose optimization
 */
class  EdgeSE3ProjectXYZ: public  g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZ();

    // I/O functions (not implemented)
    bool read(std::istream& is){ return false; };
    bool write(std::ostream& os) const{ return false; };

    // Core functions
    void computeError();                               // Compute reprojection error
    bool isDepthPositive();                           // Check if point is in front of camera
    virtual void linearizeOplus();                    // Compute Jacobians w.r.t. point and pose

    GeometricCamera* pCamera;                         // Camera model
};

/**
 * @brief Sim3 projection edge
 * Edge for visual feature reprojection with Sim3 transformation (scale + SE3)
 */
class EdgeSim3ProjectXYZ : public  g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSim3ProjectXYZ();
    
    // I/O functions (not implemented)
    virtual bool read(std::istream& is){ return false; };
    virtual bool write(std::ostream& os) const{ return false; };

    // Core functions
    void computeError();                               // Compute reprojection error with Sim3
    // virtual void linearizeOplus();                 // Jacobians (commented out)
};

/**
 * @brief Inverse Sim3 projection edge
 * Edge for visual feature reprojection with inverse Sim3 transformation
 */
class EdgeInverseSim3ProjectXYZ : public  g2o::BaseBinaryEdge<2, Eigen::Vector2d,  g2o::VertexPointXYZ, VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeInverseSim3ProjectXYZ();
    
    // I/O functions (not implemented)
    virtual bool read(std::istream& is){ return false; };
    virtual bool write(std::ostream& os) const{ return false; };

    // Core functions
    void computeError();                               // Compute reprojection error with inverse Sim3
    // virtual void linearizeOplus();                 // Jacobians (commented out)
};


// ==================================================================================
// INERTIAL EDGES - IMU and Inertial Constraints  
// ==================================================================================

/**
 * @brief Inertial measurement constraint edge
 * Edge constraining relative pose, velocity, and biases using IMU preintegration
 */
class EdgeInertial : public g2o::BaseMultiEdge<9,Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeInertial(IMU::Preintegrated* pInt);           // Constructor with preintegrated IMU data

    // I/O functions (not implemented)
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    // Core functions
    void computeError();                               // Compute IMU constraint error
    virtual void linearizeOplus();                     // Compute Jacobians w.r.t. all vertices

    // Optimization helpers - different combinations of vertices
    Eigen::Matrix<double,24,24> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,9,24> J;
        J.block<9,6>(0,0) = _jacobianOplus[0];     // Pose1
        J.block<9,3>(0,6) = _jacobianOplus[1];     // Velocity1
        J.block<9,3>(0,9) = _jacobianOplus[2];     // Gyro bias
        J.block<9,3>(0,12) = _jacobianOplus[3];    // Acc bias
        J.block<9,6>(0,15) = _jacobianOplus[4];    // Pose2
        J.block<9,3>(0,21) = _jacobianOplus[5];    // Velocity2
        return J.transpose()*information()*J;
    }

    // Hessian without first pose
    Eigen::Matrix<double,18,18> GetHessianNoPose1(){
        linearizeOplus();
        Eigen::Matrix<double,9,18> J;
        J.block<9,3>(0,0) = _jacobianOplus[1];     // Velocity1
        J.block<9,3>(0,3) = _jacobianOplus[2];     // Gyro bias
        J.block<9,3>(0,6) = _jacobianOplus[3];     // Acc bias
        J.block<9,6>(0,9) = _jacobianOplus[4];     // Pose2
        J.block<9,3>(0,15) = _jacobianOplus[5];    // Velocity2
        return J.transpose()*information()*J;
    }

    // Hessian for second pose and velocity only
    Eigen::Matrix<double,9,9> GetHessian2(){
        linearizeOplus();
        Eigen::Matrix<double,9,9> J;
        J.block<9,6>(0,0) = _jacobianOplus[4];     // Pose2
        J.block<9,3>(0,6) = _jacobianOplus[5];     // Velocity2
        return J.transpose()*information()*J;
    }

    // Preintegration Jacobians (computed during IMU preintegration)
    const Eigen::Matrix3d JRg, JVg, JPg;              // Jacobians w.r.t. gyro bias
    const Eigen::Matrix3d JVa, JPa;                   // Jacobians w.r.t. accelerometer bias
    IMU::Preintegrated* mpInt;                        // Preintegrated IMU measurements
    const double dt;                                   // Time interval
    Eigen::Vector3d g;                                 // Gravity vector
};


/**
 * @brief Inertial edge with gravity and scale optimization
 * Edge for IMU constraints where gravity direction and scale are optimized variables
 * Used in visual-inertial initialization and scale estimation
 */
class EdgeInertialGS : public g2o::BaseMultiEdge<9,Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeInertialGS(IMU::Preintegrated* pInt);         // Constructor with preintegrated IMU data

    // I/O functions (not implemented)
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    // Optimization helpers - various Hessian combinations
    Eigen::Matrix<double,27,27> GetHessian();         // Full Hessian for all variables
    Eigen::Matrix<double,27,27> GetHessian2();        // Alternative ordering
    Eigen::Matrix<double,9,9> GetHessian3();          // Bias and gravity only
    Eigen::Matrix<double,1,1> GetHessianScale();      // Scale factor only
    Eigen::Matrix<double,3,3> GetHessianBiasGyro();   // Gyro bias only
    Eigen::Matrix<double,3,3> GetHessianBiasAcc();    // Accelerometer bias only
    Eigen::Matrix<double,2,2> GetHessianGDir();       // Gravity direction only

    // Core functions
    void computeError();                               // Compute IMU constraint error
    virtual void linearizeOplus();                     // Compute Jacobians w.r.t. all vertices

    // Preintegration Jacobians
    const Eigen::Matrix3d JRg, JVg, JPg;              // Jacobians w.r.t. gyro bias
    const Eigen::Matrix3d JVa, JPa;                   // Jacobians w.r.t. accelerometer bias
    IMU::Preintegrated* mpInt;                        // Preintegrated IMU measurements
    const double dt;                                   // Time interval
    Eigen::Vector3d g, gI;                             // Gravity vectors (world and inertial frame)
};


/**
 * @brief Gyroscope bias random walk constraint
 * Edge constraining gyroscope bias evolution between consecutive frames
 */
class EdgeGyroRW : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,VertexGyroBias,VertexGyroBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeGyroRW(){}

    // I/O functions (not implemented)
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    // Core functions
    void computeError();                               // Compute bias difference
    virtual void linearizeOplus();                     // Compute Jacobians

    // Optimization helpers
    Eigen::Matrix<double,6,6> GetHessian();          // Full Hessian for both bias vertices
    Eigen::Matrix3d GetHessian2();                    // Hessian for second bias vertex only
};


/**
 * @brief Accelerometer bias random walk constraint
 * Edge constraining accelerometer bias evolution between consecutive frames
 */
class EdgeAccRW : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,VertexAccBias,VertexAccBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeAccRW(){}

    // I/O functions (not implemented)
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    // Core functions
    void computeError();                               // Compute bias difference
    virtual void linearizeOplus();                     // Compute Jacobians

    // Optimization helpers
    Eigen::Matrix<double,6,6> GetHessian();          // Full Hessian for both bias vertices
    Eigen::Matrix3d GetHessian2();                    // Hessian for second bias vertex only
};


/**
 * @brief Constraint for pose-IMU state
 * Helper class storing pose, velocity, bias, and associated information matrix
 */
class ConstraintPoseImu
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ConstraintPoseImu(const Eigen::Matrix3d &Rwb_, const Eigen::Vector3d &twb_, const Eigen::Vector3d &vwb_,
                       const Eigen::Vector3d &bg_, const Eigen::Vector3d &ba_, const Matrix15d &H_):
                       Rwb(Rwb_), twb(twb_), vwb(vwb_), bg(bg_), ba(ba_), H(H_)
    {
        // Ensure information matrix is symmetric and positive semi-definite
        H = (H+H)/2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,15,15> > es(H);
        Eigen::Matrix<double,15,1> eigs = es.eigenvalues();
        for(int i=0;i<15;i++)
            if(eigs[i]<1e-12)
                eigs[i]=0;
        H = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    }

    Eigen::Matrix3d Rwb;                               // Rotation from body to world
    Eigen::Vector3d twb;                               // Translation from body to world
    Eigen::Vector3d vwb;                               // Velocity in world frame
    Eigen::Vector3d bg;                                // Gyroscope bias
    Eigen::Vector3d ba;                                // Accelerometer bias
    Matrix15d H;                                       // Information matrix
};

/**
 * @brief Prior constraint for pose-IMU state
 * Edge applying prior constraint on pose, velocity, and biases
 */
class EdgePriorPoseImu : public g2o::BaseMultiEdge<15,Vector15d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgePriorPoseImu(ConstraintPoseImu* c);          // Constructor with constraint

    // I/O functions (not implemented)
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    // Core functions
    void computeError();                               // Compute constraint error
    virtual void linearizeOplus();                     // Compute Jacobians

    // Optimization helpers
    Eigen::Matrix<double,15,15> GetHessian();         // Full Hessian
    Eigen::Matrix<double,9,9> GetHessianNoPose();     // Hessian without pose

    // Prior state
    Eigen::Matrix3d Rwb;                               // Prior rotation
    Eigen::Vector3d twb, vwb;                         // Prior translation and velocity
    Eigen::Vector3d bg, ba;                           // Prior biases
};

/**
 * @brief Prior constraint for accelerometer bias
 * Edge applying prior constraint on accelerometer bias
 */
class EdgePriorAcc : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexAccBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgePriorAcc(const Eigen::Vector3f &bprior_):bprior(bprior_.cast<double>()){}

    // I/O functions (not implemented)
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    // Core functions
    void computeError();                               // Compute bias constraint error
    virtual void linearizeOplus();                     // Compute Jacobians

    // Optimization helpers
    Eigen::Matrix<double,3,3> GetHessian();          // Get Hessian matrix

    const Eigen::Vector3d bprior;                     // Prior accelerometer bias
};

/**
 * @brief Prior constraint for gyroscope bias
 * Edge applying prior constraint on gyroscope bias
 */
class EdgePriorGyro : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexGyroBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgePriorGyro(const Eigen::Vector3f &bprior_):bprior(bprior_.cast<double>()){}

    // I/O functions (not implemented)
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    // Core functions
    void computeError();                               // Compute bias constraint error
    virtual void linearizeOplus();                     // Compute Jacobians

    // Optimization helpers
    Eigen::Matrix<double,3,3> GetHessian();          // Get Hessian matrix

    const Eigen::Vector3d bprior;                     // Prior gyroscope bias
};


// ==================================================================================
// GEOMETRIC EDGES - Geometric Constraints
// ==================================================================================

/**
 * @brief 4-DoF pose constraint edge
 * Edge for relative pose constraints with 4 degrees of freedom (rotation + translation)
 */
class Edge4DoF : public g2o::BaseBinaryEdge<6,Vector6d,VertexPose4DoF,VertexPose4DoF>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Edge4DoF(const Eigen::Matrix4d &deltaT);          // Constructor with relative transformation

    // I/O functions (not implemented)
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    // Core functions
    void computeError();                               // Compute relative pose error

    // Relative transformation components
    Eigen::Matrix4d dTij;                             // Full transformation matrix
    Eigen::Matrix3d dRij;                             // Rotation component
    Eigen::Vector3d dtij;                             // Translation component
};

/**
 * @brief Collinearity constraint edge
 * Edge enforcing collinearity constraint between three 3D points
 */
class EdgeColine : public g2o::BaseMultiEdge<3, Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeColine();

    // I/O functions (not implemented)
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    // Core functions
    void computeError();                               // Compute collinearity constraint error
    void linearizeOplus();                            // Compute Jacobians w.r.t. 3D points
};

