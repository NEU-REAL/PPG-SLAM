#pragma once

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

class KeyFrame;
class Frame;
class GeometricCamera;

class EdgeMono : public g2o::BaseBinaryEdge<2,Eigen::Vector2d,g2o::VertexPointXYZ,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeMono(int cam_idx_=0): cam_idx(cam_idx_)
    {}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();

    virtual void linearizeOplus();

    bool isDepthPositive();

    Eigen::Matrix<double,2,9> GetJacobian();

    Eigen::Matrix<double,9,9> GetHessian();

public:
    const int cam_idx;
};

class EdgeMonoOnlyPose : public g2o::BaseUnaryEdge<2,Eigen::Vector2d,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeMonoOnlyPose(const Eigen::Vector3f &Xw_, int cam_idx_=0):Xw(Xw_.cast<double>()), cam_idx(cam_idx_)
    {}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();

    virtual void linearizeOplus();

    bool isDepthPositive();

    Eigen::Matrix<double,6,6> GetHessian();

public:
    const Eigen::Vector3d Xw;
    const int cam_idx;
};

class EdgeInertial : public g2o::BaseMultiEdge<9,Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeInertial(IMU::Preintegrated* pInt);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();
    virtual void linearizeOplus();

    Eigen::Matrix<double,24,24> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,9,24> J;
        J.block<9,6>(0,0) = _jacobianOplus[0];
        J.block<9,3>(0,6) = _jacobianOplus[1];
        J.block<9,3>(0,9) = _jacobianOplus[2];
        J.block<9,3>(0,12) = _jacobianOplus[3];
        J.block<9,6>(0,15) = _jacobianOplus[4];
        J.block<9,3>(0,21) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,18,18> GetHessianNoPose1(){
        linearizeOplus();
        Eigen::Matrix<double,9,18> J;
        J.block<9,3>(0,0) = _jacobianOplus[1];
        J.block<9,3>(0,3) = _jacobianOplus[2];
        J.block<9,3>(0,6) = _jacobianOplus[3];
        J.block<9,6>(0,9) = _jacobianOplus[4];
        J.block<9,3>(0,15) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,9,9> GetHessian2(){
        linearizeOplus();
        Eigen::Matrix<double,9,9> J;
        J.block<9,6>(0,0) = _jacobianOplus[4];
        J.block<9,3>(0,6) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }

    const Eigen::Matrix3d JRg, JVg, JPg;
    const Eigen::Matrix3d JVa, JPa;
    IMU::Preintegrated* mpInt;
    const double dt;
    Eigen::Vector3d g;
};


// Edge inertial whre gravity is included as optimizable variable and it is not supposed to be pointing in -z axis, as well as scale
class EdgeInertialGS : public g2o::BaseMultiEdge<9,Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // EdgeInertialGS(IMU::Preintegrated* pInt);
    EdgeInertialGS(IMU::Preintegrated* pInt);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    Eigen::Matrix<double,27,27> GetHessian();
    
    Eigen::Matrix<double,27,27> GetHessian2();

    Eigen::Matrix<double,9,9> GetHessian3();

    Eigen::Matrix<double,1,1> GetHessianScale();

    Eigen::Matrix<double,3,3> GetHessianBiasGyro();

    Eigen::Matrix<double,3,3> GetHessianBiasAcc();

    Eigen::Matrix<double,2,2> GetHessianGDir();

    void computeError();

    virtual void linearizeOplus();

    const Eigen::Matrix3d JRg, JVg, JPg;
    const Eigen::Matrix3d JVa, JPa;
    IMU::Preintegrated* mpInt;
    const double dt;
    Eigen::Vector3d g, gI;
};



class EdgeGyroRW : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,VertexGyroBias,VertexGyroBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeGyroRW(){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();

    virtual void linearizeOplus();

    Eigen::Matrix<double,6,6> GetHessian();

    Eigen::Matrix3d GetHessian2();
};


class EdgeAccRW : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,VertexAccBias,VertexAccBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeAccRW(){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();

    virtual void linearizeOplus();

    Eigen::Matrix<double,6,6> GetHessian();

    Eigen::Matrix3d GetHessian2();
};

class ConstraintPoseImu
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ConstraintPoseImu(const Eigen::Matrix3d &Rwb_, const Eigen::Vector3d &twb_, const Eigen::Vector3d &vwb_,
                       const Eigen::Vector3d &bg_, const Eigen::Vector3d &ba_, const Matrix15d &H_):
                       Rwb(Rwb_), twb(twb_), vwb(vwb_), bg(bg_), ba(ba_), H(H_)
    {
        H = (H+H)/2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,15,15> > es(H);
        Eigen::Matrix<double,15,1> eigs = es.eigenvalues();
        for(int i=0;i<15;i++)
            if(eigs[i]<1e-12)
                eigs[i]=0;
        H = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    }

    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;
    Eigen::Vector3d vwb;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    Matrix15d H;
};

class EdgePriorPoseImu : public g2o::BaseMultiEdge<15,Vector15d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgePriorPoseImu(ConstraintPoseImu* c);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();
    virtual void linearizeOplus();

    Eigen::Matrix<double,15,15> GetHessian();

    Eigen::Matrix<double,9,9> GetHessianNoPose();

    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb, vwb;
    Eigen::Vector3d bg, ba;
};

// Priors for biases
class EdgePriorAcc : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexAccBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgePriorAcc(const Eigen::Vector3f &bprior_):bprior(bprior_.cast<double>()){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();

    virtual void linearizeOplus();

    Eigen::Matrix<double,3,3> GetHessian();

    const Eigen::Vector3d bprior;
};

class EdgePriorGyro : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexGyroBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgePriorGyro(const Eigen::Vector3f &bprior_):bprior(bprior_.cast<double>()){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();
    virtual void linearizeOplus();

    Eigen::Matrix<double,3,3> GetHessian();

    const Eigen::Vector3d bprior;
};


class Edge4DoF : public g2o::BaseBinaryEdge<6,Vector6d,VertexPose4DoF,VertexPose4DoF>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Edge4DoF(const Eigen::Matrix4d &deltaT);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();

    // virtual void linearizeOplus(); // numerical implementation
    Eigen::Matrix4d dTij;
    Eigen::Matrix3d dRij;
    Eigen::Vector3d dtij;
};

class EdgeColine : public g2o::BaseMultiEdge<3, Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeColine();

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();

    void linearizeOplus();
};


class  EdgeSE3ProjectXYZOnlyPose: public  g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
        EdgeSE3ProjectXYZOnlyPose(){}
    
        bool read(std::istream& is){ return false; };
    
        bool write(std::ostream& os) const{ return false; };
    
        void computeError();
    
        bool isDepthPositive();
    
        virtual void linearizeOplus();
    
        Eigen::Vector3d Xw;
        GeometricCamera* pCamera;
    };

class  EdgeSE3ProjectXYZ: public  g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZ();

    bool read(std::istream& is){ return false; };

    bool write(std::ostream& os) const{ return false; };

    void computeError();

    bool isDepthPositive();

    virtual void linearizeOplus();

    GeometricCamera* pCamera;
};

class EdgeSim3ProjectXYZ : public  g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSim3ProjectXYZ();
    virtual bool read(std::istream& is){ return false; };
    virtual bool write(std::ostream& os) const{ return false; };

    void computeError();
    // virtual void linearizeOplus();
};

class EdgeInverseSim3ProjectXYZ : public  g2o::BaseBinaryEdge<2, Eigen::Vector2d,  g2o::VertexPointXYZ, VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeInverseSim3ProjectXYZ();
    virtual bool read(std::istream& is){ return false; };
    virtual bool write(std::ostream& os) const{ return false; };

    void computeError();
    // virtual void linearizeOplus();
};
    