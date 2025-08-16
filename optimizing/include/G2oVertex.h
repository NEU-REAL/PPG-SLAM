#pragma once

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_multi_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sim3/sim3.h"
#include "g2o/types/slam3d/se3quat.h"

#include<opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Frame.h>
#include <KeyFrame.h>

#include <math.h>

class KeyFrame;
class Frame;
class GeometricCamera;

typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;

template<typename T = double>
Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T,3,3> &R) 
{
    Eigen::JacobiSVD<Eigen::Matrix<T,3,3>> svd(R,Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z);

Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);

Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z);

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v);

Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z);

Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v);

Eigen::Matrix3d Skew(const Eigen::Vector3d &w);

class ImuCamPose
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuCamPose(){}
    ImuCamPose(KeyFrame* pKF, GeometricCamera* pCam);
    ImuCamPose(Frame* pF, GeometricCamera* pCam);
    ImuCamPose(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, KeyFrame* pKF, GeometricCamera* pCam);

    void Update(const double *pu); // update in the imu reference
    void UpdateW(const double *pu); // update in the world reference
    Eigen::Vector2d Project(const Eigen::Vector3d &Xw) const; // Mono
    bool isDepthPositive(const Eigen::Vector3d &Xw) const;

public:
    // For IMU
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;

    // For single camera
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    Eigen::Matrix3d Rcb, Rbc;
    Eigen::Vector3d tcb, tbc;
    double bf;
    GeometricCamera* pCamera;

    // For posegraph 4DoF
    Eigen::Matrix3d Rwb0;
    Eigen::Matrix3d DR;

    int its;
};

// Optimizable parameters are IMU pose
class VertexPose : public g2o::BaseVertex<6,ImuCamPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose(){}
    VertexPose(KeyFrame* pKF, GeometricCamera*pCam){
        setEstimate(ImuCamPose(pKF,pCam));
    }
    VertexPose(Frame* pF, GeometricCamera*pCam){
        setEstimate(ImuCamPose(pF,pCam));
    }

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_)
    {
        _estimate.Update(update_);
        updateCache();
    }
};

class VertexPose4DoF : public g2o::BaseVertex<4,ImuCamPose>
{
    // Translation and yaw are the only optimizable variables
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose4DoF(){}
    VertexPose4DoF(KeyFrame* pKF, GeometricCamera* pCam){
        setEstimate(ImuCamPose(pKF,pCam));
    }
    VertexPose4DoF(Frame* pF, GeometricCamera* pCam){
        setEstimate(ImuCamPose(pF,pCam));
    }
    VertexPose4DoF(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, KeyFrame* pKF, GeometricCamera* pCam){

        setEstimate(ImuCamPose(_Rwc, _twc, pKF, pCam));
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        double update6DoF[6];
        update6DoF[0] = 0;
        update6DoF[1] = 0;
        update6DoF[2] = update_[0];
        update6DoF[3] = update_[1];
        update6DoF[4] = update_[2];
        update6DoF[5] = update_[3];
        _estimate.UpdateW(update6DoF);
        updateCache();
    }
};

class VertexVelocity : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexVelocity(){}
    VertexVelocity(KeyFrame* pKF)
    {
        setEstimate(pKF->GetVelocity().cast<double>());
    }
    VertexVelocity(Frame* pF)
    {
        setEstimate(pF->GetVelocity().cast<double>());
    }
    

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d uv;
        uv << update_[0], update_[1], update_[2];
        setEstimate(estimate()+uv);
    }
};

class VertexGyroBias : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGyroBias(){}
    VertexGyroBias(KeyFrame* pKF)
    {
        setEstimate(pKF->GetGyroBias().cast<double>());
    }
    VertexGyroBias(Frame* pF)
    {
        Eigen::Vector3d bg;
        bg << pF->mImuBias.bwx, pF->mImuBias.bwy,pF->mImuBias.bwz;
        setEstimate(bg);
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d ubg;
        ubg << update_[0], update_[1], update_[2];
        setEstimate(estimate()+ubg);
    }
};


class VertexAccBias : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexAccBias(){}
    VertexAccBias(KeyFrame* pKF)
    {
        setEstimate(pKF->GetAccBias().cast<double>());
    }
    VertexAccBias(Frame* pF)
    {
        Eigen::Vector3d ba;
        ba << pF->mImuBias.bax, pF->mImuBias.bay,pF->mImuBias.baz;
        setEstimate(ba);
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d uba;
        uba << update_[0], update_[1], update_[2];
        setEstimate(estimate()+uba);
    }
};

// Gravity direction vertex
class GDirection
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GDirection(){}
    GDirection(Eigen::Matrix3d pRwg): Rwg(pRwg){}
    void Update(const double *pu)
    {
        Rwg=Rwg*ExpSO3(pu[0],pu[1],0.0);
    }
    Eigen::Matrix3d Rwg, Rgw;
    int its;
};

class VertexGDir : public g2o::BaseVertex<2,GDirection>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGDir(){}
    VertexGDir(Eigen::Matrix3d pRwg){
        setEstimate(GDirection(pRwg));
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        _estimate.Update(update_);
        updateCache();
    }
};

// scale vertex
class VertexScale : public g2o::BaseVertex<1,double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexScale(){
        setEstimate(1.0);
    }
    VertexScale(double ps){
        setEstimate(ps);
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl(){
        setEstimate(1.0);
    }

    virtual void oplusImpl(const double *update_){
        setEstimate(estimate()*exp(*update_));
    }
};

class VertexSim3Expmap : public g2o::BaseVertex<7, g2o::Sim3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexSim3Expmap();
    virtual bool read(std::istream& is){ return false; };
    virtual bool write(std::ostream& os) const{ return false; };

    virtual void setToOriginImpl() {
        _estimate = g2o::Sim3();
    }

    virtual void oplusImpl(const double* update_)
    {
        Eigen::Map<g2o::Vector7> update(const_cast<double*>(update_));

        if (_fix_scale)
            update[6] = 0;

        g2o::Sim3 s(update);
        setEstimate(s*estimate());
    }

    GeometricCamera* pCamera1, *pCamera2;

    bool _fix_scale;
};

