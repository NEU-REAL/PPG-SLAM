#include "G2oVertex.h"
using namespace IMU;

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W +0.5*W*W;
        return NormalizeRotation(res);
    }
    else
    {
        Eigen::Matrix3d res =Eigen::Matrix3d::Identity() + W*sin(d)/d + W*W*(1.0-cos(d))/d2;
        return NormalizeRotation(res);
    }
}

Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w)
{
    return ExpSO3(w[0],w[1],w[2]);
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    const double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
    const double costheta = (tr-1.0)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if(fabs(s)<1e-5)
        return w;
    else
        return theta*w/s;
}

Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
        return Eigen::Matrix3d::Identity();
    else
        return Eigen::Matrix3d::Identity() + W/2 + W*W*(1.0/d2 - (1.0+cos(d))/(2.0*d*sin(d)));
}

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v)
{
    return InverseRightJacobianSO3(v[0],v[1],v[2]);
}

Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        return Eigen::Matrix3d::Identity();
    }
    else
    {
        return Eigen::Matrix3d::Identity() - W*(1.0-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}

Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v)
{
    return RightJacobianSO3(v[0],v[1],v[2]);
}

Eigen::Matrix3d Skew(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d W;
    W << 0.0, -w[2], w[1],w[2], 0.0, -w[0],-w[1],  w[0], 0.0;
    return W;
}

ImuCamPose::ImuCamPose(KeyFrame *pKF, GeometricCamera* pCam):its(0)
{
    // Load IMU pose
    twb = pKF->GetImuPosition().cast<double>();
    Rwb = pKF->GetImuRotation().cast<double>();

    // Load camera pose
    tcw = pKF->GetTranslation().cast<double>();
    Rcw = pKF->GetRotation().cast<double>();
    tcb = pKF->mpImuCalib->mTcb.translation().cast<double>();
    Rcb = pKF->mpImuCalib->mTcb.rotationMatrix().cast<double>();
    Rbc = Rcb.transpose();
    tbc = pKF->mpImuCalib->mTbc.translation().cast<double>();
    pCamera = pCam;

    // For posegraph 4DoF
    Rwb0 = Rwb;
    DR.setIdentity();
}

ImuCamPose::ImuCamPose(Frame *pF, GeometricCamera* pCam):its(0)
{
    // Load IMU pose
    twb = pF->GetImuPosition().cast<double>();
    Rwb = pF->GetImuRotation().cast<double>();

    // Load camera pose
    tcw = pF->GetPose().translation().cast<double>();
    Rcw = pF->GetPose().rotationMatrix().cast<double>();
    tcb = pF->mpImuCalib->mTcb.translation().cast<double>();
    Rcb = pF->mpImuCalib->mTcb.rotationMatrix().cast<double>();
    Rbc = Rcb.transpose();
    tbc = pF->mpImuCalib->mTbc.translation().cast<double>();
    pCamera = pCam;

    // For posegraph 4DoF
    Rwb0 = Rwb;
    DR.setIdentity();
}

ImuCamPose::ImuCamPose(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, KeyFrame* pKF,  GeometricCamera* pCam): its(0)
{
    // This is only for posegraph, single camera setup
    tcb = pKF->mpImuCalib->mTcb.translation().cast<double>();
    Rcb = pKF->mpImuCalib->mTcb.rotationMatrix().cast<double>();
    Rbc = Rcb.transpose();
    tbc = pKF->mpImuCalib->mTbc.translation().cast<double>();
    twb = _Rwc * tcb + _twc;
    Rwb = _Rwc * Rcb;
    Rcw = _Rwc.transpose();
    tcw = -Rcw * _twc;
    pCamera = pCam;

    // For posegraph 4DoF
    Rwb0 = Rwb;
    DR.setIdentity();
}

Eigen::Vector2d ImuCamPose::Project(const Eigen::Vector3d &Xw) const
{
    Eigen::Vector3d Xc = Rcw * Xw + tcw;

    return pCamera->project(Xc);
}

bool ImuCamPose::isDepthPositive(const Eigen::Vector3d &Xw) const
{
    return (Rcw.row(2) * Xw + tcw(2)) > 0.0;
}

void ImuCamPose::Update(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];

    // Update body pose
    twb += Rwb * ut;
    Rwb = Rwb * ExpSO3(ur);

    // Normalize rotation after 5 updates
    its++;
    if(its>=3)
    {
        NormalizeRotation(Rwb);
        its=0;
    }

    // Update camera pose
    const Eigen::Matrix3d Rbw = Rwb.transpose();
    const Eigen::Vector3d tbw = -Rbw * twb;

    Rcw = Rcb * Rbw;
    tcw = Rcb * tbw + tcb;
}

void ImuCamPose::UpdateW(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];


    const Eigen::Matrix3d dR = ExpSO3(ur);
    DR = dR * DR;
    Rwb = DR * Rwb0;
    // Update body pose
    twb += ut;

    // Normalize rotation after 5 updates
    its++;
    if(its>=5)
    {
        DR(0,2) = 0.0;
        DR(1,2) = 0.0;
        DR(2,0) = 0.0;
        DR(2,1) = 0.0;
        NormalizeRotation(DR);
        its = 0;
    }

    // Update camera pose
    const Eigen::Matrix3d Rbw = Rwb.transpose();
    const Eigen::Vector3d tbw = -Rbw * twb;

    Rcw = Rcb * Rbw;
    tcw = Rcb * tbw + tcb;
}
