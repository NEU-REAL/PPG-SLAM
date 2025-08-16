
#include "G2oEdge.h" 

void EdgeMono::computeError()
{
    const g2o::VertexPointXYZ* VPoint = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
    const Eigen::Vector2d obs(_measurement);
    _error = obs - VPose->estimate().Project(VPoint->estimate(),cam_idx);
}

void EdgeMono::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
    const g2o::VertexPointXYZ* VPoint = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);

    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw[cam_idx];
    const Eigen::Vector3d &tcw = VPose->estimate().tcw[cam_idx];
    const Eigen::Vector3d Xc = Rcw*VPoint->estimate() + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc[cam_idx]*Xc+VPose->estimate().tbc[cam_idx];
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb[cam_idx];

    const Eigen::Matrix<double,2,3> proj_jac = VPose->estimate().pCamera[cam_idx]->projectJac(Xc);
    _jacobianOplusXi = -proj_jac * Rcw;

    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);

    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
            -z , 0.0, x, 0.0, 1.0, 0.0,
            y ,  -x , 0.0, 0.0, 0.0, 1.0;

    // Use temporary variable to avoid alignment issues
    Eigen::Matrix<double, 2, 6> temp_xj = proj_jac * Rcb * SE3deriv;
    _jacobianOplusXj = temp_xj;
}

bool EdgeMono::isDepthPositive()
{
    const g2o::VertexPointXYZ* VPoint = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
    return VPose->estimate().isDepthPositive(VPoint->estimate(),cam_idx);
}


Eigen::Matrix<double,2,9> EdgeMono::GetJacobian()
{
    linearizeOplus();
    Eigen::Matrix<double,2,9> J;
    J.block<2,3>(0,0) = _jacobianOplusXi;
    J.block<2,6>(0,3) = _jacobianOplusXj;
    return J;
}

Eigen::Matrix<double,9,9> EdgeMono::GetHessian()
{
    linearizeOplus();
    Eigen::Matrix<double,2,9> J;
    J.block<2,3>(0,0) = _jacobianOplusXi;
    J.block<2,6>(0,3) = _jacobianOplusXj;
    return J.transpose()*information()*J;
}

void EdgeMonoOnlyPose::computeError()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
    const Eigen::Vector2d obs(_measurement);
    _error = obs - VPose->estimate().Project(Xw,cam_idx);
}


bool EdgeMonoOnlyPose::isDepthPositive()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
    return VPose->estimate().isDepthPositive(Xw,cam_idx);
}

Eigen::Matrix<double,6,6> EdgeMonoOnlyPose::GetHessian()
{
    linearizeOplus();
    return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
}

void EdgeMonoOnlyPose::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);

    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw[cam_idx];
    const Eigen::Vector3d &tcw = VPose->estimate().tcw[cam_idx];
    const Eigen::Vector3d Xc = Rcw*Xw + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc[cam_idx]*Xc+VPose->estimate().tbc[cam_idx];
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb[cam_idx];

    Eigen::Matrix<double,2,3> proj_jac = VPose->estimate().pCamera[cam_idx]->projectJac(Xc);

    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);
    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
            -z , 0.0, x, 0.0, 1.0, 0.0,
            y ,  -x , 0.0, 0.0, 0.0, 1.0;
    
    // Use temporary variable to avoid alignment issues
    Eigen::Matrix<double, 2, 6> temp_xi = proj_jac * Rcb * SE3deriv;
    _jacobianOplusXi = temp_xi;
}

EdgeInertial::EdgeInertial(IMU::Preintegrated *pInt):JRg(pInt->JRg.cast<double>()),
    JVg(pInt->JVg.cast<double>()), JPg(pInt->JPg.cast<double>()), JVa(pInt->JVa.cast<double>()),
    JPa(pInt->JPa.cast<double>()), mpInt(pInt), dt(pInt->dT)
{
    // This edge links 6 vertices
    resize(6);
    g << 0, 0, -IMU::GRAVITY_VALUE;

    Matrix9d Info = pInt->C.block<9,9>(0,0).cast<double>().inverse();
    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
    Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
    for(int i=0;i<9;i++)
        if(eigs[i]<1e-12)
            eigs[i]=0;
    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    setInformation(Info);
}

void EdgeInertial::computeError()
{
    // TODO Maybe Reintegrate inertial measurments when difference between linearization point and current estimate is too big
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const IMU::Bias b1(VA1->estimate()[0],VA1->estimate()[1],VA1->estimate()[2],VG1->estimate()[0],VG1->estimate()[1],VG1->estimate()[2]);
    const Eigen::Matrix3d dR = mpInt->GetDeltaRotation(b1).cast<double>();
    const Eigen::Vector3d dV = mpInt->GetDeltaVelocity(b1).cast<double>();
    const Eigen::Vector3d dP = mpInt->GetDeltaPosition(b1).cast<double>();

    const Eigen::Vector3d er = LogSO3(dR.transpose()*VP1->estimate().Rwb.transpose()*VP2->estimate().Rwb);
    const Eigen::Vector3d ev = VP1->estimate().Rwb.transpose()*(VV2->estimate() - VV1->estimate() - g*dt) - dV;
    const Eigen::Vector3d ep = VP1->estimate().Rwb.transpose()*(VP2->estimate().twb - VP1->estimate().twb
                                                               - VV1->estimate()*dt - g*dt*dt/2) - dP;

    _error << er, ev, ep;
}

void EdgeInertial::linearizeOplus()
{
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2= static_cast<const VertexVelocity*>(_vertices[5]);
    const IMU::Bias b1(VA1->estimate()[0],VA1->estimate()[1],VA1->estimate()[2],VG1->estimate()[0],VG1->estimate()[1],VG1->estimate()[2]);
    const IMU::Bias db = mpInt->GetDeltaBias(b1);
    Eigen::Vector3d dbg;
    dbg << db.bwx, db.bwy, db.bwz;

    const Eigen::Matrix3d Rwb1 = VP1->estimate().Rwb;
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = VP2->estimate().Rwb;

    const Eigen::Matrix3d dR = mpInt->GetDeltaRotation(b1).cast<double>();
    const Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
    const Eigen::Vector3d er = LogSO3(eR);
    const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);

    // Jacobians wrt Pose 1
    _jacobianOplus[0].setZero();
     // rotation
    _jacobianOplus[0].block<3,3>(0,0) = -invJr*Rwb2.transpose()*Rwb1; // OK
    _jacobianOplus[0].block<3,3>(3,0) = SO3d::hat(Rbw1*(VV2->estimate() - VV1->estimate() - g*dt)); // OK
    _jacobianOplus[0].block<3,3>(6,0) = SO3d::hat(Rbw1*(VP2->estimate().twb - VP1->estimate().twb
                                                   - VV1->estimate()*dt - 0.5*g*dt*dt)); // OK
    // translation
    _jacobianOplus[0].block<3,3>(6,3) = -Eigen::Matrix3d::Identity(); // OK

    // Jacobians wrt Velocity 1
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(3,0) = -Rbw1; // OK
    _jacobianOplus[1].block<3,3>(6,0) = -Rbw1*dt; // OK

    // Jacobians wrt Gyro 1
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(0,0) = -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg; // OK
    _jacobianOplus[2].block<3,3>(3,0) = -JVg; // OK
    _jacobianOplus[2].block<3,3>(6,0) = -JPg; // OK

    // Jacobians wrt Accelerometer 1
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(3,0) = -JVa; // OK
    _jacobianOplus[3].block<3,3>(6,0) = -JPa; // OK

    // Jacobians wrt Pose 2
    _jacobianOplus[4].setZero();
    // rotation
    _jacobianOplus[4].block<3,3>(0,0) = invJr; // OK
    // translation
    _jacobianOplus[4].block<3,3>(6,3) = Rbw1*Rwb2; // OK

    // Jacobians wrt Velocity 2
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3,3>(3,0) = Rbw1; // OK
}

EdgeInertialGS::EdgeInertialGS(IMU::Preintegrated *pInt):JRg(pInt->JRg.cast<double>()),
    JVg(pInt->JVg.cast<double>()), JPg(pInt->JPg.cast<double>()), JVa(pInt->JVa.cast<double>()),
    JPa(pInt->JPa.cast<double>()), mpInt(pInt), dt(pInt->dT)
{
    // This edge links 8 vertices
    resize(8);
    gI << 0, 0, -IMU::GRAVITY_VALUE;

    Matrix9d Info = pInt->C.block<9,9>(0,0).cast<double>().inverse();
    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
    Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
    for(int i=0;i<9;i++)
        if(eigs[i]<1e-12)
            eigs[i]=0;
    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    setInformation(Info);
}

Eigen::Matrix<double,27,27> EdgeInertialGS::GetHessian()
{
    linearizeOplus();
    Eigen::Matrix<double,9,27> J;
    J.block<9,6>(0,0) = _jacobianOplus[0];
    J.block<9,3>(0,6) = _jacobianOplus[1];
    J.block<9,3>(0,9) = _jacobianOplus[2];
    J.block<9,3>(0,12) = _jacobianOplus[3];
    J.block<9,6>(0,15) = _jacobianOplus[4];
    J.block<9,3>(0,21) = _jacobianOplus[5];
    J.block<9,2>(0,24) = _jacobianOplus[6];
    J.block<9,1>(0,26) = _jacobianOplus[7];
    return J.transpose()*information()*J;
}

Eigen::Matrix<double,27,27> EdgeInertialGS::GetHessian2()
{
    linearizeOplus();
    Eigen::Matrix<double,9,27> J;
    J.block<9,3>(0,0) = _jacobianOplus[2];
    J.block<9,3>(0,3) = _jacobianOplus[3];
    J.block<9,2>(0,6) = _jacobianOplus[6];
    J.block<9,1>(0,8) = _jacobianOplus[7];
    J.block<9,3>(0,9) = _jacobianOplus[1];
    J.block<9,3>(0,12) = _jacobianOplus[5];
    J.block<9,6>(0,15) = _jacobianOplus[0];
    J.block<9,6>(0,21) = _jacobianOplus[4];
    return J.transpose()*information()*J;
}

Eigen::Matrix<double,9,9> EdgeInertialGS::GetHessian3()
{
    linearizeOplus();
    Eigen::Matrix<double,9,9> J;
    J.block<9,3>(0,0) = _jacobianOplus[2];
    J.block<9,3>(0,3) = _jacobianOplus[3];
    J.block<9,2>(0,6) = _jacobianOplus[6];
    J.block<9,1>(0,8) = _jacobianOplus[7];
    return J.transpose()*information()*J;
}



Eigen::Matrix<double,1,1> EdgeInertialGS::GetHessianScale()
{
    linearizeOplus();
    Eigen::Matrix<double,9,1> J = _jacobianOplus[7];
    return J.transpose()*information()*J;
}

Eigen::Matrix<double,3,3> EdgeInertialGS::GetHessianBiasGyro()
{
    linearizeOplus();
    Eigen::Matrix<double,9,3> J = _jacobianOplus[2];
    return J.transpose()*information()*J;
}

Eigen::Matrix<double,3,3> EdgeInertialGS::GetHessianBiasAcc()
{
    linearizeOplus();
    Eigen::Matrix<double,9,3> J = _jacobianOplus[3];
    return J.transpose()*information()*J;
}

Eigen::Matrix<double,2,2> EdgeInertialGS::GetHessianGDir()
{
    linearizeOplus();
    Eigen::Matrix<double,9,2> J = _jacobianOplus[6];
    return J.transpose()*information()*J;
}

void EdgeInertialGS::computeError()
{
    // TODO Maybe Reintegrate inertial measurments when difference between linearization point and current estimate is too big
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const VertexGDir* VGDir = static_cast<const VertexGDir*>(_vertices[6]);
    const VertexScale* VS = static_cast<const VertexScale*>(_vertices[7]);
    const IMU::Bias b(VA->estimate()[0],VA->estimate()[1],VA->estimate()[2],VG->estimate()[0],VG->estimate()[1],VG->estimate()[2]);
    g = VGDir->estimate().Rwg*gI;
    const double s = VS->estimate();
    const Eigen::Matrix3d dR = mpInt->GetDeltaRotation(b).cast<double>();
    const Eigen::Vector3d dV = mpInt->GetDeltaVelocity(b).cast<double>();
    const Eigen::Vector3d dP = mpInt->GetDeltaPosition(b).cast<double>();

    const Eigen::Vector3d er = LogSO3(dR.transpose()*VP1->estimate().Rwb.transpose()*VP2->estimate().Rwb);
    const Eigen::Vector3d ev = VP1->estimate().Rwb.transpose()*(s*(VV2->estimate() - VV1->estimate()) - g*dt) - dV;
    const Eigen::Vector3d ep = VP1->estimate().Rwb.transpose()*(s*(VP2->estimate().twb - VP1->estimate().twb - VV1->estimate()*dt) - g*dt*dt/2) - dP;

    _error << er, ev, ep;
}

void EdgeInertialGS::linearizeOplus()
{
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const VertexGDir* VGDir = static_cast<const VertexGDir*>(_vertices[6]);
    const VertexScale* VS = static_cast<const VertexScale*>(_vertices[7]);
    const IMU::Bias b(VA->estimate()[0],VA->estimate()[1],VA->estimate()[2],VG->estimate()[0],VG->estimate()[1],VG->estimate()[2]);
    const IMU::Bias db = mpInt->GetDeltaBias(b);

    Eigen::Vector3d dbg;
    dbg << db.bwx, db.bwy, db.bwz;

    const Eigen::Matrix3d Rwb1 = VP1->estimate().Rwb;
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = VP2->estimate().Rwb;
    const Eigen::Matrix3d Rwg = VGDir->estimate().Rwg;
    Eigen::MatrixXd Gm = Eigen::MatrixXd::Zero(3,2);
    Gm(0,1) = -IMU::GRAVITY_VALUE;
    Gm(1,0) = IMU::GRAVITY_VALUE;
    const double s = VS->estimate();
    const Eigen::MatrixXd dGdTheta = Rwg*Gm;
    const Eigen::Matrix3d dR = mpInt->GetDeltaRotation(b).cast<double>();
    const Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
    const Eigen::Vector3d er = LogSO3(eR);
    const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);

    // Jacobians wrt Pose 1
    _jacobianOplus[0].setZero();
     // rotation
    _jacobianOplus[0].block<3,3>(0,0) = -invJr*Rwb2.transpose()*Rwb1;
    _jacobianOplus[0].block<3,3>(3,0) = SO3d::hat(Rbw1*(s*(VV2->estimate() - VV1->estimate()) - g*dt));
    _jacobianOplus[0].block<3,3>(6,0) = SO3d::hat(Rbw1*(s*(VP2->estimate().twb - VP1->estimate().twb
                                                   - VV1->estimate()*dt) - 0.5*g*dt*dt));
    // translation
    _jacobianOplus[0].block<3,3>(6,3) = Eigen::DiagonalMatrix<double,3>(-s,-s,-s);

    // Jacobians wrt Velocity 1
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(3,0) = -s*Rbw1;
    _jacobianOplus[1].block<3,3>(6,0) = -s*Rbw1*dt;

    // Jacobians wrt Gyro bias
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(0,0) = -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg;
    _jacobianOplus[2].block<3,3>(3,0) = -JVg;
    _jacobianOplus[2].block<3,3>(6,0) = -JPg;

    // Jacobians wrt Accelerometer bias
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(3,0) = -JVa;
    _jacobianOplus[3].block<3,3>(6,0) = -JPa;

    // Jacobians wrt Pose 2
    _jacobianOplus[4].setZero();
    // rotation
    _jacobianOplus[4].block<3,3>(0,0) = invJr;
    // translation
    _jacobianOplus[4].block<3,3>(6,3) = s*Rbw1*Rwb2;

    // Jacobians wrt Velocity 2
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3,3>(3,0) = s*Rbw1;

    // Jacobians wrt Gravity direction
    _jacobianOplus[6].setZero();
    _jacobianOplus[6].block<3,2>(3,0) = -Rbw1*dGdTheta*dt;
    _jacobianOplus[6].block<3,2>(6,0) = -0.5*Rbw1*dGdTheta*dt*dt;

    // Jacobians wrt scale factor
    _jacobianOplus[7].setZero();
    _jacobianOplus[7].block<3,1>(3,0) = Rbw1*(VV2->estimate()-VV1->estimate());
    _jacobianOplus[7].block<3,1>(6,0) = Rbw1*(VP2->estimate().twb-VP1->estimate().twb-VV1->estimate()*dt);
}


void EdgeGyroRW::computeError()
{
    const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[0]);
    const VertexGyroBias* VG2= static_cast<const VertexGyroBias*>(_vertices[1]);
    _error = VG2->estimate()-VG1->estimate();
}

void EdgeGyroRW::linearizeOplus()
{
    _jacobianOplusXi = -Eigen::Matrix3d::Identity();
    _jacobianOplusXj.setIdentity();
}

Eigen::Matrix<double,6,6> EdgeGyroRW::GetHessian()
{
    linearizeOplus();
    Eigen::Matrix<double,3,6> J;
    J.block<3,3>(0,0) = _jacobianOplusXi;
    J.block<3,3>(0,3) = _jacobianOplusXj;
    return J.transpose()*information()*J;
}

Eigen::Matrix3d EdgeGyroRW::GetHessian2()
{
    linearizeOplus();
    return _jacobianOplusXj.transpose()*information()*_jacobianOplusXj;
}

void EdgeAccRW::computeError()
{
    const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[0]);
    const VertexAccBias* VA2= static_cast<const VertexAccBias*>(_vertices[1]);
    _error = VA2->estimate()-VA1->estimate();
}

void EdgeAccRW::linearizeOplus()
{
    _jacobianOplusXi = -Eigen::Matrix3d::Identity();
    _jacobianOplusXj.setIdentity();
}

Eigen::Matrix<double,6,6> EdgeAccRW::GetHessian()
{
    linearizeOplus();
    Eigen::Matrix<double,3,6> J;
    J.block<3,3>(0,0) = _jacobianOplusXi;
    J.block<3,3>(0,3) = _jacobianOplusXj;
    return J.transpose()*information()*J;
}

Eigen::Matrix3d EdgeAccRW::GetHessian2()
{
    linearizeOplus();
    return _jacobianOplusXj.transpose()*information()*_jacobianOplusXj;
}

EdgePriorPoseImu::EdgePriorPoseImu(ConstraintPoseImu *c)
{
    resize(4);
    Rwb = c->Rwb;
    twb = c->twb;
    vwb = c->vwb;
    bg = c->bg;
    ba = c->ba;
    setInformation(c->H);
}

Eigen::Matrix<double,15,15> EdgePriorPoseImu::GetHessian()
{
    linearizeOplus();
    Eigen::Matrix<double,15,15> J;
    J.block<15,6>(0,0) = _jacobianOplus[0];
    J.block<15,3>(0,6) = _jacobianOplus[1];
    J.block<15,3>(0,9) = _jacobianOplus[2];
    J.block<15,3>(0,12) = _jacobianOplus[3];
    return J.transpose()*information()*J;
}

Eigen::Matrix<double,9,9> EdgePriorPoseImu::GetHessianNoPose()
{
    linearizeOplus();
    Eigen::Matrix<double,15,9> J;
    J.block<15,3>(0,0) = _jacobianOplus[1];
    J.block<15,3>(0,3) = _jacobianOplus[2];
    J.block<15,3>(0,6) = _jacobianOplus[3];
    return J.transpose()*information()*J;
}


void EdgePriorPoseImu::computeError()
{
    const VertexPose* VP = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV = static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG = static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA = static_cast<const VertexAccBias*>(_vertices[3]);

    const Eigen::Vector3d er = LogSO3(Rwb.transpose()*VP->estimate().Rwb);
    const Eigen::Vector3d et = Rwb.transpose()*(VP->estimate().twb-twb);
    const Eigen::Vector3d ev = VV->estimate() - vwb;
    const Eigen::Vector3d ebg = VG->estimate() - bg;
    const Eigen::Vector3d eba = VA->estimate() - ba;

    _error << er, et, ev, ebg, eba;
}

void EdgePriorPoseImu::linearizeOplus()
{
    const VertexPose* VP = static_cast<const VertexPose*>(_vertices[0]);
    const Eigen::Vector3d er = LogSO3(Rwb.transpose()*VP->estimate().Rwb);
    _jacobianOplus[0].setZero();
    _jacobianOplus[0].block<3,3>(0,0) = InverseRightJacobianSO3(er);
    _jacobianOplus[0].block<3,3>(3,3) = Rwb.transpose()*VP->estimate().Rwb;
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(6,0) = Eigen::Matrix3d::Identity();
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(9,0) = Eigen::Matrix3d::Identity();
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(12,0) = Eigen::Matrix3d::Identity();
}

void EdgePriorAcc::computeError()
{
    const VertexAccBias* VA = static_cast<const VertexAccBias*>(_vertices[0]);
    _error = bprior - VA->estimate();
}

void EdgePriorAcc::linearizeOplus()
{
    // Jacobian wrt bias
    _jacobianOplusXi.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
}

Eigen::Matrix<double,3,3> EdgePriorAcc::GetHessian()
{
    linearizeOplus();
    return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
}


void EdgePriorGyro::computeError()
{
    const VertexGyroBias* VG = static_cast<const VertexGyroBias*>(_vertices[0]);
    _error = bprior - VG->estimate();
}

void EdgePriorGyro::linearizeOplus()
{
    // Jacobian wrt bias
    _jacobianOplusXi.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
}

Eigen::Matrix<double,3,3> EdgePriorGyro::GetHessian()
{
    linearizeOplus();
    return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
}

Edge4DoF::Edge4DoF(const Eigen::Matrix4d &deltaT)
{
    dTij = deltaT;
    dRij = deltaT.block<3,3>(0,0);
    dtij = deltaT.block<3,1>(0,3);
}

void Edge4DoF::computeError()
{
    const VertexPose4DoF* VPi = static_cast<const VertexPose4DoF*>(_vertices[0]);
    const VertexPose4DoF* VPj = static_cast<const VertexPose4DoF*>(_vertices[1]);
    _error << LogSO3(VPi->estimate().Rcw[0]*VPj->estimate().Rcw[0].transpose()*dRij.transpose()),
             VPi->estimate().Rcw[0]*(-VPj->estimate().Rcw[0].transpose()*VPj->estimate().tcw[0])+VPi->estimate().tcw[0] - dtij;
}

EdgeColine::EdgeColine()
{
    resize(3);
};

void EdgeColine::computeError()
{
    _error.setZero();;
    return;
    const g2o::VertexPointXYZ* p1 = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    const g2o::VertexPointXYZ* p2 = static_cast<const g2o::VertexPointXYZ*>(_vertices[1]);
    const g2o::VertexPointXYZ* p3 = static_cast<const g2o::VertexPointXYZ*>(_vertices[2]);
    Eigen::Vector3d v1 = p2->estimate() - p1->estimate();
    Eigen::Vector3d v2 = p3->estimate() - p2->estimate();
    Eigen::Vector3d v1_norm = v1.normalized();
    Eigen::Vector3d v2_norm = v2.normalized();
    _error = v1_norm.cross(v2_norm);
}

void EdgeColine::linearizeOplus()
{
    _jacobianOplus[0].setZero();
    _jacobianOplus[1].setZero();
    _jacobianOplus[2].setZero();
    return;

    const g2o::VertexPointXYZ* p1 = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    const g2o::VertexPointXYZ* p2 = static_cast<const g2o::VertexPointXYZ*>(_vertices[1]);
    const g2o::VertexPointXYZ* p3 = static_cast<const g2o::VertexPointXYZ*>(_vertices[2]);
    Eigen::Vector3d v1 = p2->estimate() - p1->estimate();
    Eigen::Vector3d v2 = p3->estimate() - p2->estimate();

    Eigen::Vector3d v1_norm = v1.normalized();
    Eigen::Vector3d v2_norm = v2.normalized();
   
    Eigen::Matrix3d jaco_r_v1, jaco_r_v2;
    jaco_r_v1 = -Skew(v2_norm) * (Eigen::Matrix3d::Identity() - v1*v1.transpose()/(v1.transpose()*v1))/v1.norm();
    jaco_r_v2 =  Skew(v1_norm) * (Eigen::Matrix3d::Identity() - v2*v2.transpose()/(v2.transpose()*v2))/v2.norm();
    
    _jacobianOplus[0] = -jaco_r_v1;
    _jacobianOplus[1] = jaco_r_v1 - jaco_r_v2;
    _jacobianOplus[2] = jaco_r_v2;
}


void EdgeSE3ProjectXYZOnlyPose::computeError() 
{
    const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    Eigen::Vector2d obs(_measurement);
    _error = obs-pCamera->project(v1->estimate().map(Xw));
}

void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
    g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
    Eigen::Vector3d xyz_trans = vi->estimate().map(Xw);

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];

    Eigen::Matrix<double,3,6> SE3deriv;
    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
                    -z , 0.0, x, 0.0, 1.0, 0.0,
                    y ,  -x , 0.0, 0.0, 0.0, 1.0;

    // Create a temporary aligned matrix for the result to avoid memory alignment issues
    Eigen::Matrix<double, 2, 3> proj_jac = pCamera->projectJac(xyz_trans);
    Eigen::Matrix<double, 2, 6> result = -proj_jac * SE3deriv;
    _jacobianOplusXi = result;
}

bool EdgeSE3ProjectXYZOnlyPose::isDepthPositive() 
{
    const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw))(2)>0.0;
}

EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, g2o::VertexSE3Expmap>() 
{}

void EdgeSE3ProjectXYZ::computeError()
{
    const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    const g2o::VertexPointXYZ* v2 = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    Eigen::Vector2d obs(_measurement);
    _error = obs-pCamera->project(v1->estimate().map(v2->estimate()));
}

bool EdgeSE3ProjectXYZ::isDepthPositive() 
{
    const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    const g2o::VertexPointXYZ* v2 = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    return ((v1->estimate().map(v2->estimate()))(2)>0.0);
}

void EdgeSE3ProjectXYZ::linearizeOplus() {
    g2o::VertexSE3Expmap * vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
    g2o::SE3Quat T(vj->estimate());
    g2o::VertexPointXYZ* vi = static_cast<g2o::VertexPointXYZ*>(_vertices[0]);
    Eigen::Vector3d xyz = vi->estimate();
    Eigen::Vector3d xyz_trans = T.map(xyz);

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];

    auto projectJac = -pCamera->projectJac(xyz_trans);

    // Use temporary variable to avoid alignment issues
    Eigen::Matrix<double, 2, 3> temp_xi = projectJac * T.rotation().toRotationMatrix();
    _jacobianOplusXi = temp_xi;

    Eigen::Matrix<double,3,6> SE3deriv;
    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
            -z , 0.0, x, 0.0, 1.0, 0.0,
            y ,  -x , 0.0, 0.0, 0.0, 1.0;

    // Use temporary variable to avoid alignment issues  
    Eigen::Matrix<double, 2, 6> temp_xj = projectJac * SE3deriv;
    _jacobianOplusXj = temp_xj;
}

VertexSim3Expmap::VertexSim3Expmap() : BaseVertex<7, g2o::Sim3>()
{
    _marginalized=false;
    _fix_scale = false;
}

EdgeSim3ProjectXYZ::EdgeSim3ProjectXYZ() :
        g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, VertexSim3Expmap>()
{}

void EdgeSim3ProjectXYZ::computeError()
{
    const VertexSim3Expmap* v1 = static_cast<const VertexSim3Expmap*>(_vertices[1]);
    const g2o::VertexPointXYZ* v2 = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);

    Eigen::Vector2d obs(_measurement);
    _error = obs-v1->pCamera1->project(v1->estimate().map(v2->estimate()));
}


EdgeInverseSim3ProjectXYZ::EdgeInverseSim3ProjectXYZ() :
        g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, VertexSim3Expmap>()
{}

void EdgeInverseSim3ProjectXYZ::computeError()
{
    const VertexSim3Expmap* v1 = static_cast<const VertexSim3Expmap*>(_vertices[1]);
    const g2o::VertexPointXYZ* v2 = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);

    Eigen::Vector2d obs(_measurement);
    _error = obs-v1->pCamera2->project((v1->estimate().inverse().map(v2->estimate())));
}