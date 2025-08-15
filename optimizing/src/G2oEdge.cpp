#include "G2oEdge.h"

// ============================================================================
//                        VISUAL REPROJECTION EDGES
// ============================================================================

void EdgeMono::computeError()
{
    const auto* vertex_point = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    const auto* vertex_pose = static_cast<const VertexPose*>(_vertices[1]);
    const Eigen::Vector2d observation(_measurement);
    
    _error = observation - vertex_pose->estimate().Project(vertex_point->estimate(), cam_idx);
}

void EdgeMono::linearizeOplus()
{
    const auto* vertex_pose = static_cast<const VertexPose*>(_vertices[1]);
    const auto* vertex_point = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);

    // Camera frame transformations
    const Eigen::Matrix3d& Rcw = vertex_pose->estimate().Rcw[cam_idx];
    const Eigen::Vector3d& tcw = vertex_pose->estimate().tcw[cam_idx];
    const Eigen::Vector3d Xc = Rcw * vertex_point->estimate() + tcw;
    
    // Body frame transformations
    const Eigen::Vector3d Xb = vertex_pose->estimate().Rbc[cam_idx] * Xc + 
                              vertex_pose->estimate().tbc[cam_idx];
    const Eigen::Matrix3d& Rcb = vertex_pose->estimate().Rcb[cam_idx];

    // Projection jacobian
    const Eigen::Matrix<double, 2, 3> proj_jacobian = 
        vertex_pose->estimate().pCamera[cam_idx]->projectJac(Xc);
    _jacobianOplusXi = -proj_jacobian * Rcw;

    // SE3 derivative matrix
    Eigen::Matrix<double, 3, 6> SE3_derivative;
    const double x = Xb(0);
    const double y = Xb(1);
    const double z = Xb(2);
    
    SE3_derivative << 0.0,   z, -y, 1.0, 0.0, 0.0,
                     -z,  0.0,  x, 0.0, 1.0, 0.0,
                      y,  -x, 0.0, 0.0, 0.0, 1.0;

    _jacobianOplusXj = proj_jacobian * Rcb * SE3_derivative;
}

bool EdgeMono::isDepthPositive()
{
    const auto* vertex_point = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    const auto* vertex_pose = static_cast<const VertexPose*>(_vertices[1]);
    return vertex_pose->estimate().isDepthPositive(vertex_point->estimate(), cam_idx);
}

Eigen::Matrix<double, 2, 9> EdgeMono::GetJacobian()
{
    linearizeOplus();
    Eigen::Matrix<double, 2, 9> jacobian;
    jacobian.block<2, 3>(0, 0) = _jacobianOplusXi;
    jacobian.block<2, 6>(0, 3) = _jacobianOplusXj;
    return jacobian;
}

Eigen::Matrix<double, 9, 9> EdgeMono::GetHessian()
{
    const Eigen::Matrix<double, 2, 9> jacobian = GetJacobian();
    return jacobian.transpose() * information() * jacobian;
}

void EdgeMonoOnlyPose::computeError()
{
    const auto* vertex_pose = static_cast<const VertexPose*>(_vertices[0]);
    const Eigen::Vector2d observation(_measurement);
    _error = observation - vertex_pose->estimate().Project(Xw, cam_idx);
}

bool EdgeMonoOnlyPose::isDepthPositive()
{
    const auto* vertex_pose = static_cast<const VertexPose*>(_vertices[0]);
    return vertex_pose->estimate().isDepthPositive(Xw, cam_idx);
}

Eigen::Matrix<double, 6, 6> EdgeMonoOnlyPose::GetHessian()
{
    linearizeOplus();
    return _jacobianOplusXi.transpose() * information() * _jacobianOplusXi;
}

void EdgeMonoOnlyPose::linearizeOplus()
{
    const auto* vertex_pose = static_cast<const VertexPose*>(_vertices[0]);

    // Camera frame transformations  
    const Eigen::Matrix3d& Rcw = vertex_pose->estimate().Rcw[cam_idx];
    const Eigen::Vector3d& tcw = vertex_pose->estimate().tcw[cam_idx];
    const Eigen::Vector3d Xc = Rcw * Xw + tcw;
    
    // Body frame transformations
    const Eigen::Vector3d Xb = vertex_pose->estimate().Rbc[cam_idx] * Xc + 
                              vertex_pose->estimate().tbc[cam_idx];
    const Eigen::Matrix3d& Rcb = vertex_pose->estimate().Rcb[cam_idx];

    // Projection jacobian
    const Eigen::Matrix<double, 2, 3> proj_jacobian = 
        vertex_pose->estimate().pCamera[cam_idx]->projectJac(Xc);

    // SE3 derivative matrix
    Eigen::Matrix<double, 3, 6> SE3_derivative;
    const double x = Xb(0);
    const double y = Xb(1);
    const double z = Xb(2);
    
    SE3_derivative << 0.0,   z, -y, 1.0, 0.0, 0.0,
                     -z,  0.0,  x, 0.0, 1.0, 0.0,
                      y,  -x, 0.0, 0.0, 0.0, 1.0;
                      
    _jacobianOplusXi = proj_jacobian * Rcb * SE3_derivative;
}

// ============================================================================
//                          INERTIAL MEASUREMENT EDGES  
// ============================================================================

EdgeInertial::EdgeInertial(IMU::Preintegrated* pInt)
    : JRg(pInt->JRg.cast<double>())
    , JVg(pInt->JVg.cast<double>())
    , JPg(pInt->JPg.cast<double>())
    , JVa(pInt->JVa.cast<double>())
    , JPa(pInt->JPa.cast<double>())
    , mpInt(pInt)
    , dt(pInt->dT)
{
    // This edge connects 6 vertices: [pose1, vel1, gyro_bias1, acc_bias1, pose2, vel2]
    resize(6);
    
    // Set gravity vector
    g << 0, 0, -IMU::GRAVITY_VALUE;

    // Compute information matrix from covariance
    Eigen::Matrix<double, 9, 9> information_matrix = 
        pInt->C.block<9, 9>(0, 0).cast<double>().inverse();
    information_matrix = (information_matrix + information_matrix.transpose()) / 2.0;
    
    // Ensure positive definiteness
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> eigen_solver(information_matrix);
    Eigen::Matrix<double, 9, 1> eigenvalues = eigen_solver.eigenvalues();
    
    for (int i = 0; i < 9; i++) {
        if (eigenvalues[i] < 1e-12) {
            eigenvalues[i] = 0;
        }
    }
    
    information_matrix = eigen_solver.eigenvectors() * 
                        eigenvalues.asDiagonal() * 
                        eigen_solver.eigenvectors().transpose();
    setInformation(information_matrix);
}

void EdgeInertial::computeError()
{
    // Extract vertices
    const auto* vertex_pose1 = static_cast<const VertexPose*>(_vertices[0]);
    const auto* vertex_vel1 = static_cast<const VertexVelocity*>(_vertices[1]);
    const auto* vertex_gyro_bias1 = static_cast<const VertexGyroBias*>(_vertices[2]);
    const auto* vertex_acc_bias1 = static_cast<const VertexAccBias*>(_vertices[3]);
    const auto* vertex_pose2 = static_cast<const VertexPose*>(_vertices[4]);
    const auto* vertex_vel2 = static_cast<const VertexVelocity*>(_vertices[5]);
    
    // Construct bias from estimates
    const IMU::Bias bias1(vertex_acc_bias1->estimate()[0], vertex_acc_bias1->estimate()[1], vertex_acc_bias1->estimate()[2],
                         vertex_gyro_bias1->estimate()[0], vertex_gyro_bias1->estimate()[1], vertex_gyro_bias1->estimate()[2]);
    
    // Get preintegrated measurements with current bias
    const Eigen::Matrix3d delta_rotation = mpInt->GetDeltaRotation(bias1).cast<double>();
    const Eigen::Vector3d delta_velocity = mpInt->GetDeltaVelocity(bias1).cast<double>();
    const Eigen::Vector3d delta_position = mpInt->GetDeltaPosition(bias1).cast<double>();

    // Compute rotation error
    const Eigen::Vector3d rotation_error = LogSO3(
        delta_rotation.transpose() * 
        vertex_pose1->estimate().Rwb.transpose() * 
        vertex_pose2->estimate().Rwb
    );
    
    // Compute velocity error  
    const Eigen::Vector3d velocity_error = 
        vertex_pose1->estimate().Rwb.transpose() * 
        (vertex_vel2->estimate() - vertex_vel1->estimate() - g * dt) - delta_velocity;
    
    // Compute position error
    const Eigen::Vector3d position_error = 
        vertex_pose1->estimate().Rwb.transpose() * 
        (vertex_pose2->estimate().twb - vertex_pose1->estimate().twb - 
         vertex_vel1->estimate() * dt - 0.5 * g * dt * dt) - delta_position;

    _error << rotation_error, velocity_error, position_error;
}

void EdgeInertial::linearizeOplus()
{
    // Extract vertices
    const auto* vertex_pose1 = static_cast<const VertexPose*>(_vertices[0]);
    const auto* vertex_vel1 = static_cast<const VertexVelocity*>(_vertices[1]);
    const auto* vertex_gyro_bias1 = static_cast<const VertexGyroBias*>(_vertices[2]);
    const auto* vertex_acc_bias1 = static_cast<const VertexAccBias*>(_vertices[3]);
    const auto* vertex_pose2 = static_cast<const VertexPose*>(_vertices[4]);
    const auto* vertex_vel2 = static_cast<const VertexVelocity*>(_vertices[5]);
    
    // Construct bias and bias delta
    const IMU::Bias bias1(vertex_acc_bias1->estimate()[0], vertex_acc_bias1->estimate()[1], vertex_acc_bias1->estimate()[2],
                         vertex_gyro_bias1->estimate()[0], vertex_gyro_bias1->estimate()[1], vertex_gyro_bias1->estimate()[2]);
    const IMU::Bias delta_bias = mpInt->GetDeltaBias(bias1);
    Eigen::Vector3d delta_bias_gyro;
    delta_bias_gyro << delta_bias.bwx, delta_bias.bwy, delta_bias.bwz;

    // Rotation matrices
    const Eigen::Matrix3d Rwb1 = vertex_pose1->estimate().Rwb;
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = vertex_pose2->estimate().Rwb;

    // Rotation terms for jacobian computation
    const Eigen::Matrix3d delta_rotation = mpInt->GetDeltaRotation(bias1).cast<double>();
    const Eigen::Matrix3d error_rotation = delta_rotation.transpose() * Rbw1 * Rwb2;
    const Eigen::Vector3d rotation_error = LogSO3(error_rotation);
    const Eigen::Matrix3d inverse_right_jacobian = InverseRightJacobianSO3(rotation_error);

    // Jacobians w.r.t. Pose 1
    _jacobianOplus[0].setZero();
    // Rotation component
    _jacobianOplus[0].block<3, 3>(0, 0) = -inverse_right_jacobian * Rwb2.transpose() * Rwb1;
    _jacobianOplus[0].block<3, 3>(3, 0) = SO3d::hat(Rbw1 * (vertex_vel2->estimate() - vertex_vel1->estimate() - g * dt));
    _jacobianOplus[0].block<3, 3>(6, 0) = SO3d::hat(Rbw1 * (vertex_pose2->estimate().twb - vertex_pose1->estimate().twb -
                                                                     vertex_vel1->estimate() * dt - 0.5 * g * dt * dt));
    // Translation component
    _jacobianOplus[0].block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity();

    // Jacobians w.r.t. Velocity 1
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3, 3>(3, 0) = -Rbw1;
    _jacobianOplus[1].block<3, 3>(6, 0) = -Rbw1 * dt;

    // Jacobians w.r.t. Gyro Bias 1
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3, 3>(0, 0) = -inverse_right_jacobian * error_rotation.transpose() * 
                                          RightJacobianSO3(JRg * delta_bias_gyro) * JRg;
    _jacobianOplus[2].block<3, 3>(3, 0) = -JVg;
    _jacobianOplus[2].block<3, 3>(6, 0) = -JPg;

    // Jacobians w.r.t. Accelerometer Bias 1
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3, 3>(3, 0) = -JVa;
    _jacobianOplus[3].block<3, 3>(6, 0) = -JPa;

    // Jacobians w.r.t. Pose 2
    _jacobianOplus[4].setZero();
    // Rotation component
    _jacobianOplus[4].block<3, 3>(0, 0) = inverse_right_jacobian;
    // Translation component
    _jacobianOplus[4].block<3, 3>(6, 3) = Rbw1 * Rwb2;

    // Jacobians w.r.t. Velocity 2
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3, 3>(3, 0) = Rbw1;
}

EdgeInertialGS::EdgeInertialGS(IMU::Preintegrated* pInt)
    : JRg(pInt->JRg.cast<double>())
    , JVg(pInt->JVg.cast<double>())
    , JPg(pInt->JPg.cast<double>())
    , JVa(pInt->JVa.cast<double>())
    , JPa(pInt->JPa.cast<double>())
    , mpInt(pInt)
    , dt(pInt->dT)
{
    // This edge connects 8 vertices: [pose1, vel1, gyro_bias, acc_bias, pose2, vel2, gravity_dir, scale]
    resize(8);
    
    // Set initial gravity vector
    gI << 0, 0, -IMU::GRAVITY_VALUE;

    // Compute information matrix from covariance
    Eigen::Matrix<double, 9, 9> information_matrix = 
        pInt->C.block<9, 9>(0, 0).cast<double>().inverse();
    information_matrix = (information_matrix + information_matrix.transpose()) / 2.0;
    
    // Ensure positive definiteness
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> eigen_solver(information_matrix);
    Eigen::Matrix<double, 9, 1> eigenvalues = eigen_solver.eigenvalues();
    
    for (int i = 0; i < 9; i++) {
        if (eigenvalues[i] < 1e-12) {
            eigenvalues[i] = 0;
        }
    }
    
    information_matrix = eigen_solver.eigenvectors() * 
                        eigenvalues.asDiagonal() * 
                        eigen_solver.eigenvectors().transpose();
    setInformation(information_matrix);
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
    _jacobianOplus[7].block<3, 1>(3, 0) = Rbw1 * (VV2->estimate() - VV1->estimate());
    _jacobianOplus[7].block<3, 1>(6, 0) = Rbw1 * (VP2->estimate().twb - 
                                                  VP1->estimate().twb - 
                                                  VV1->estimate() * dt);
}

// ============================================================================
//                         BIAS RANDOM WALK EDGES
// ============================================================================

void EdgeGyroRW::computeError()
{
    const auto* vertex_gyro_bias1 = static_cast<const VertexGyroBias*>(_vertices[0]);
    const auto* vertex_gyro_bias2 = static_cast<const VertexGyroBias*>(_vertices[1]);
    _error = vertex_gyro_bias2->estimate() - vertex_gyro_bias1->estimate();
}

void EdgeGyroRW::linearizeOplus()
{
    _jacobianOplusXi = -Eigen::Matrix3d::Identity();
    _jacobianOplusXj = Eigen::Matrix3d::Identity();
}

Eigen::Matrix<double, 6, 6> EdgeGyroRW::GetHessian()
{
    linearizeOplus();
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.block<3, 3>(0, 0) = _jacobianOplusXi;
    jacobian.block<3, 3>(0, 3) = _jacobianOplusXj;
    return jacobian.transpose() * information() * jacobian;
}

Eigen::Matrix3d EdgeGyroRW::GetHessian2()
{
    linearizeOplus();
    return _jacobianOplusXj.transpose() * information() * _jacobianOplusXj;
}

void EdgeAccRW::computeError()
{
    const auto* vertex_acc_bias1 = static_cast<const VertexAccBias*>(_vertices[0]);
    const auto* vertex_acc_bias2 = static_cast<const VertexAccBias*>(_vertices[1]);
    _error = vertex_acc_bias2->estimate() - vertex_acc_bias1->estimate();
}

void EdgeAccRW::linearizeOplus()
{
    _jacobianOplusXi = -Eigen::Matrix3d::Identity();
    _jacobianOplusXj = Eigen::Matrix3d::Identity();
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
    const auto* vertex_acc_bias = static_cast<const VertexAccBias*>(_vertices[0]);
    _error = bprior - vertex_acc_bias->estimate();
}

void EdgePriorAcc::linearizeOplus()
{
    // Jacobian with respect to accelerometer bias
    _jacobianOplusXi.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
}

Eigen::Matrix<double,3,3> EdgePriorAcc::GetHessian()
{
    linearizeOplus();
    return _jacobianOplusXi.transpose() * information() * _jacobianOplusXi;
}


void EdgePriorGyro::computeError()
{
    const auto* vertex_gyro_bias = static_cast<const VertexGyroBias*>(_vertices[0]);
    _error = bprior - vertex_gyro_bias->estimate();
}

void EdgePriorGyro::linearizeOplus()
{
    // Jacobian with respect to gyroscope bias
    _jacobianOplusXi.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
}

Eigen::Matrix<double,3,3> EdgePriorGyro::GetHessian()
{
    linearizeOplus();
    return _jacobianOplusXi.transpose() * information() * _jacobianOplusXi;
}

Edge4DoF::Edge4DoF(const Eigen::Matrix4d &deltaT)
{
    dTij = deltaT;
    dRij = deltaT.block<3,3>(0,0);
    dtij = deltaT.block<3,1>(0,3);
}

void Edge4DoF::computeError()
{
    const auto* vertex_pose_i = static_cast<const VertexPose4DoF*>(_vertices[0]);
    const auto* vertex_pose_j = static_cast<const VertexPose4DoF*>(_vertices[1]);
    
    // Rotation error: log(R_i * R_j^T * dR_ij^T)
    auto rotation_error = LogSO3(vertex_pose_i->estimate().Rcw[0] * 
                                vertex_pose_j->estimate().Rcw[0].transpose() * 
                                dRij.transpose());
    
    // Translation error: R_i * (-R_j^T * t_j) + t_i - dt_ij
    auto translation_error = vertex_pose_i->estimate().Rcw[0] * 
                           (-vertex_pose_j->estimate().Rcw[0].transpose() * 
                            vertex_pose_j->estimate().tcw[0]) + 
                           vertex_pose_i->estimate().tcw[0] - dtij;
    
    _error << rotation_error, translation_error;
}

EdgeColine::EdgeColine()
{
    resize(3);
}

void EdgeColine::computeError()
{
    _error.setZero();
    return;
    
    const auto* point1 = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    const auto* point2 = static_cast<const g2o::VertexPointXYZ*>(_vertices[1]);
    const auto* point3 = static_cast<const g2o::VertexPointXYZ*>(_vertices[2]);
    
    Eigen::Vector3d v1 = point2->estimate() - point1->estimate();
    Eigen::Vector3d v2 = point3->estimate() - point2->estimate();
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

    const auto* point1 = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    const auto* point2 = static_cast<const g2o::VertexPointXYZ*>(_vertices[1]);
    const auto* point3 = static_cast<const g2o::VertexPointXYZ*>(_vertices[2]);
    
    Eigen::Vector3d v1 = point2->estimate() - point1->estimate();
    Eigen::Vector3d v2 = point3->estimate() - point2->estimate();
    Eigen::Vector3d v1_norm = v1.normalized();
    Eigen::Vector3d v2_norm = v2.normalized();
   
    Eigen::Matrix3d jacobian_r_v1, jacobian_r_v2;
    jacobian_r_v1 = -Skew(v2_norm) * 
                   (Eigen::Matrix3d::Identity() - v1*v1.transpose()/(v1.transpose()*v1))/v1.norm();
    jacobian_r_v2 = Skew(v1_norm) * 
                   (Eigen::Matrix3d::Identity() - v2*v2.transpose()/(v2.transpose()*v2))/v2.norm();
    
    _jacobianOplus[0] = -jacobian_r_v1;
    _jacobianOplus[1] = jacobian_r_v1 - jacobian_r_v2;
    _jacobianOplus[2] = jacobian_r_v2;
}


// ============================================================================
//                         SE3 PROJECTION EDGES
// ============================================================================

void EdgeSE3ProjectXYZOnlyPose::computeError() 
{
    const auto* vertex_pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    Eigen::Vector2d observation(_measurement);
    _error = observation - pCamera->project(vertex_pose->estimate().map(Xw));
}

void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() 
{
    auto* vertex_pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
    Eigen::Vector3d xyz_transformed = vertex_pose->estimate().map(Xw);

    double x = xyz_transformed[0];
    double y = xyz_transformed[1];
    double z = xyz_transformed[2];

    Eigen::Matrix<double,3,6> se3_derivative;
    se3_derivative << 0.0,    z,   -y, 1.0, 0.0, 0.0,
                     -z,   0.0,    x, 0.0, 1.0, 0.0,
                      y,   -x,   0.0, 0.0, 0.0, 1.0;

    _jacobianOplusXi = -pCamera->projectJac(xyz_transformed) * se3_derivative;
}

bool EdgeSE3ProjectXYZOnlyPose::isDepthPositive() 
{
    const auto* vertex_pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    return (vertex_pose->estimate().map(Xw))(2) > 0.0;
}

EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() 
    : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, g2o::VertexSE3Expmap>() 
{
}

void EdgeSE3ProjectXYZ::computeError()
{
    const auto* vertex_pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    const auto* vertex_point = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    Eigen::Vector2d observation(_measurement);
    _error = observation - pCamera->project(vertex_pose->estimate().map(vertex_point->estimate()));
}

bool EdgeSE3ProjectXYZ::isDepthPositive() 
{
    const auto* vertex_pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    const auto* vertex_point = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    return ((vertex_pose->estimate().map(vertex_point->estimate()))(2) > 0.0);
}

void EdgeSE3ProjectXYZ::linearizeOplus() 
{
    auto* vertex_pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[1]);
    g2o::SE3Quat transform(vertex_pose->estimate());
    auto* vertex_point = static_cast<g2o::VertexPointXYZ*>(_vertices[0]);
    Eigen::Vector3d xyz = vertex_point->estimate();
    Eigen::Vector3d xyz_transformed = transform.map(xyz);

    double x = xyz_transformed[0];
    double y = xyz_transformed[1];
    double z = xyz_transformed[2];

    auto project_jacobian = -pCamera->projectJac(xyz_transformed);

    _jacobianOplusXi = project_jacobian * transform.rotation().toRotationMatrix();

    Eigen::Matrix<double,3,6> se3_derivative;
    se3_derivative << 0.0,    z,   -y, 1.0, 0.0, 0.0,
                     -z,   0.0,    x, 0.0, 1.0, 0.0,
                      y,   -x,   0.0, 0.0, 0.0, 1.0;

    _jacobianOplusXj = project_jacobian * se3_derivative;
}

// ============================================================================
//                         SIM3 VERTEX AND PROJECTION EDGES
// ============================================================================

VertexSim3Expmap::VertexSim3Expmap() : BaseVertex<7, g2o::Sim3>()
{
    _marginalized = false;
    _fix_scale = false;
}

EdgeSim3ProjectXYZ::EdgeSim3ProjectXYZ() :
    g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, VertexSim3Expmap>()
{
}

void EdgeSim3ProjectXYZ::computeError()
{
    const auto* vertex_sim3 = static_cast<const VertexSim3Expmap*>(_vertices[1]);
    const auto* vertex_point = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);

    Eigen::Vector2d observation(_measurement);
    _error = observation - vertex_sim3->pCamera1->project(vertex_sim3->estimate().map(vertex_point->estimate()));
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