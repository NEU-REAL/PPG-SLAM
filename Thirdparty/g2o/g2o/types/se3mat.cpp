#include "se3mat.h"

namespace g2o {


void SE3mat::Retract(const Eigen::Vector3d dr, const Eigen::Vector3d &dt)
{
    t += R*dt;
    R = R*ExpSO3(dr);
}

Eigen::Matrix3d SE3mat::ExpSO3(const Eigen::Vector3d r)
{
    Eigen::Matrix3d W;
    W << 0, -r[2], r[1],
         r[2], 0, -r[0],
         -r[1], r[0], 0;

    const double theta = r.norm();

    if(theta<1e-6)
        return Eigen::Matrix3d::Identity() + W + 0.5l*W*W;
    else
        return Eigen::Matrix3d::Identity() + W*sin(theta)/theta + W*W*(1-cos(theta))/(theta*theta);
}

Eigen::Vector3d SE3mat::LogSO3(const Eigen::Matrix3d R)
{
    const double tr = R(0,0)+R(1,1)+R(2,2);
    
    // Clamp trace to valid range for acos
    double clamped_arg = std::max(-1.0, std::min(1.0, (tr-1.0)*0.5));
    const double theta = acos(clamped_arg);
    
    Eigen::Vector3d w;
    w << R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1);
    
    if(theta < 1e-6)
        return 0.5 * w;
    else {
        double sin_theta = sin(theta);
        if (abs(sin_theta) < 1e-6) {
            // Handle singularity when theta ≈ π
            return 0.5 * w;
        }
        return theta * w / (2.0 * sin_theta);
    }
}

} //namespace g2o
