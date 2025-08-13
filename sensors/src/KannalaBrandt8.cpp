// Camera model based on https://github.com/UZ-SLAMLab/ORB_SLAM3
#include "KannalaBrandt8.h"

const float precision = 1e-6f; // Precision for floating point comparisons

KannalaBrandt8::KannalaBrandt8(const std::vector<float> &_vParameters, int width, int height, float fps)
    : GeometricCamera(_vParameters, width, height, fps)
{
    mnId = 0;
    mnType = CAM_FISHEYE;
    Eigen::Matrix3f eigenK = toK_();
    mpTvr = new TwoViewReconstruction(eigenK);
}

Eigen::Vector2d KannalaBrandt8::project(const Eigen::Vector3d &v3D)
{
    const double x2_plus_y2 = v3D[0] * v3D[0] + v3D[1] * v3D[1];
    const double theta = atan2f(sqrtf(x2_plus_y2), v3D[2]);
    const double psi = atan2f(v3D[1], v3D[0]);

    const double theta2 = theta * theta;
    const double theta3 = theta * theta2;
    const double theta5 = theta3 * theta2;
    const double theta7 = theta5 * theta2;
    const double theta9 = theta7 * theta2;
    const double r = theta + mvParameters[4] * theta3 + mvParameters[5] * theta5 + mvParameters[6] * theta7 + mvParameters[7] * theta9;
    Eigen::Vector2d res;
    res[0] = mvParameters[0] * r * cos(psi) + mvParameters[2];
    res[1] = mvParameters[1] * r * sin(psi) + mvParameters[3];
    return res;
}

Eigen::Vector2f KannalaBrandt8::project(const Eigen::Vector3f &v3D)
{
    const float x2_plus_y2 = v3D[0] * v3D[0] + v3D[1] * v3D[1];
    const float theta = atan2f(sqrtf(x2_plus_y2), v3D[2]);
    const float psi = atan2f(v3D[1], v3D[0]);
    const float theta2 = theta * theta;
    const float theta3 = theta * theta2;
    const float theta5 = theta3 * theta2;
    const float theta7 = theta5 * theta2;
    const float theta9 = theta7 * theta2;
    const float r = theta + mvParameters[4] * theta3 + mvParameters[5] * theta5 + mvParameters[6] * theta7 + mvParameters[7] * theta9;
    Eigen::Vector2f res;
    res[0] = mvParameters[0] * r * cos(psi) + mvParameters[2];
    res[1] = mvParameters[1] * r * sin(psi) + mvParameters[3];
    return res;
}

Eigen::Vector3f KannalaBrandt8::unproject(const Eigen::Vector2f &p2D)
{
   // Use Newthon method to solve for theta with good precision (err ~ e-6)
   Eigen::Vector2f pw;
   pw << (p2D[0]- mvParameters[2])/mvParameters[0], (p2D[1]- mvParameters[3])/mvParameters[1];
   float scale = 1.f;
   float theta_d = pw.norm();
   theta_d = fminf(fmaxf(-CV_PI / 2.f, theta_d), CV_PI / 2.f);

   if (theta_d > 1e-8)
   {
       // Compensate distortion iteratively
       float theta = theta_d;

       for (int j = 0; j < 10; j++)
       {
           float theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 = theta4 * theta4;
           float k0_theta2 = mvParameters[4] * theta2, k1_theta4 = mvParameters[5] * theta4;
           float k2_theta6 = mvParameters[6] * theta6, k3_theta8 = mvParameters[7] * theta8;
           float theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                             (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);
           theta = theta - theta_fix;
           if (fabsf(theta_fix) < precision)
               break;
       }
       // scale = theta - theta_d;
       scale = std::tan(theta) / theta_d;
   }
   return Eigen::Vector3f(pw[0] * scale, pw[1] * scale, 1.f);
}

Eigen::Matrix<double, 2, 3> KannalaBrandt8::projectJac(const Eigen::Vector3d &v3D)
{
    double x2 = v3D[0] * v3D[0], y2 = v3D[1] * v3D[1], z2 = v3D[2] * v3D[2];
    double r2 = x2 + y2;
    double r = sqrt(r2);
    double r3 = r2 * r;
    double theta = atan2(r, v3D[2]);

    double theta2 = theta * theta, theta3 = theta2 * theta;
    double theta4 = theta2 * theta2, theta5 = theta4 * theta;
    double theta6 = theta2 * theta4, theta7 = theta6 * theta;
    double theta8 = theta4 * theta4, theta9 = theta8 * theta;

    double f = theta + theta3 * mvParameters[4] + theta5 * mvParameters[5] + theta7 * mvParameters[6] +
               theta9 * mvParameters[7];
    double fd = 1 + 3 * mvParameters[4] * theta2 + 5 * mvParameters[5] * theta4 + 7 * mvParameters[6] * theta6 +
                9 * mvParameters[7] * theta8;

    Eigen::Matrix<double, 2, 3> JacGood;
    JacGood(0, 0) = mvParameters[0] * (fd * v3D[2] * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
    JacGood(1, 0) =
        mvParameters[1] * (fd * v3D[2] * v3D[1] * v3D[0] / (r2 * (r2 + z2)) - f * v3D[1] * v3D[0] / r3);

    JacGood(0, 1) =
        mvParameters[0] * (fd * v3D[2] * v3D[1] * v3D[0] / (r2 * (r2 + z2)) - f * v3D[1] * v3D[0] / r3);
    JacGood(1, 1) = mvParameters[1] * (fd * v3D[2] * y2 / (r2 * (r2 + z2)) + f * x2 / r3);

    JacGood(0, 2) = -mvParameters[0] * fd * v3D[0] / (r2 + z2);
    JacGood(1, 2) = -mvParameters[1] * fd * v3D[1] / (r2 + z2);

    return JacGood;
}

bool KannalaBrandt8::ReconstructWithTwoViews( const std::vector<KeyPointEx> &vKeys1, const std::vector<KeyPointEx> &vKeys2, 
                                const std::vector<int> &vMatches12, Sophus::SE3f &T21, std::vector<cv::Point3f> &vP3D, 
                                std::vector<bool> &vbTriangulated)
{
    return mpTvr->Reconstruct(vKeys1, vKeys2, vMatches12, T21, vP3D, vbTriangulated);
}

cv::Mat KannalaBrandt8::toK()
{
    cv::Mat K = (cv::Mat_<float>(3, 3) << mvParameters[0], 0.f, mvParameters[2], 0.f, mvParameters[1], mvParameters[3], 0.f, 0.f, 1.f);
    return K;
}

cv::Mat KannalaBrandt8::toD()
{
    cv::Mat K = (cv::Mat_<float>(4, 1) << mvParameters[4], mvParameters[5], mvParameters[6], mvParameters[7]);
    return K;
}

Eigen::Matrix3f KannalaBrandt8::toK_()
{
    Eigen::Matrix3f K;
    K << mvParameters[0], 0.f, mvParameters[2], 0.f, mvParameters[1], mvParameters[3], 0.f, 0.f, 1.f;
    return K;
}

int KannalaBrandt8::imWidth()
{
    return mnWdith;
}

int KannalaBrandt8::imHeight()
{
    return mnHeight;
}

bool KannalaBrandt8::epipolarConstrain(const KeyPointEx &kp1, const KeyPointEx &kp2,
                                       const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12)
{
    Eigen::Vector3f p3D;
    return this->TriangulateMatches(kp1, kp2, R12, t12, p3D) > 0.0001f;
}

float KannalaBrandt8::TriangulateMatches(const KeyPointEx &kp1, const KeyPointEx &kp2, const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12, Eigen::Vector3f &p3D)
{
    Eigen::Vector3f r1 = unproject(kp1.mPos);
    Eigen::Vector3f r2 = unproject(kp2.mPos);

    // Check parallax
    Eigen::Vector3f r21 = R12 * r2;
    const float cosParallaxRays = r1.dot(r21) / (r1.norm() * r21.norm());
    if (cosParallaxRays > 0.9998)
        return -1;

    // Parallax is good, so we try to triangulate
    cv::Point2f p11, p22;
    p11.x = r1[0];
    p11.y = r1[1];
    p22.x = r2[0];
    p22.y = r2[1];
    Eigen::Vector3f x3D;
    Eigen::Matrix<float, 3, 4> Tcw1;
    Tcw1 << Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero();

    Eigen::Matrix<float, 3, 4> Tcw2;
    Eigen::Matrix3f R21 = R12.transpose();
    Tcw2 << R21, -R21 * t12;
    Triangulate(p11, p22, Tcw1, Tcw2, x3D);
    // cv::Mat x3Dt = x3D.t();

    float z1 = x3D(2);
    if (z1 <= 0)
        return -2;

    float z2 = R21.row(2).dot(x3D) + Tcw2(2, 3);
    if (z2 <= 0)
        return -3;

    // Check reprojection error
    Eigen::Vector2f err1 = project(x3D) - kp1.mPos;
    if (err1[0]*err1[0] + err1[1]*err1[1] > 5.991)
        return -4;

    Eigen::Vector3f x3D2 = R21 * x3D + Tcw2.col(3);
    Eigen::Vector2f err2 = project(x3D2) - kp2.mPos;
    if (err2[0]*err2[0] + err2[1]*err2[1] > 5.991)
        return -5;
    p3D = x3D;
    return z1;
}

void KannalaBrandt8::Triangulate(const cv::Point2f &p1, const cv::Point2f &p2, const Eigen::Matrix<float, 3, 4> &Tcw1,
                                 const Eigen::Matrix<float, 3, 4> &Tcw2, Eigen::Vector3f &x3D)
{
    Eigen::Matrix<float, 4, 4> A;
    A.row(0) = p1.x * Tcw1.row(2) - Tcw1.row(0);
    A.row(1) = p1.y * Tcw1.row(2) - Tcw1.row(1);
    A.row(2) = p2.x * Tcw2.row(2) - Tcw2.row(0);
    A.row(3) = p2.y * Tcw2.row(2) - Tcw2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4f x3Dh = svd.matrixV().col(3);
    x3D = x3Dh.head(3) / x3Dh(3);
}