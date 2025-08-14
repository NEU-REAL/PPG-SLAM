// Camera model based on https://github.com/UZ-SLAMLab/ORB_SLAM3
#include "Pinhole.h"

Pinhole::Pinhole(const std::vector<float> &_vParameters, int width, int height, float fps)
    : GeometricCamera(_vParameters, width, height, fps)
{
    mnId = 0;
    mnType = CAM_PINHOLE;
    Eigen::Matrix3f eigenK = toK_();
    mpTvr = new TwoViewReconstruction(eigenK);
    InitializeImageBounds();
}

Eigen::Vector2d Pinhole::project(const Eigen::Vector3d &v3D)
{
    Eigen::Vector2d res;
    res[0] = mvParameters[0] * v3D[0] / v3D[2] + mvParameters[2];
    res[1] = mvParameters[1] * v3D[1] / v3D[2] + mvParameters[3];
    return res;
}

Eigen::Vector2f Pinhole::project(const Eigen::Vector3f &v3D)
{
    Eigen::Vector2f res;
    res[0] = mvParameters[0] * v3D[0] / v3D[2] + mvParameters[2];
    res[1] = mvParameters[1] * v3D[1] / v3D[2] + mvParameters[3];
    return res;
}

Eigen::Vector3f Pinhole::unproject(const Eigen::Vector2f &p2D)
{
    return Eigen::Vector3f((p2D[0] - mvParameters[2]) / mvParameters[0], (p2D[1] - mvParameters[3]) / mvParameters[1], 1.f);
}

Eigen::Matrix<double, 2, 3> Pinhole::projectJac(const Eigen::Vector3d &v3D)
{
    Eigen::Matrix<double, 2, 3> Jac;
    Jac(0, 0) = mvParameters[0] / v3D[2];
    Jac(0, 1) = 0.f;
    Jac(0, 2) = -mvParameters[0] * v3D[0] / (v3D[2] * v3D[2]);
    Jac(1, 0) = 0.f;
    Jac(1, 1) = mvParameters[1] / v3D[2];
    Jac(1, 2) = -mvParameters[1] * v3D[1] / (v3D[2] * v3D[2]);

    return Jac;
}

bool Pinhole::ReconstructWithTwoViews(const std::vector<KeyPointEx> &vKeys1, const std::vector<KeyPointEx> &vKeys2, const std::vector<int> &vMatches12,
                                      Sophus::SE3f &T21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated)
{
    return mpTvr->Reconstruct(vKeys1, vKeys2, vMatches12, T21, vP3D, vbTriangulated);
}

cv::Mat Pinhole::toK()
{
    cv::Mat K = (cv::Mat_<float>(3, 3) << mvParameters[0], 0.f, mvParameters[2], 0.f, mvParameters[1], mvParameters[3], 0.f, 0.f, 1.f);
    return K;
}

cv::Mat Pinhole::toD()
{
    cv::Mat K = (cv::Mat_<float>(4, 1) << mvParameters[4], mvParameters[5], mvParameters[6], mvParameters[7]);
    return K;
}

int Pinhole::imWidth()
{
    return mnWdith;
}
int Pinhole::imHeight()
{
    return mnHeight;
}

Eigen::Matrix3f Pinhole::toK_()
{
    Eigen::Matrix3f K;
    K << mvParameters[0], 0.f, mvParameters[2], 0.f, mvParameters[1], mvParameters[3], 0.f, 0.f, 1.f;
    return K;
}

bool Pinhole::epipolarConstrain(const KeyPointEx &kp1, const KeyPointEx &kp2, const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12)
{
    // Compute Fundamental Matrix
    Eigen::Matrix3f t12x = Sophus::SO3f::hat(t12);
    Eigen::Matrix3f K1 = toK_();
    Eigen::Matrix3f K2 = toK_();
    Eigen::Matrix3f F12 = K1.transpose().inverse() * t12x * R12 * K2.inverse();
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.mPos[0] * F12(0, 0) + kp1.mPos[1] * F12(1, 0) + F12(2, 0);
    const float b = kp1.mPos[0] * F12(0, 1) + kp1.mPos[1] * F12(1, 1) + F12(2, 1);
    const float c = kp1.mPos[0] * F12(0, 2) + kp1.mPos[1] * F12(1, 2) + F12(2, 2);
    const float num = a * kp2.mPos[0] + b * kp2.mPos[1] + c;
    const float den = a * a + b * b;
    if (den == 0)
        return false;
    const float dsqr = num * num / den;
    return dsqr < 3.84;
}