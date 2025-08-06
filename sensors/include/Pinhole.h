#pragma once
#include <assert.h>
#include "GeometricCamera.h"

class Pinhole : public GeometricCamera
{
public:
    Pinhole(const std::vector<float> &_vParameters, int width, int height, float fps);

    Eigen::Vector2d project(const Eigen::Vector3d &v3D);
    Eigen::Vector2f project(const Eigen::Vector3f &v3D);

    Eigen::Vector3f unproject(const Eigen::Vector2f &p2D);

    Eigen::Matrix<double, 2, 3> projectJac(const Eigen::Vector3d &v3D);

    bool ReconstructWithTwoViews(const std::vector<KeyPointEx> &vKeys1, const std::vector<KeyPointEx> &vKeys2, const std::vector<int> &vMatches12,
                                 Sophus::SE3f &T21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

    cv::Mat toK();
    cv::Mat toD();
    Eigen::Matrix3f toK_();
    int imWidth();
    int imHeight();

    bool epipolarConstrain(const KeyPointEx &kp1, const KeyPointEx &kp2, const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12);

private:
    TwoViewReconstruction * mpTvr; 
};