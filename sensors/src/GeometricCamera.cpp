/**
 * @file GeometricCamera.cpp
 * @brief Implementation of geometric camera interface
 */

#include "GeometricCamera.h"
#include <opencv2/opencv.hpp>
#include <algorithm>

void KeyPointEx::updatePreturb(float deltX, float deltY, float sc)
{
    if (mfScore > sc)
        return;
    mPos += Eigen::Vector2f(deltX, deltY);
    mfScore = sc;
}

GeometricCamera::GeometricCamera(const std::vector<float>& parameters, int width, int height, float fps) 
    : mvParameters(parameters), mnWidth(width), mnHeight(height), mfFps(fps) {}

bool GeometricCamera::IsInImage(const float& x, const float& y) const
{
    return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

void GeometricCamera::InitializeImageBounds()
{
    if (mnType == CAM_PINHOLE) {
        // Handle distortion for pinhole cameras
        cv::Mat corners(4, 2, CV_32F);
        corners.at<float>(0, 0) = 0.0;        corners.at<float>(0, 1) = 0.0;
        corners.at<float>(1, 0) = imWidth();  corners.at<float>(1, 1) = 0.0;
        corners.at<float>(2, 0) = 0.0;        corners.at<float>(2, 1) = imHeight();
        corners.at<float>(3, 0) = imWidth();  corners.at<float>(3, 1) = imHeight();

        corners = corners.reshape(2);
        cv::Mat K = toK();
        cv::Mat D = toD();
        cv::undistortPoints(corners, corners, K, D, cv::Mat(), K);
        corners = corners.reshape(1);

        // Set undistorted bounds
        mnMinX = static_cast<int>(std::min(corners.at<float>(0, 0), corners.at<float>(2, 0)));
        mnMaxX = static_cast<int>(std::max(corners.at<float>(1, 0), corners.at<float>(3, 0)));
        mnMinY = static_cast<int>(std::min(corners.at<float>(0, 1), corners.at<float>(1, 1)));
        mnMaxY = static_cast<int>(std::max(corners.at<float>(2, 1), corners.at<float>(3, 1)));
    }
    else {
        // Use simple bounds for fisheye cameras
        mnMinX = 0;
        mnMinY = 0;
        mnMaxX = imWidth();
        mnMaxY = imHeight();
    }
    
    // Initialize grid parameters
    mnGridCols = FRAME_GRID_COLS;
    mnGridRows = FRAME_GRID_ROWS;
    mfGridElementWidthInv = static_cast<float>(mnGridCols) / static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(mnGridRows) / static_cast<float>(mnMaxY - mnMinY);
}
