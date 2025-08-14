#include "GeometricCamera.h"
#include <opencv2/opencv.hpp>
#include <algorithm>

bool GeometricCamera::IsInImage(const float &x, const float &y) const
{
    return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

void GeometricCamera::InitializeImageBounds()
{
    if(mnType == CAM_PINHOLE)
    {
        // For pinhole cameras, handle distortion properly
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imWidth(); mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imHeight();
        mat.at<float>(3,0)=imWidth(); mat.at<float>(3,1)=imHeight();

        mat=mat.reshape(2);
        cv::Mat K = toK();
        cv::Mat D = toD();
        cv::undistortPoints(mat,mat,K,D,cv::Mat(),K);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = (int)std::min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = (int)std::max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = (int)std::min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = (int)std::max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        // For fisheye cameras, use simple image boundaries
        mnMinX = 0;
        mnMinY = 0;
        mnMaxX = imWidth();
        mnMaxY = imHeight();
    }
    
    // Set grid parameters
    mnGridCols = FRAME_GRID_COLS;
    mnGridRows = FRAME_GRID_ROWS;
    mfGridElementWidthInv = static_cast<float>(mnGridCols) / static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(mnGridRows) / static_cast<float>(mnMaxY - mnMinY);
}
