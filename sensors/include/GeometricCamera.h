#pragma once
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Geometry>
#include "TwoViewReconstruction.h"

class KeyPointEx
{
public:
    KeyPointEx()
    {};
    KeyPointEx(float x_, float y_, float sc): mPos(x_,y_), mfScore(sc), mbOut(true)
    {};
    void updatePreturb(float deltX, float deltY, float sc)
    {
        if(mfScore > sc)
            return;
        mPos = mPos + Eigen::Vector2f(deltX,deltY);
        mfScore = sc;
    };
public:
    Eigen::Vector2f mPos, mPosUn;
    float mfScore;
    std::vector<unsigned int> mvConnected; //!< connected lines
    std::vector<std::pair<unsigned int, unsigned int>> mvColine; //!< connected colinear point pairs
    bool mbOut;
};

class GeometricCamera
{
public:
    GeometricCamera(const std::vector<float> &_vParameters, int width, int height, float fps) : 
        mvParameters(_vParameters), mnWdith(width),mnHeight(height), mfFps(fps) {}

    virtual Eigen::Vector2d project(const Eigen::Vector3d &v3D) = 0;  // for g2o
    virtual Eigen::Vector2f project(const Eigen::Vector3f &v3D) = 0;  // for OpenCV

    virtual Eigen::Vector3f unproject(const Eigen::Vector2f &p2D) = 0;

    virtual Eigen::Matrix<double, 2, 3> projectJac(const Eigen::Vector3d &v3D) = 0;

    virtual bool ReconstructWithTwoViews( const std::vector<KeyPointEx> &vKeys1, const std::vector<KeyPointEx> &vKeys2, 
                            const std::vector<int> &vMatches12, Sophus::SE3f &T21, std::vector<cv::Point3f> &vP3D, 
                            std::vector<bool> &vbTriangulated) = 0;

    virtual cv::Mat toK() = 0;
    virtual cv::Mat toD() = 0;
    virtual Eigen::Matrix3f toK_() = 0;
    virtual int imWidth() = 0;
    virtual int imHeight() = 0;

    virtual bool epipolarConstrain(const KeyPointEx &kp1, const KeyPointEx &kp2, const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12) = 0;

    const unsigned int CAM_PINHOLE = 0;
    const unsigned int CAM_FISHEYE = 1;

    // ==================== GRID CONSTANTS ====================
    static const int FRAME_GRID_ROWS = 48;   ///< Number of grid rows for feature matching acceleration
    static const int FRAME_GRID_COLS = 64;   ///< Number of grid columns for feature matching acceleration

    // ==================== IMAGE BOUNDS AND GRID PARAMETERS ====================
    
    /**
     * @brief Check if point is within image bounds
     */
    bool IsInImage(const float &x, const float &y) const;

    // ==================== STATIC IMAGE PARAMETERS ====================
    int mnGridCols;                    ///< Number of grid columns for feature matching acceleration
    int mnGridRows;                    ///< Number of grid rows for feature matching acceleration
    float mfGridElementWidthInv;       ///< Inverse width of each grid element
    float mfGridElementHeightInv;      ///< Inverse height of each grid element
    
    int mnMinX;                        ///< Minimum X coordinate of image bounds
    int mnMinY;                        ///< Minimum Y coordinate of image bounds
    int mnMaxX;                        ///< Maximum X coordinate of image bounds
    int mnMaxY;                        ///< Maximum Y coordinate of image bounds

protected:
    /**
     * @brief Initialize image bounds and grid parameters for feature matching
     */
    void InitializeImageBounds();

public:
    std::vector<float> mvParameters;
    int mnWdith, mnHeight;
    float mfFps;
    unsigned int mnId;
    unsigned int mnType;
};