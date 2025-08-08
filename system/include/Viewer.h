#pragma once

#include "MapPoint.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <pangolin/pangolin.h>
#include <mutex>
#include <unordered_set>
#include "Tracking.h"
#include "System.h"
#include "Frame.h"

#include <mutex>

class Tracking;
class System;

class MSViewing
{
public:
    static MSViewing& get()
    {
        static MSViewing single_instance;
        return single_instance;
    }
private:
    MSViewing() = default;
    ~MSViewing()
    {
        mbFinishRequested = true;
    }
    MSViewing(const MSViewing& other) = delete;
    MSViewing& operator=(const MSViewing& other) = delete;
    static MSViewing* singleViewing;
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void Launch(Map* pMap);
    void Run();
    void UpdateFrame(Frame &F);

    cv::Mat DrawFrame();
    // draw map
    void DrawMapPoints();
    void DrawMapColines();
    void DrawMapEdges();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    void SetCurrentCameraPose(const Sophus::SE3f &Tcw);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw);

    void SaveTrajectory(const string &filename);
    void SaveKeyFrameTrajectory(const string &filename);

    void RequestFinish();
    std::atomic<bool> mbFinishRequested;

    Map* mpMap;
    std::thread* mptViewer;

public:
    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    // Info of the frame to be drawn
    cv::Mat mIm;
    // initialize
    vector<KeyPointEx> mvIniKeys;
    vector<int> mvIniMatches;
    // frame
    vector<KeyPointEx> mvCurrentKeys;
    vector<KeyEdge> mvCurrentEdges;
    // map points
    vector<bool> mvbOutliers;
    vector<MapPoint*> mvpMapPoints;
    vector<MapEdge*> mvpMapEdges;
    int mnTracked;

    int mState;

    unsigned int long mnCurFrameID;

    Sophus::SE3<float> mTcw;
    GeometricCamera* mpCamera;

    vector<MapPoint*> mvpLocalMap;

    std::atomic<bool> mbShowPoint;
    std::atomic<bool> mbShowColine;
    std::atomic<bool> mbShowEdge;
    std::atomic<bool> mbunFaded;
    std::atomic<bool> mbFinish;
    std::atomic<bool> mbStepByStep;
    std::atomic<bool> mbStep;

    Sophus::SE3f mCameraPose;

    std::deque<Sophus::SE3f> mCameraPoses;
    std::mutex mMutex;
};