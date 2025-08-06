#pragma once

#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "Map.h"

#include <mutex>
#include <atomic>

class System;
class Tracking;
class LoopClosing;

class MSLocalMapping
{
public:
    static MSLocalMapping& get()
    {
        static MSLocalMapping single_instance;
        return single_instance;
    }
private:
    MSLocalMapping() = default;
    ~MSLocalMapping()
    {
        mbFinishRequested = true;
    }
    MSLocalMapping(const MSLocalMapping& other) = delete;
    MSLocalMapping& operator=(const MSLocalMapping& other) = delete;
    static MSLocalMapping* singleViewing;
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void Launch(Map *pMap);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);
    void EmptyQueue();

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();

    int KeyframesInQueue()
    {
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

    double GetCurrKFTime();
    KeyFrame *GetCurrKF();

    Eigen::Matrix3d mRwg;
    Eigen::Vector3d mbg;
    Eigen::Vector3d mba;
    double mScale;
    Eigen::MatrixXd infoInertial;
    double mFirstTs;

    bool bInitializing;

    bool mbLocalMappingIdle;

public:
    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void UpdateNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();
    void KeyFrameCulling();

    System *mpSystem;

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    std::atomic<bool> mbFinishRequested;

    Map* mpMap;

    std::list<KeyFrame *> mlNewKeyFrames;

    KeyFrame *mpCurrentKeyFrame;

    std::list<MapEdge *> mlpRecentAddedMapEdges;
    std::list<MapColine *> mlpRecentAddedMapColines;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;


    void InitializeIMU(float priorG = 1e2, float priorA = 1e6, bool bFirst = false);
    void ScaleRefinement();

    std::thread* mptLocalMapping;

    float mTinit;
};