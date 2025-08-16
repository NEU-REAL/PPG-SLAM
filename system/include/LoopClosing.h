#pragma once
#include <boost/algorithm/string.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include "g2o/types/sim3/types_seven_dof_expmap.h"

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Viewer.h"
#include "Optimizer.h"

class LocalMapping;
class Map;
class Viewer;

class MSLoopClosing
{
    public:
    static MSLoopClosing& get()
    {
        static MSLoopClosing single_instance;
        return single_instance;
    }
private:
    MSLoopClosing() = default;
    ~MSLoopClosing()
    {
        mbFinishRequested = true;
    }
    MSLoopClosing(const MSLoopClosing& other) = delete;
    MSLoopClosing& operator=(const MSLoopClosing& other) = delete;
    static MSLoopClosing* singleViewing;
    
public:
    void Launch(Map* pMap, const bool bActiveLC);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF);

    bool isRunningGBA()
    {
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }

    void RequestFinish();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    bool CheckNewKeyFrames();
    //Methods to implement the new place recognition algorithm
    bool NewDetectCommonRegions();
    bool DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                        std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs);
    bool DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand, KeyFrame* &pMatchedKF, KeyFrame* &pLastCurrentKF, g2o::Sim3 &g2oScw,
                                     int &nNumCoincidences, std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs);
    bool DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                            std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs);
    int FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw, g2o::Sim3 &g2oScw,
                                set<MapPoint*> &spMatchedMPinOrigin, vector<MapPoint*> &vpMapPoints,
                                vector<MapPoint*> &vpMatchedMapPoints);

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<MapPoint*> &vpMapPoints);

    void CorrectLoop();

    void ResetIfRequested();
    bool mbResetRequested;
    Map* mpMapToReset;
    std::mutex mMutexReset;

    std::atomic<bool> mbFinishRequested;

    Map* mpMap;

    std::list<KeyFrame*> mlpLoopKeyFrameQueue;

    std::mutex mMutexLoopQueue;

    // Loop detector variables
    KeyFrame* mpCurrentKF;
    KeyFrame* mpLastCurrentKF;
    std::vector<KeyFrame*> mvpCurrentConnectedKFs;
    std::vector<MapPoint*> mvpLoopMapPoints;
    cv::Mat mScw;
    g2o::Sim3 mg2oScw;

    bool mbLoopDetected;
    int mnLoopNumCoincidences;
    int mnLoopNumNotFound;
    KeyFrame* mpLoopLastCurrentKF;
    g2o::Sim3 mg2oLoopSlw;
    g2o::Sim3 mg2oLoopScw;
    KeyFrame* mpLoopMatchedKF;
    std::vector<MapPoint*> mvpLoopMPs;
    std::vector<MapPoint*> mvpLoopMatchedMPs;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo case
    int mnFullBAIdx;

    // To (de)activate LC
    bool mbActiveLC = true;

    std::thread* mptLoopClosing;
};