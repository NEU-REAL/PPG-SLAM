#pragma once

#include <set>
#include <mutex>
#include "IMU.h"
#include "PPGGraph.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "GeometricCamera.h"
#include "Frame.h"

class GeometricCamera;
class KeyPointEx;
class MapEdge;
class MapColine;
class MapPoint;
class KeyFrame;
class Frame;

using namespace std;

class Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Map(GeometricCamera* pCam, IMU::Calib *pImu, DBoW3::Vocabulary *pVoc);
    ~Map();

    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void AddMapEdge(MapEdge *pME);
    void AddMapColine(MapColine* pMC);

    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void EraseMapEdge(MapEdge *pME);
    void EraseMapColine(MapColine* pMC);

    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapEdge*> GetAllMapEdges();
    std::vector<MapColine*> GetAllMapColines();

    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();
    long unsigned int GetMaxKFid();

    KeyFrame* GetOriginKF();

    void clear();

    bool CheckMapChanged();
    void InfoMapChange();

    void SetImuInitialized();
    bool isImuInitialized();

    void ApplyScaledRotation(const Sophus::SE3f &T, const float s, const bool bScaledVel=false);

    void SetIniertialBA1();
    void SetIniertialBA2();
    bool GetIniertialBA1();
    bool GetIniertialBA2();

    vector<KeyFrame*> DetectNBestCandidates(KeyFrame *pKF, unsigned int nNumCandidates);
    vector<KeyFrame*> DetectRelocalizationCandidates(Frame *F);

    void IncreseMap(KeyFrame* pNewKF);
    
    std::mutex mMutexMapUpdate;

    GeometricCamera* mpCamera;
    IMU::Calib *mpImuCalib;

protected:
    long unsigned int mnMaxKFid;
    bool mbImuInitialized;

    int mnMapChange;
    int mnLastMapChange;

    bool mbIMU_BA1;
    bool mbIMU_BA2;

    std::set<MapPoint*> mspMapPoints;
    std::set<MapEdge*> mspMapEdges;
    std::set<MapColine*> mspMapColines;
    std::set<KeyFrame*> mspKeyFrames;

    KeyFrame* mpKFinitial;

    // Mutex
    std::mutex mMutexMap;

    std::list<MapPoint *> mlpRecentAddedMapPoints;

// Database
public:
    // Associated vocabulary
    DBoW3::Vocabulary* mpVoc;
    static double imuIniTm;
private:
    // Inverted file
    std::vector<list<KeyFrame*> > mvInvertedFile;
    // Mutex
    std::mutex mMutexDatabase;
};