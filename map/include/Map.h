/**
 * @file Map.h
 * @brief SLAM Map class with keyframes, map points, and place recognition
 */

#pragma once

#include <set>
#include <mutex>
#include "IMU.h"
#include "PPGGraph.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "GeometricCamera.h"
#include "Frame.h"

// Forward declarations
class GeometricCamera;
class KeyPointEx;
class MapEdge;
class MapColine;
class MapPoint;
class KeyFrame;
class Frame;

using namespace std;

/// Global map containing keyframes, map points and database
class Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // ==================== CONSTRUCTION ====================
    
    Map(GeometricCamera* pCam, IMU::Calib *pImu, DBoW3::Vocabulary *pVoc);
    ~Map();

    // ==================== ELEMENT MANAGEMENT ====================
    
    /// Add elements
    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void AddMapEdge(MapEdge *pME);
    void AddMapColine(MapColine* pMC);

    /// Remove elements
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void EraseMapEdge(MapEdge *pME);
    void EraseMapColine(MapColine* pMC);

    /// Get all elements
    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapEdge*> GetAllMapEdges();
    std::vector<MapColine*> GetAllMapColines();

    // ==================== MAP STATISTICS ====================
    
    long unsigned int MapPointsInMap();
    long unsigned int KeyFramesInMap();
    long unsigned int GetMaxKFid();
    KeyFrame* GetOriginKF();

    // ==================== MAP STATE ====================
    
    void clear();
    bool CheckMapChanged();
    void InfoMapChange();

    // ==================== IMU INITIALIZATION ====================
    
    void SetImuInitialized();
    bool isImuInitialized();
    void ApplyScaledRotation(const Sophus::SE3f &T, const float s, const bool bScaledVel=false);

    // ==================== BUNDLE ADJUSTMENT FLAGS ====================
    // NOTE: Function names have spelling error but maintained for compatibility
    
    void SetInertialBA();    ///< Should be "SetInertialBA2" 
    bool GetInertialBA();    ///< Should be "GetInertialBA2"

    // ==================== PLACE RECOGNITION ====================
    
    vector<KeyFrame*> DetectNBestCandidates(KeyFrame *pKF, unsigned int nNumCandidates);
    vector<KeyFrame*> DetectRelocalizationCandidates(Frame *F);

    // ==================== LOCAL MAPPING ====================
    
    void IncreMap(KeyFrame* pNewKF);

    // ==================== PUBLIC MEMBERS ====================
    
    std::mutex mMutexMapUpdate;
    GeometricCamera* mpCamera;
    IMU::Calib *mpImuCalib;
    DBoW3::Vocabulary* mpVoc;
    static double imuIniTm;

private:
    // ==================== INTERNAL METHODS ====================
    
    /// Triangulate new map points from matches
    void TriangulateNewMapPoints(KeyFrame* pNewKF, const vector<KeyFrame*>& vpNeighKFs);
    
    /// Check triangulation validity
    bool IsValidTriangulation(const Eigen::Vector3f& x3D, const KeyPointEx& kp1, const KeyPointEx& kp2,
                             const Eigen::Matrix3f& Rcw1, const Eigen::Vector3f& tcw1,
                             const Eigen::Matrix3f& Rcw2, const Eigen::Vector3f& tcw2);
    
    /// Create map edges for new keyframe
    void CreateMapEdges(KeyFrame* pNewKF);
    
    /// Create map colines for new keyframe
    void CreateMapColines(KeyFrame* pNewKF);

    // ==================== MAP STATE ====================
    
    long unsigned int mnMaxKFid;
    bool mbImuInitialized;
    int mnMapChange;
    int mnLastMapChange;
    bool mbIMU_BA;

    // ==================== MAP ELEMENTS ====================
    
    std::set<MapPoint*> mspMapPoints;
    std::set<MapEdge*> mspMapEdges;
    std::set<MapColine*> mspMapColines;
    std::set<KeyFrame*> mspKeyFrames;
    KeyFrame* mpKFinitial;
    std::list<MapPoint*> mlpRecentAddedMapPoints;

    // ==================== DATABASE ====================
    
    std::vector<list<KeyFrame*>> mvInvertedFile;

    // ==================== THREAD SAFETY ====================
    
    std::mutex mMutexMap;
    std::mutex mMutexDatabase;
};