/**
 * @file Map.h
 * @brief SLAM Map class with keyframes, map points, and place recognition
 */

#pragma once

// ==================== SYSTEM INCLUDES ====================
#include <set>
#include <mutex>

// ==================== LOCAL INCLUDES ====================
#include "IMU.h"
#include "PPGGraph.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "GeometricCamera.h"
#include "Frame.h"

// ==================== FORWARD DECLARATIONS ====================
class GeometricCamera;
class KeyPointEx;
class MapEdge;
class MapColine;
class MapPoint;
class KeyFrame;
class Frame;

using namespace std;

/**
 * @class Map
 * @brief Global map containing keyframes, map points and place recognition database
 */
class Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // ==================== CONSTRUCTORS/DESTRUCTORS ====================
    
    /**
     * @brief Constructor with camera, IMU and vocabulary
     * @param pCam Camera model
     * @param pImu IMU calibration
     * @param pVoc BoW vocabulary
     */
    Map(GeometricCamera* pCam, IMU::Calib *pImu, DBoW3::Vocabulary *pVoc);
    
    /**
     * @brief Destructor
     */
    ~Map();

    // ==================== ELEMENT MANAGEMENT ====================
    
    /// Element addition functions
    void AddKeyFrame(KeyFrame* pKF);                            ///< Add keyframe to map
    void AddMapPoint(MapPoint* pMP);                            ///< Add map point to map
    void AddMapEdge(MapEdge *pME);                              ///< Add map edge to map
    void AddMapColine(MapColine* pMC);                          ///< Add map coline to map

    /// Element removal functions
    void EraseMapPoint(MapPoint* pMP);                          ///< Remove map point from map
    void EraseKeyFrame(KeyFrame* pKF);                          ///< Remove keyframe from map
    void EraseMapEdge(MapEdge *pME);                            ///< Remove map edge from map
    void EraseMapColine(MapColine* pMC);                        ///< Remove map coline from map

    /// Element getter functions
    std::vector<KeyFrame*> GetAllKeyFrames();                  ///< Get all keyframes in map
    std::vector<MapPoint*> GetAllMapPoints();                  ///< Get all map points in map
    std::vector<MapEdge*> GetAllMapEdges();                    ///< Get all map edges in map
    std::vector<MapColine*> GetAllMapColines();                ///< Get all map colines in map

    // ==================== MAP STATISTICS ====================
    
    long unsigned int MapPointsInMap();                        ///< Get number of map points
    long unsigned int KeyFramesInMap();                        ///< Get number of keyframes
    long unsigned int GetMaxKFid();                            ///< Get maximum keyframe ID
    KeyFrame* GetOriginKF();                                   ///< Get origin keyframe

    // ==================== MAP STATE ====================
    
    /**
     * @brief Clear all map elements
     */
    void clear();
    
    /**
     * @brief Check if map has changed
     * @return True if map changed
     */
    bool CheckMapChanged();
    
    /**
     * @brief Update map change information
     */
    void InfoMapChange();

    // ==================== IMU INITIALIZATION ====================
    
    /**
     * @brief Set IMU as initialized
     */
    void SetImuInitialized();
    
    /**
     * @brief Check if IMU is initialized
     * @return True if IMU is initialized
     */
    bool isImuInitialized();
    
    /**
     * @brief Apply scaled rotation to map
     * @param T Transformation
     * @param s Scale factor
     * @param bScaledVel Scale velocities flag
     */
    void ApplyScaledRotation(const Sophus::SE3f &T, const float s, const bool bScaledVel=false);

    // ==================== BUNDLE ADJUSTMENT FLAGS ====================
    
    /**
     * @brief Set inertial bundle adjustment flag
     */
    void SetInertialBA();
    
    /**
     * @brief Get inertial bundle adjustment flag
     * @return Bundle adjustment status
     */
    bool GetInertialBA();

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