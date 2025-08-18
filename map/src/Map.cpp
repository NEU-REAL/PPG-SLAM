/**
 * @file Map.cpp
 * @brief SLAM Map implementation
 */

#include <mutex>
#include "Matcher.h"
#include "Map.h"

// ==================== STATIC MEMBERS ====================

double Map::imuIniTm = 5.0;

// ==================== CONSTRUCTION ====================

Map::Map(GeometricCamera* pCam, IMU::Calib *pImu, DBoW3::Vocabulary *pVoc) 
    : mpCamera(pCam), mpImuCalib(pImu), mpVoc(pVoc),
      mnMaxKFid(0), mbImuInitialized(false), 
      mnMapChange(0), mnLastMapChange(0), 
      mbIMU_BA(false)
{
    mvInvertedFile.resize(pVoc->size());
}

Map::~Map()
{
    mspMapPoints.clear();
    mspKeyFrames.clear();
    mlpRecentAddedMapPoints.clear();
}

// ==================== ELEMENT MANAGEMENT ====================

void Map::AddKeyFrame(KeyFrame *pKF)
{
    {
        unique_lock<mutex> lock(mMutexMap);
        if(mspKeyFrames.empty())
        {
            mpKFinitial = pKF;
            cout << "First KF:" << pKF->mnId << endl;
        }
        mspKeyFrames.insert(pKF);
        if(pKF->mnId > mnMaxKFid)
            mnMaxKFid = pKF->mnId;
    }

    {
        unique_lock<mutex> lock(mMutexDatabase);
        for(auto vit : pKF->mBowVec)
            mvInvertedFile[vit.first].push_back(pKF);
    }
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::AddMapEdge(MapEdge *pME)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapEdges.insert(pME);
}

void Map::AddMapColine(MapColine *pMC)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapColines.insert(pMC);
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);
    // NOTE: MapPoint deletion is handled by caller to avoid ownership conflicts
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);
    }

    {
        unique_lock<mutex> lock(mMutexDatabase);
        for(auto vit : pKF->mBowVec)
        {
            list<KeyFrame*> &lKFs = mvInvertedFile[vit.first];
            for(auto lit = lKFs.begin(); lit != lKFs.end(); lit++)
            {
                if(pKF == *lit)
                {
                    lKFs.erase(lit);
                    break;
                }
            }
        }
    }
}

void Map::EraseMapEdge(MapEdge *pME)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapEdges.erase(pME);
}

void Map::EraseMapColine(MapColine *pMC)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapColines.erase(pMC);
}

// ==================== ELEMENT RETRIEVAL ====================

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
}

vector<MapEdge*> Map::GetAllMapEdges()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapEdge*>(mspMapEdges.begin(), mspMapEdges.end());
}

vector<MapColine*> Map::GetAllMapColines()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapColine*>(mspMapColines.begin(), mspMapColines.end());
}

// ==================== MAP STATISTICS ====================

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

KeyFrame* Map::GetOriginKF()
{
    return mpKFinitial;
}

// ==================== MAP STATE ====================

void Map::clear()
{
    mspMapEdges.clear();
    mspMapPoints.clear();
    mspKeyFrames.clear();
    mbImuInitialized = false;
    mbIMU_BA = false;
    mlpRecentAddedMapPoints.clear();
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

bool Map::CheckMapChanged()
{
    unique_lock<mutex> lock(mMutexMap);
    if(mnMapChange > mnLastMapChange)
    {
        mnLastMapChange = mnMapChange;
        return true;
    }
    return false;
}

void Map::InfoMapChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnMapChange++;
}

// ==================== IMU INITIALIZATION ====================

void Map::SetImuInitialized()
{
    unique_lock<mutex> lock(mMutexMap);
    mbImuInitialized = true;
}

bool Map::isImuInitialized()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbImuInitialized;
}

void Map::ApplyScaledRotation(const SE3f &T, const float s, const bool bScaledVel)
{
    unique_lock<mutex> lock(mMutexMap);

    SE3f Tyw = T;
    Eigen::Matrix3f Ryw = Tyw.rotationMatrix();
    Eigen::Vector3f tyw = Tyw.translation();

    // Transform keyframes
    for(auto pKF : mspKeyFrames)
    {
        SE3f Twc = pKF->GetPoseInverse();
        Twc.translation() *= s;
        SE3f Tyc = Tyw * Twc;
        SE3f Tcy = Tyc.inverse();
        pKF->SetPose(Tcy);
        
        Eigen::Vector3f Vw = pKF->GetVelocity();
        if(!bScaledVel)
            pKF->SetVelocity(Ryw * Vw);
        else
            pKF->SetVelocity(Ryw * Vw * s);
    }
    
    // Transform map points
    for(auto pMP : mspMapPoints)
    {
        pMP->SetWorldPos(s * Ryw * pMP->GetWorldPos() + tyw);
        pMP->UpdateNormalAndDepth();
    }
    mnMapChange++;
}

// ==================== BUNDLE ADJUSTMENT FLAGS ====================

void Map::SetInertialBA()
{
    unique_lock<mutex> lock(mMutexMap);
    mbIMU_BA = true;
}

bool Map::GetInertialBA()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbIMU_BA;
}


// ==================== PLACE RECOGNITION ====================

vector<KeyFrame*> Map::DetectNBestCandidates(KeyFrame *pKF, unsigned int nNumCandidates)
{
    list<KeyFrame*> lKFsSharingWords;
    set<KeyFrame*> spConnectedKF;

    // Find keyframes sharing words with current frame
    {
        unique_lock<mutex> lock(mMutexDatabase);
        spConnectedKF = pKF->GetConnectedKeyFrames();

        for(auto vit = pKF->mBowVec.begin(); vit != pKF->mBowVec.end(); vit++)
        {
            list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];
            for(auto pKFi : lKFs)
            {
                if(pKFi->mnPlaceRecognitionQuery != pKF->mnId)
                {
                    pKFi->mnPlaceRecognitionWords = 0;
                    if(!spConnectedKF.count(pKFi))
                    {
                        pKFi->mnPlaceRecognitionQuery = pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnPlaceRecognitionWords++;
            }
        }
    }
    
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Find candidates with enough shared words
    int maxCommonWords = 0;
    for(auto pKFi : lKFsSharingWords)
        if(pKFi->mnPlaceRecognitionWords > maxCommonWords)
            maxCommonWords = pKFi->mnPlaceRecognitionWords;

    int minCommonWords = maxCommonWords * 0.8f;
    list<pair<float,KeyFrame*>> lScoreAndMatch;

    // Compute similarity scores
    for(auto pKFi : lKFsSharingWords)
    {
        if(pKFi->mnPlaceRecognitionWords > minCommonWords)
        {
            float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);
            pKFi->mPlaceRecognitionScore = si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    // Accumulate scores by covisibility
    list<pair<float,KeyFrame*>> lAccScoreAndMatch;
    for(auto it : lScoreAndMatch)
    {
        KeyFrame* pKFi = it.second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it.first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        
        for(auto pKF2 : vpNeighs)
        {
            if(pKF2->mnPlaceRecognitionQuery != pKF->mnId)
                continue;

            accScore += pKF2->mPlaceRecognitionScore;
            if(pKF2->mPlaceRecognitionScore > bestScore)
            {
                pBestKF = pKF2;
                bestScore = pKF2->mPlaceRecognitionScore;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
    }

    // Sort by accumulated score
    lAccScoreAndMatch.sort([](const pair<float, KeyFrame*> &a, const pair<float, KeyFrame*> &b)
                          { return a.first > b.first; });

    // Return top candidates
    vector<KeyFrame*> vpLoopCand;
    vpLoopCand.reserve(nNumCandidates);
    set<KeyFrame*> spAlreadyAddedKF;

    for(auto it : lAccScoreAndMatch)
    {
        KeyFrame* pKFi = it.second;
        if(vpLoopCand.size() >= nNumCandidates)
            break;
        if(!pKFi->isBad() && !spAlreadyAddedKF.count(pKFi))
        {
            vpLoopCand.push_back(pKFi);
            spAlreadyAddedKF.insert(pKFi);
        }
    }
    return vpLoopCand;
}


vector<KeyFrame*> Map::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords;

    // Find keyframes sharing words with current frame
    {
        unique_lock<mutex> lock(mMutexDatabase);
        for(auto vit = F->mBowVec.begin(); vit != F->mBowVec.end(); vit++)
        {
            list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];
            for(auto pKFi : lKFs)
            {
                if(pKFi->mnRelocQuery != F->mnId)
                {
                    pKFi->mnRelocWords = 0;
                    pKFi->mnRelocQuery = F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Find candidates with enough shared words
    int maxCommonWords = 0;
    for(auto pKFi : lKFsSharingWords)
        if(pKFi->mnRelocWords > maxCommonWords)
            maxCommonWords = pKFi->mnRelocWords;

    int minCommonWords = maxCommonWords * 0.8f;
    list<pair<float,KeyFrame*>> lScoreAndMatch;

    // Compute similarity scores
    for(auto pKFi : lKFsSharingWords)
    {
        if(pKFi->mnRelocWords > minCommonWords)
        {
            float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
            pKFi->mRelocScore = si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    // Accumulate scores by covisibility
    list<pair<float,KeyFrame*>> lAccScoreAndMatch;
    float bestAccScore = 0;

    for(auto it : lScoreAndMatch)
    {
        KeyFrame* pKFi = it.second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it.first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        
        for(auto pKF2 : vpNeighs)
        {
            if(pKF2->mnRelocQuery != F->mnId)
                continue;

            accScore += pKF2->mRelocScore;
            if(pKF2->mRelocScore > bestScore)
            {
                pBestKF = pKF2;
                bestScore = pKF2->mRelocScore;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if(accScore > bestAccScore)
            bestAccScore = accScore;
    }

    // Return candidates with high scores
    float minScoreToRetain = 0.75f * bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    
    for(auto it : lAccScoreAndMatch)
    {
        const float &si = it.first;
        if(si > minScoreToRetain)
        {
            KeyFrame* pKFi = it.second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }
    return vpRelocCandidates;
}

// ==================== LOCAL MAPPING ====================

void Map::IncreMap(KeyFrame* pNewKF)
{
    // Update map point observations
    const vector<MapPoint*> vpMapPointMatches = pNewKF->GetMapPointMatches();
    for(size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP && !pMP->isBad())
        {
            pMP->AddObservation(pNewKF, i);
            pMP->UpdateNormalAndDepth();
            pMP->ComputeDistinctiveDescriptors();
        }
    }

    // Clean up recent map points
    auto lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = pNewKF->mnId;

    while(lit != mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio() < 0.25f)
        {
            pMP->SetBadFlag();
            EraseMapPoint(pMP);
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= 2)
        {
            pMP->SetBadFlag();
            EraseMapPoint(pMP);
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else
        {
            lit++;
        }
    }

    // Get neighbor keyframes for triangulation
    vector<KeyFrame*> vpNeighKFs;
    KeyFrame* pKF = pNewKF;
    unsigned int count = 0;
    const unsigned int nn = 5;
    
    while((vpNeighKFs.size() <= nn) && (pKF->mPrevKF) && (count++ < nn))
    {
        auto it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
        if(it == vpNeighKFs.end())
            vpNeighKFs.push_back(pKF->mPrevKF);
        pKF = pKF->mPrevKF;
    }

    // Triangulate new map points
    TriangulateNewMapPoints(pNewKF, vpNeighKFs);
    
    // Create map edges
    CreateMapEdges(pNewKF);
    
    // Create map colines
    CreateMapColines(pNewKF);
    
    // Update keyframe connections and add to map
    pNewKF->UpdateConnections();
    AddKeyFrame(pNewKF);
}

void Map::TriangulateNewMapPoints(KeyFrame* pNewKF, const vector<KeyFrame*>& vpNeighKFs)
{
    const float th = 0.6f;
    Matcher matcher(mpCamera, th);

    SE3<float> sophTcw1 = pNewKF->GetPose();
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4();
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0);
    Eigen::Vector3f tcw1 = sophTcw1.translation();

    // Search matches and triangulate with each neighbor
    for(size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        KeyFrame* pKF2 = vpNeighKFs[i];
        vector<pair<size_t,size_t>> vMatchedIndices;
        matcher.SearchForTriangulation(pNewKF, pKF2, vMatchedIndices, true);

        SE3<float> sophTcw2 = pKF2->GetPose();
        Eigen::Matrix<float,3,4> eigTcw2 = sophTcw2.matrix3x4();
        Eigen::Matrix<float,3,3> Rcw2 = eigTcw2.block<3,3>(0,0);
        Eigen::Vector3f tcw2 = sophTcw2.translation();

        // Triangulate each match
        for(auto match : vMatchedIndices)
        {
            const int idx1 = match.first;
            const int idx2 = match.second;
            
            const KeyPointEx &kp1 = pNewKF->mvKeysUn[idx1];
            const KeyPointEx &kp2 = pKF2->mvKeysUn[idx2];

            // Triangulate point
            Eigen::Vector3f xn1 = mpCamera->unproject(kp1.mPos);
            Eigen::Vector3f xn2 = mpCamera->unproject(kp2.mPos);

            Eigen::Matrix4f A;
            A.block<1,4>(0,0) = xn1(0) * eigTcw1.block<1,4>(2,0) - eigTcw1.block<1,4>(0,0);
            A.block<1,4>(1,0) = xn1(1) * eigTcw1.block<1,4>(2,0) - eigTcw1.block<1,4>(1,0);
            A.block<1,4>(2,0) = xn2(0) * eigTcw2.block<1,4>(2,0) - eigTcw2.block<1,4>(0,0);
            A.block<1,4>(3,0) = xn2(1) * eigTcw2.block<1,4>(2,0) - eigTcw2.block<1,4>(1,0);
            
            Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
            Eigen::Vector4f x3Dh = svd.matrixV().col(3);
            
            if(x3Dh(3) == 0)
                continue;
                
            Eigen::Vector3f x3D = x3Dh.head(3) / x3Dh(3);

            // Check triangulation validity
            if(!IsValidTriangulation(x3D, kp1, kp2, Rcw1, tcw1, Rcw2, tcw2))
                continue;

            // Create new map point
            MapPoint* pMP = new MapPoint(x3D, pNewKF);
            pMP->AddObservation(pNewKF, idx1);
            pMP->AddObservation(pKF2, idx2);
            pNewKF->AddMapPoint(pMP, idx1);
            pKF2->AddMapPoint(pMP, idx2);
            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();
            
            AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }
}

bool Map::IsValidTriangulation(const Eigen::Vector3f& x3D, const KeyPointEx& kp1, const KeyPointEx& kp2,
                              const Eigen::Matrix3f& Rcw1, const Eigen::Vector3f& tcw1,
                              const Eigen::Matrix3f& Rcw2, const Eigen::Vector3f& tcw2)
{
    // Check if point is in front of both cameras
    float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
    float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
    if(z1 <= 0 || z2 <= 0)
        return false;

    // Check reprojection error in first keyframe
    const float x1 = Rcw1.row(0).dot(x3D) + tcw1(0);
    const float y1 = Rcw1.row(1).dot(x3D) + tcw1(1);
    Eigen::Vector2f uv1 = mpCamera->project(Eigen::Vector3f(x1, y1, z1));
    float errX1 = uv1[0] - kp1.mPos[0];
    float errY1 = uv1[1] - kp1.mPos[1];
    if((errX1*errX1 + errY1*errY1) > 5.991)
        return false;

    // Check reprojection error in second keyframe
    const float x2 = Rcw2.row(0).dot(x3D) + tcw2(0);
    const float y2 = Rcw2.row(1).dot(x3D) + tcw2(1);
    Eigen::Vector2f uv2 = mpCamera->project(Eigen::Vector3f(x2, y2, z2));
    float errX2 = uv2[0] - kp2.mPos[0];
    float errY2 = uv2[1] - kp2.mPos[1];
    if((errX2*errX2 + errY2*errY2) > 5.991)
        return false;

    return true;
}

void Map::CreateMapEdges(KeyFrame* pNewKF)
{
    for(unsigned int lid_cur = 0; lid_cur < pNewKF->mvKeyEdges.size(); lid_cur++)
    {
        MapEdge* pME = pNewKF->GetMapEdge(lid_cur);
        if(pME && !pME->isBad())
            continue;
            
        KeyEdge ke_cur = pNewKF->mvKeyEdges[lid_cur];
        MapPoint *pMP1 = pNewKF->GetMapPoint(ke_cur.startIdx);
        MapPoint *pMP2 = pNewKF->GetMapPoint(ke_cur.endIdx);
        
        if(!pMP1 || !pMP2 || pMP1->isBad() || pMP2->isBad())
            continue;

        // Check viewing angle
        Eigen::Vector3f v_ = (pMP1->GetWorldPos() - pMP2->GetWorldPos()).normalized();
        Eigen::Vector3f v1_ = (pNewKF->GetCameraCenter() - pMP1->GetWorldPos()).normalized();
        Eigen::Vector3f v2_ = (pNewKF->GetCameraCenter() - pMP2->GetWorldPos()).normalized();
        
        if(std::fabs(v_.dot(v1_)) > MapEdge::viewCosTh || std::fabs(v_.dot(v2_)) > MapEdge::viewCosTh)
            continue;

        // Try to find existing edge
        pME = pMP1->getEdge(pMP2);
        if(pME && !pME->isBad())
        {
            pNewKF->AddMapEdge(pME, lid_cur);
            pME->addObservation(pNewKF, lid_cur);
            pME->checkValid();
        }
        else
        {
            // Create new edge
            pME = new MapEdge(pMP1, pMP2);
            pNewKF->AddMapEdge(pME, lid_cur);
            pME->addObservation(pNewKF, lid_cur);
            AddMapEdge(pME);
        }
    }
}

void Map::CreateMapColines(KeyFrame* pNewKF)
{
    for(unsigned int pid_cur = 0; pid_cur < pNewKF->mvKeysUn.size(); pid_cur++)
    {
        MapPoint* pMP = pNewKF->GetMapPoint(pid_cur);
        if(!pMP || pMP->isBad())
            continue;
            
        const KeyPointEx &kp_cur = pNewKF->mvKeysUn[pid_cur];
        for(auto cp_cur : kp_cur.mvColine)
        {
            MapPoint* pMP1 = pNewKF->GetMapPoint(cp_cur.first);
            MapPoint* pMP2 = pNewKF->GetMapPoint(cp_cur.second);
            
            if(!pMP1 || !pMP2 || pMP1->isBad() || pMP2->isBad())
                continue;
                
            MapColine* pMC = pMP->addColine(pMP1, pMP2, pNewKF);
            if(pMC)
                AddMapColine(pMC);
        }
    }
}
