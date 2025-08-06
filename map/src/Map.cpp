#include<mutex>
#include "Matcher.h"
#include "Map.h"


float Map::imuIniTm = 5;

Map::Map(GeometricCamera* pCam, IMU::Calib *pImu, DBoW3::Vocabulary *pVoc) : mpCamera(pCam), 
    mpImuCalib(pImu), mnMaxKFid(0), mbImuInitialized(false), 
    mnMapChange(0), mnLastMapChange(0), mbIMU_BA1(false), mbIMU_BA2(false), mpVoc(pVoc)
{
    mvInvertedFile.resize(pVoc->size());
}

Map::~Map()
{
    mspMapPoints.clear();
    mspKeyFrames.clear();
    mlpRecentAddedMapPoints.clear();

}

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
        if(pKF->mnId>mnMaxKFid)
        {
            mnMaxKFid=pKF->mnId;
        }
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

void Map::EraseMapEdge(MapEdge *pME)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapEdges.erase(pME);
}

void Map::AddMapColine(MapColine *pMC)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapColines.insert(pMC);
}

void Map::EraseMapColine(MapColine *pMC)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapColines.erase(pMC);
}


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

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);
    }

    {
        unique_lock<mutex> lock(mMutexDatabase);
        // Erase elements in the Inverse File for the entry
        for(auto vit : pKF->mBowVec)
        {
            // List of keyframes that share the word
            list<KeyFrame*> &lKFs = mvInvertedFile[vit.first];
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                if(pKF==*lit)
                {
                    lKFs.erase(lit);
                    break;
                }
            }
        }
    }
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

vector<MapEdge*> Map::GetAllMapEdges()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapEdge*>(mspMapEdges.begin(),mspMapEdges.end());
}

vector<MapColine*> Map::GetAllMapColines()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapColine*>(mspMapColines.begin(),mspMapColines.end());
}


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

void Map::clear()
{
    mspMapEdges.clear();
    mspMapPoints.clear();
    mspKeyFrames.clear();
    mbImuInitialized = false;
    mbIMU_BA1 = false;
    mbIMU_BA2 = false;
    mlpRecentAddedMapPoints.clear();
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

void Map::ApplyScaledRotation(const Sophus::SE3f &T, const float s, const bool bScaledVel)
{
    unique_lock<mutex> lock(mMutexMap);

    // Body position (IMU) of first keyframe is fixed to (0,0,0)
    Sophus::SE3f Tyw = T;
    Eigen::Matrix3f Ryw = Tyw.rotationMatrix();
    Eigen::Vector3f tyw = Tyw.translation();

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(); sit!=mspKeyFrames.end(); sit++)
    {
        KeyFrame* pKF = *sit;
        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Twc.translation() *= s;
        Sophus::SE3f Tyc = Tyw*Twc;
        Sophus::SE3f Tcy = Tyc.inverse();
        pKF->SetPose(Tcy);
        Eigen::Vector3f Vw = pKF->GetVelocity();
        if(!bScaledVel)
            pKF->SetVelocity(Ryw*Vw);
        else
            pKF->SetVelocity(Ryw*Vw*s);

    }
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(); sit!=mspMapPoints.end(); sit++)
    {
        MapPoint* pMP = *sit;
        pMP->SetWorldPos(s * Ryw * pMP->GetWorldPos() + tyw);
        pMP->UpdateNormalAndDepth();
    }
    mnMapChange++;
}

void Map::SetIniertialBA1()
{
    unique_lock<mutex> lock(mMutexMap);
    mbIMU_BA1 = true;
}

void Map::SetIniertialBA2()
{
    unique_lock<mutex> lock(mMutexMap);
    mbIMU_BA2 = true;
}

bool Map::GetIniertialBA1()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbIMU_BA1;
}

bool Map::GetIniertialBA2()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbIMU_BA2;
}

void Map::InfoMapChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnMapChange++;
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


vector<KeyFrame*> Map::DetectNBestCandidates(KeyFrame *pKF, unsigned int nNumCandidates)
{
    list<KeyFrame*> lKFsSharingWords;
    set<KeyFrame*> spConnectedKF;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutexDatabase);

        spConnectedKF = pKF->GetConnectedKeyFrames();

        for(DBoW3::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;

                if(pKFi->mnPlaceRecognitionQuery!=pKF->mnId)
                {
                    pKFi->mnPlaceRecognitionWords=0;
                    if(!spConnectedKF.count(pKFi))
                    {
                        pKFi->mnPlaceRecognitionQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnPlaceRecognitionWords++;
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnPlaceRecognitionWords>maxCommonWords)
            maxCommonWords=(*lit)->mnPlaceRecognitionWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        if(pKFi->mnPlaceRecognitionWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);
            pKFi->mPlaceRecognitionScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnPlaceRecognitionQuery!=pKF->mnId)
                continue;

            accScore+=pKF2->mPlaceRecognitionScore;
            if(pKF2->mPlaceRecognitionScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mPlaceRecognitionScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    lAccScoreAndMatch.sort( [](const pair<float, KeyFrame*> & a, const pair<float, KeyFrame*> & b)
                            {return a.first > b.first;});

    vector<KeyFrame*> vpLoopCand;
    vpLoopCand.reserve(nNumCandidates);
    set<KeyFrame*> spAlreadyAddedKF;

    for(auto it : lAccScoreAndMatch)
    {
        KeyFrame* pKFi = it.second;
        if(vpLoopCand.size() >= nNumCandidates)
            break;
        if(!pKFi->isBad() && !spAlreadyAddedKF.count(pKFi))
            vpLoopCand.push_back(pKFi);
    }
    return vpLoopCand;
}


vector<KeyFrame*> Map::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutexDatabase);

        for(DBoW3::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery!=F->mnId)
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnRelocWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

            accScore+=pKF2->mRelocScore;
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }
    return vpRelocCandidates;
}

void Map::IncreseMap(KeyFrame* pNewKF)
{
    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = pNewKF->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP && !pMP->isBad())
        {
            pMP->AddObservation(pNewKF, i);
            pMP->UpdateNormalAndDepth();
            pMP->ComputeDistinctiveDescriptors();
        }
    }
    // update Edge observation
    for(unsigned int lid_cur=0; lid_cur<pNewKF->mvKeyEdges.size(); lid_cur++)
    {
        MapPoint* pMP1 = pNewKF->GetMapPoint(pNewKF->mvKeyEdges[lid_cur].startIdx);
        MapPoint* pMP2 = pNewKF->GetMapPoint(pNewKF->mvKeyEdges[lid_cur].endIdx);
        if(pMP1 == nullptr || pMP2 == nullptr || pMP1->isBad() || pMP2->isBad())
            continue;
        Eigen::Vector3f v_ = (pMP1->GetWorldPos() - pMP2->GetWorldPos()).normalized();
        Eigen::Vector3f v1_ = (pNewKF->GetCameraCenter() - pMP1->GetWorldPos()).normalized();
        Eigen::Vector3f v2_ = (pNewKF->GetCameraCenter() - pMP1->GetWorldPos()).normalized();
        if(std::fabs(v_.dot(v1_))>MapEdge::viewCosTh || std::fabs(v_.dot(v2_))>MapEdge::viewCosTh)
            continue;
        MapEdge *pME = pMP1->getEdge(pMP2);
        if(pME && !pME->isBad())
        {
            pNewKF->AddMapEdge(pME, lid_cur);
            pME->addObservation(pNewKF, lid_cur);
            pME->checkValid();
        }
    }
    // update coline observation
    for(unsigned int pid_cur=0; pid_cur<pNewKF->mvKeysUn.size(); pid_cur++)
    {
        MapPoint* pMP = pNewKF->GetMapPoint(pid_cur);
        if(pMP == nullptr || pMP->isBad())
            continue;   
        const KeyPointEx &kp_cur = pNewKF->mvKeysUn[pid_cur];
        for(auto cp_cur : kp_cur.mvColine)
        {
            MapPoint* pMPs = pNewKF->GetMapPoint(cp_cur.first);
            MapPoint* pMPe = pNewKF->GetMapPoint(cp_cur.second);
            if(pMPs == nullptr || pMPe == nullptr || pMPs->isBad() || pMPe->isBad())
                continue;
            MapColine* pMC = pMP->addColine(pMPs, pMPe, pNewKF);
            if(pMC)
                AddMapColine(pMC);
        }
    }

    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = pNewKF->mnId;

    int borrar = mlpRecentAddedMapPoints.size();

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
            lit = mlpRecentAddedMapPoints.erase(lit);
        else if(pMP->GetFoundRatio()<0.25f)
        {
            pMP->SetBadFlag();
            EraseMapPoint(pMP);
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=2)
        {
            pMP->SetBadFlag();
            EraseMapPoint(pMP);
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
        {
            lit++;
            borrar--;
        }
    }
    // Retrieve neighbor keyframes in covisibility graph
    unsigned int nn = 5;
    vector<KeyFrame*> vpNeighKFs;
    KeyFrame* pKF = pNewKF;
    unsigned int count=0; 
    while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn))
    {
        vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
        if(it==vpNeighKFs.end())
            vpNeighKFs.push_back(pKF->mPrevKF);
        pKF = pKF->mPrevKF;
    }
    float th = 0.6f;
    Matcher matcher(mpCamera, th);

    Sophus::SE3<float> sophTcw1 = pNewKF->GetPose();
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4();
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0);
    Eigen::Matrix<float,3,3> Rwc1 = Rcw1.transpose();
    Eigen::Vector3f tcw1 = sophTcw1.translation();
    Eigen::Vector3f Ow1 = pNewKF->GetCameraCenter();

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        KeyFrame* pKF2 = vpNeighKFs[i];

        GeometricCamera* pCamera = mpCamera;
        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(pNewKF, pKF2, vMatchedIndices, true);

        Sophus::SE3<float> sophTcw2 = pKF2->GetPose();
        Eigen::Matrix<float,3,4> eigTcw2 = sophTcw2.matrix3x4();
        Eigen::Matrix<float,3,3> Rcw2 = eigTcw2.block<3,3>(0,0);
        Eigen::Matrix<float,3,3> Rwc2 = Rcw2.transpose();
        Eigen::Vector3f tcw2 = sophTcw2.translation();

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const KeyPointEx &kp1 = pNewKF->mvKeysUn[idx1];
            const KeyPointEx &kp2 =pKF2->mvKeysUn[idx2];

            // Check parallax between rays
            Eigen::Vector3f xn1 = pCamera->unproject(kp1.mPos);
            Eigen::Vector3f xn2 = pCamera->unproject(kp2.mPos);

            Eigen::Matrix4f A;
            A.block<1,4>(0,0) = xn1(0) * eigTcw1.block<1,4>(2,0) - eigTcw1.block<1,4>(0,0);
            A.block<1,4>(1,0) = xn1(1) * eigTcw1.block<1,4>(2,0) - eigTcw1.block<1,4>(1,0);
            A.block<1,4>(2,0) = xn2(0) * eigTcw2.block<1,4>(2,0) - eigTcw2.block<1,4>(0,0);
            A.block<1,4>(3,0) = xn2(1) * eigTcw2.block<1,4>(2,0) - eigTcw2.block<1,4>(1,0);
            Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
            Eigen::Vector4f x3Dh = svd.matrixV().col(3);
            if(x3Dh(3)==0)
                continue;
            // Euclidean coordinates
            Eigen::Vector3f x3D = x3Dh.head(3)/x3Dh(3);

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float x1 = Rcw1.row(0).dot(x3D)+tcw1(0);
            const float y1 = Rcw1.row(1).dot(x3D)+tcw1(1);

            Eigen::Vector2f uv1 = pCamera->project(Eigen::Vector3f(x1,y1,z1));
            float errX1 = uv1[0] - kp1.mPos[0];
            float errY1 = uv1[1] - kp1.mPos[1];
            if((errX1*errX1+errY1*errY1)>5.991)
                continue;

            //Check reprojection error in second keyframe
            const float x2 = Rcw2.row(0).dot(x3D)+tcw2(0);
            const float y2 = Rcw2.row(1).dot(x3D)+tcw2(1);

            Eigen::Vector2f uv2 = pCamera->project(Eigen::Vector3f(x2,y2,z2));
            float errX2 = uv2[0] - kp2.mPos[0];
            float errY2 = uv2[1] - kp2.mPos[1];
            if((errX2*errX2+errY2*errY2)>5.991)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D, pNewKF);

            pMP->AddObservation(pNewKF,idx1);
            pMP->AddObservation(pKF2,idx2);

            pNewKF->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
        for( unsigned int lid_cur=0 ; lid_cur < pNewKF->mvKeyEdges.size(); lid_cur++)
        {
            MapEdge* pME = pNewKF->GetMapEdge(lid_cur);
            if(pME && !pME->isBad())
                continue;
            KeyEdge ke_cur = pNewKF->mvKeyEdges[lid_cur];
            MapPoint *pMP1 = pNewKF->GetMapPoint(ke_cur.startIdx);
            MapPoint *pMP2 = pNewKF->GetMapPoint(ke_cur.endIdx);
            if(pMP1 == nullptr || pMP2 == nullptr || pMP1->isBad() || pMP2->isBad())
                continue;
            Eigen::Vector3f v_ = (pMP1->GetWorldPos() - pMP2->GetWorldPos()).normalized();
            Eigen::Vector3f v1_ = (pNewKF->GetCameraCenter() - pMP1->GetWorldPos()).normalized();
            Eigen::Vector3f v2_ = (pNewKF->GetCameraCenter() - pMP2->GetWorldPos()).normalized();
            if(std::fabs(v_.dot(v1_))>MapEdge::viewCosTh || std::fabs(v_.dot(v2_))>MapEdge::viewCosTh)
                continue;
            pME = pMP1->getEdge(pMP2);
            if(pME && !pME->isBad())
            {
                pNewKF->AddMapEdge(pME, lid_cur);
                pME->addObservation(pNewKF, lid_cur);
                continue;
            }
            pME = new MapEdge(pMP1, pMP2, this);
            pNewKF->AddMapEdge(pME, lid_cur);
            pME->addObservation(pNewKF, lid_cur);
            AddMapEdge(pME);
        }
        // add colines
        for(unsigned int pid_cur=0; pid_cur<pNewKF->mvKeysUn.size(); pid_cur++)
        {
            MapPoint* pMP = pNewKF->GetMapPoint(pid_cur);
            if(pMP == nullptr || pMP->isBad())
                continue;   
            const KeyPointEx &kp_cur = pNewKF->mvKeysUn[pid_cur];
            for(auto cp_cur : kp_cur.mvColine)
            {
                MapPoint* pMP1 = pNewKF->GetMapPoint(cp_cur.first);
                MapPoint* pMP2 = pNewKF->GetMapPoint(cp_cur.second);
                if(pMP1 == nullptr || pMP2 == nullptr || pMP1->isBad() || pMP2->isBad())
                    continue;
                MapColine* pMC = pMP->addColine(pMP1, pMP2, pNewKF);
                if(pMC)
                    AddMapColine(pMC);
            }
        }
    }    

    // Insert Keyframe in Map
    AddKeyFrame(pNewKF);  
}
