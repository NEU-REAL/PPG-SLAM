
#include "KeyFrame.h"
#include "IMU.h"
#include <mutex>
#include <algorithm>
#include <list>
#include <iostream>

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame():
        mnFrameId(0),  mTimeStamp(0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
        mfGridElementWidthInv(0), mfGridElementHeightInv(0),
        mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), 
        mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnMergeQuery(0), mnMergeWords(0), mnBAGlobalForKF(0),
        mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
        N(0), mvKeys(std::vector<KeyPointEx>()), mvKeysUn(std::vector<KeyPointEx>()),
        mvKeyEdges(std::vector<KeyEdge>()), mnMinX(0), mnMinY(0), mnMaxX(0),
        mnMaxY(0), mPrevKF(nullptr), mNextKF(nullptr), mbNotErase(false),
        mbToBeErased(false), mbBad(false), 
        mnNumberOfOpt(0), mbHasVelocity(false) 
{
    mnId=nNextId++;
    mGrid.resize(FRAME_GRID_COLS, std::vector<std::vector<std::size_t>>(FRAME_GRID_ROWS, std::vector<std::size_t>()));
}

void KeyFrame::SetPose(const Sophus::SE3f &Tcw)
{
    std::unique_lock<std::mutex> lock(mMutexPose);

    mTcw = Tcw;
    mRcw = mTcw.rotationMatrix();
    mTwc = mTcw.inverse();
    mRwc = mTwc.rotationMatrix();

    if (mpImuCalib->mbIsSet) // TODO Use a flag instead of the OpenCV matrix
    {
        mOwb = mRwc * mpImuCalib->mTcb.translation() + mTwc.translation();
    }
}

void KeyFrame::SetVelocity(const Eigen::Vector3f &Vw)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mVw = Vw;
    mbHasVelocity = true;
}

Sophus::SE3f KeyFrame::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTcw;
}

Sophus::SE3f KeyFrame::GetPoseInverse()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTwc;
}

Eigen::Vector3f KeyFrame::GetCameraCenter(){
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTwc.translation();
}

Eigen::Vector3f KeyFrame::GetImuPosition()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mOwb;
}

Eigen::Matrix3f KeyFrame::GetImuRotation()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return (mTwc * mpImuCalib->mTcb).rotationMatrix();
}

Sophus::SE3f KeyFrame::GetImuPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTwc * mpImuCalib->mTcb;
}

Eigen::Matrix3f KeyFrame::GetRotation(){
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mRcw;
}

Eigen::Vector3f KeyFrame::GetTranslation()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTcw.translation();
}

Eigen::Vector3f KeyFrame::GetVelocity()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mVw;
}

bool KeyFrame::isVelocitySet()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mbHasVelocity;
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    std::vector<std::pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(std::map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(std::make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    std::list<KeyFrame*> lKFs;
    std::list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        if(!vPairs[i].second->isBad())
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
    }

    mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
}

std::set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    std::set<KeyFrame*> s;
    for(std::map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

std::vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
    {
        return std::vector<KeyFrame*>();
    }

    std::vector<int>::iterator it = std::upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);

    if(it==mvOrderedWeights.end() && mvOrderedWeights.back() < w)
    {
        return std::vector<KeyFrame*>();
    }
    else
    {
        int n = it-mvOrderedWeights.begin();
        return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const int &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int index = pMP->GetIndexInKeyFrame(this);
    if(index != -1)
        mvpMapPoints[index]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const int &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

std::set<MapPoint*> KeyFrame::GetMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    std::set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

std::vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections(bool upParent)
{
    std::map<KeyFrame*,int> KFcounter;

    std::vector<MapPoint*> vpMP;

    {
        std::unique_lock<std::mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(std::vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        std::map<KeyFrame*, int> observations = pMP->GetObservations();

        for(std::map<KeyFrame*, int>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId || mit->first->isBad())
                continue;
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 5; //byz:增加共视关键帧数量

    std::vector<std::pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    if(!upParent)
        std::cout << "UPDATE_CONN: current KF " << mnId << std::endl;
    for(std::map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(!upParent)
            std::cout << "  UPDATE_CONN: KF " << mit->first->mnId << " ; num matches: " << mit->second << std::endl;
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            vPairs.push_back(std::make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    if(vPairs.empty())
    {
        vPairs.push_back(std::make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    sort(vPairs.begin(),vPairs.end());
    std::list<KeyFrame*> lKFs;
    std::list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);

        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

    }
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

std::set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(std::map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);

    for(size_t i=0; i<mvpMapPoints.size(); i++)
    {
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);
    }

    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        mNextKF->mpImuPreintegrated->MergePrevious(mpImuPreintegrated);
        mTcp = mTcw * mPrevKF->GetPoseInverse();
        mNextKF->mPrevKF = mPrevKF;
        mPrevKF->mNextKF = mNextKF;
        mbBad = true;
    }
}

bool KeyFrame::isBad()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}


std::vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    std::vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = std::max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = std::min((int)mnGridCols-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = std::max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = std::min((int)mnGridRows-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const std::vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const KeyPointEx &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.mPos[0]-x;
                const float disty = kpUn.mPos[1]-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    if(N==0)
        return -1.0;

    std::vector<MapPoint*> vpMapPoints;
    Eigen::Matrix3f Rcw;
    Eigen::Vector3f tcw;
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        tcw = mTcw.translation();
        Rcw = mRcw;
    }

    std::vector<float> vDepths;
    vDepths.reserve(N);
    Eigen::Matrix<float,1,3> Rcw2 = Rcw.row(2);
    float zcw = tcw(2);
    for(int i=0; i<N; i++) {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            Eigen::Vector3f x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw) + zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

void KeyFrame::SetNewBias(const IMU::Bias &b)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

Eigen::Vector3f KeyFrame::GetGyroBias()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
}

Eigen::Vector3f KeyFrame::GetAccBias()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bax, mImuBias.bay, mImuBias.baz);
}

IMU::Bias KeyFrame::GetImuBias()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mImuBias;
}

void KeyFrame::SetVocabulary(DBoW3::Vocabulary* pVoc)
{
    mpVocabulary = pVoc;
}

void KeyFrame::AddMapEdge(MapEdge* pME, const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapEdges[idx]=pME;
}

MapEdge* KeyFrame::GetMapEdge(int idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapEdges[idx];
}

int KeyFrame::FineEdgeIdx(unsigned int p1_id, unsigned int p2_id)
{
    if(p1_id < 0 || p1_id >= mvKeysUn.size() || p2_id < 0 || p2_id >= mvKeysUn.size())
        return -1;

    std::vector<unsigned int> l1_indices = mvKeysUn[p1_id].mvConnected;
    std::vector<unsigned int> l2_indices = mvKeysUn[p1_id].mvConnected;

    for(unsigned int l1_id : l1_indices)
    {
        for(unsigned int l2_id : l2_indices)
        {
            if(l1_id == l2_id)
                return l1_id;
        }
    }
    return -1;
}