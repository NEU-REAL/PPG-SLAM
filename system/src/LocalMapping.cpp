#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Matcher.h"
#include "Optimizer.h"

#include<mutex>
#include<chrono>
void MSLocalMapping::Launch(Map *pMap)
{
    mbResetRequested = false;
    mbFinishRequested =false;
    mpMap = pMap;
    bInitializing = false;
    mbAbortBA = false;
    mbStopped = false;
    mbStopRequested = false;
    mbNotStop = false;
    mbLocalMappingIdle = true;
    mScale = 1.0;
    infoInertial = Eigen::MatrixXd::Zero(9,9);
    mTinit = 0.f;
    mptLocalMapping = new thread(&MSLocalMapping::Run, this);
}

void MSLocalMapping::Run()
{
    while(1)
    {
        // Tracking will see that Local Mapping is busy
        mbLocalMappingIdle = false;

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            // // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            mbAbortBA = false;

            if(!CheckNewKeyFrames())//FIXME 有没有对位姿估计精度影响不显著
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            } 

            int num_FixedKF_BA = 0;
            int num_OptKF_BA = 0;
            int num_MPs_BA = 0;
            int num_edges_BA = 0;
            if(mpMap->KeyFramesInMap()>2)
            {

                if(mpMap->isImuInitialized())
                {
                    bool bLarge = MSTracking::get().GetMatchesInliers()>75;
                    bLarge = false;
                    Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpMap,num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA, bLarge, !mpMap->GetIniertialBA2());
                    
                    float dist = (mpCurrentKeyFrame->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->GetCameraCenter()).norm() + (mpCurrentKeyFrame->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->mPrevKF->GetCameraCenter()).norm();
                    if(dist>0.05)
                        mTinit += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp;
                }
                else
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap,num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);

                // Initialize IMU here
                if(!mpMap->isImuInitialized())
                    InitializeIMU(1e2, 1e10, true);
                else 
                {
                    if (!mpMap->GetIniertialBA1() && mTinit > Map::imuIniTm) // TODO:imu initialization time, 10for euroc ,5 for uma, 10 for tum \\ warning IMU 初始化时间对结果影响很大
                    {
                        cout << "start visual inertial BA" << endl;
                        mpMap->SetIniertialBA1();
                        mpMap->SetIniertialBA2();
                        InitializeIMU(1.f, 1e5, true);
                        cout << "end visual inertial BA" << endl;
                    }
                    // scale refinement
                    if (((mpMap->KeyFramesInMap())<=200) && mpMap->KeyFramesInMap() %10 == 0)
                        ScaleRefinement();
                }
            }
            MSLoopClosing::get().InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            while(isStopped() && !mbFinishRequested)
                usleep(3000);
        }
        ResetIfRequested();

        mbLocalMappingIdle = true;

        if(mbFinishRequested)
        {
            mbFinishRequested = false;
            break;
        }

        usleep(3000);
    }
}

void MSLocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool MSLocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void MSLocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }
}

void MSLocalMapping::EmptyQueue()
{
    while(CheckNewKeyFrames())
        ProcessNewKeyFrame();
}

void MSLocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 30;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    for(int i=0, imax=vpTargetKFs.size(); i<imax; i++)
    {
        const vector<KeyFrame*> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
        }
        if (mbAbortBA)
            break;
    }

    // Extend to temporal neighbors
    KeyFrame* pKFi = mpCurrentKeyFrame->mPrevKF;
    while(vpTargetKFs.size()<20 && pKFi)
    {
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
        {
            pKFi = pKFi->mPrevKF;
            continue;
        }
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
        pKFi = pKFi->mPrevKF;
    }

    // Search matches by projection from current KF in target KFs
    Matcher matcher(mpMap->mpCamera);
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        matcher.Fuse(pKFi,vpMapPointMatches);
    }


    if (mbAbortBA)
        return;

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);

    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

void MSLocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool MSLocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool MSLocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool MSLocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void MSLocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool MSLocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void MSLocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void MSLocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21;
    mpCurrentKeyFrame->UpdateBestCovisibles();
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    float redundant_th = 0.9;

    const bool bInitImu = mpMap->isImuInitialized();
    int count=0;

    // Compoute last KF from optimizable window:
    unsigned int last_ID(0);
    {
        int count = 0;
        KeyFrame* aux_KF = mpCurrentKeyFrame;
        while(count<Nd && aux_KF->mPrevKF)
        {
            aux_KF = aux_KF->mPrevKF;
            count++;
        }
        last_ID = aux_KF->mnId;
    }



    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        count++;
        KeyFrame* pKF = *vit;

        if(pKF->mnId==mpMap->GetOriginKF()->mnId || pKF->isBad())
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const map<KeyFrame*, int> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, int>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            int index = mit->second;
                            nObs++;
                            if(nObs>thObs)
                                break;
                        }
                        if(nObs>thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if(nRedundantObservations>redundant_th*nMPs)
        {
            if (mpMap->KeyFramesInMap()<=Nd)
                continue;

            if(pKF->mnId>(mpCurrentKeyFrame->mnId-2))
                continue;
            if(pKF->mnId==mpMap->GetOriginKF()->mnId)
                continue;
            if(pKF->mPrevKF && pKF->mNextKF)
            {
                const float t = pKF->mNextKF->mTimeStamp-pKF->mPrevKF->mTimeStamp;

                if((bInitImu && (pKF->mnId<last_ID) && t<1.) || (t<0.5)) //byz:少剔除一些关键帧，间隔3秒改为1秒
                {
                    pKF->SetBadFlag();
                    mpMap->EraseKeyFrame(pKF);
                }
                else if(!mpMap->GetIniertialBA2() && ((pKF->GetImuPosition()-pKF->mPrevKF->GetImuPosition()).norm()<0.02) && (t<1)) //byz:少剔除一些关键帧，间隔3秒改为1秒
                {
                    pKF->SetBadFlag();
                    mpMap->EraseKeyFrame(pKF);
                }
            }

        }
        if((count > 20 && mbAbortBA) || count>100)
        {
            break;
        }
    }
}

void MSLocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
    cout << "LM: Map reset, waiting..." << endl;

    while(1)
    {
        usleep(3000);
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
    }
    cout << "LM: Map reset, Done!!!" << endl;
}

void MSLocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        cout << "LM: Reseting map in Local Mapping..." << endl;
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested = false;
        // Inertial parameters
        mTinit = 0.f;
        cout << "LM: End reseting Local Mapping..." << endl;
    }
}

void MSLocalMapping::RequestFinish()
{
    mbFinishRequested = true;
    while(mbFinishRequested == true)
        usleep(3000);
    std::cout << "Localmapping finished."<<std::endl;
}

void MSLocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    if (mbResetRequested)
        return;

    if(mpMap->KeyFramesInMap()<10)
        return;

    // Retrieve all keyframe in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    if(vpKF.size()<10)
        return;

    mFirstTs=vpKF.front()->mTimeStamp;
    if(mpCurrentKeyFrame->mTimeStamp-mFirstTs< 2.0)
        return;

    bInitializing = true;

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0);

    // Compute and KF velocities mRwg estimation
    if (!mpMap->isImuInitialized())
    {
        Eigen::Matrix3f Rwg;
        Eigen::Vector3f dirG;
        dirG.setZero();
        for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated)
                continue;
            if (!(*itKF)->mPrevKF)
                continue;

            dirG -= (*itKF)->mPrevKF->GetImuRotation() * (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            Eigen::Vector3f _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);
            (*itKF)->mPrevKF->SetVelocity(_vel);
        }

        dirG = dirG/dirG.norm();
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f);
        Eigen::Vector3f v = gI.cross(dirG);
        const float nv = v.norm();
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        Eigen::Vector3f vzg = v*ang/nv;
        Rwg = Sophus::SO3f::exp(vzg).matrix();
        mRwg = Rwg.cast<double>();
        mTinit = mpCurrentKeyFrame->mTimeStamp-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = mpCurrentKeyFrame->GetGyroBias().cast<double>();
        mba = mpCurrentKeyFrame->GetAccBias().cast<double>();
    }

    mScale=1.0;

    Optimizer::InertialOptimization(mpMap, mRwg, mScale, mbg, mba, infoInertial, false, false, priorG, priorA);

    if (mScale<1e-1)
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }

    // Before this line we are not changing the map
    {
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        if ((fabs(mScale - 1.f) > 0.00001)) {
            Sophus::SE3f Twg(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());
            mpMap->ApplyScaledRotation(Twg, mScale, true);
            MSTracking::get().UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
        }

        // Check if initialization OK
        if (!mpMap->isImuInitialized())
            for (int i = 0; i < N; i++) {
                KeyFrame *pKF2 = vpKF[i];
                pKF2->bImu = true;
            }
    }

    MSTracking::get().UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);
    if (!mpMap->isImuInitialized())
    {
        mpMap->SetImuInitialized();
        mpCurrentKeyFrame->bImu = true;
    }

    if (bFIBA)
    {
        if (priorA!=0.f)
            Optimizer::FullInertialBA(mpMap, 100, mpCurrentKeyFrame->mnId, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpMap, 100, mpCurrentKeyFrame->mnId, NULL, false);
    }

    std::cout << "Global Bundle Adjustment finished" << std::endl << "Updating map ..."<< std::endl;

    // Get Map Mutex
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    unsigned long GBAid = mpCurrentKeyFrame->mnId;

    // Process keyframes in the queue
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // Correct All keyframes
    std::vector<KeyFrame*> vpAllKFs = mpMap->GetAllKeyFrames();
    KeyFrame* pOirKF = mpMap->GetOriginKF();
    for(KeyFrame* pKF : vpAllKFs)
    {
        if(!pKF || pKF->isBad())
            continue;
        if(pKF->mnBAGlobalForKF != GBAid)
        {
            pKF->mTcwGBA = pKF->GetPose() * pOirKF->GetPoseInverse() * pKF->mTcwGBA;
            if(pKF->isVelocitySet())
                pKF->mVwbGBA = pKF->mTcwGBA.so3().inverse() * pKF->GetPose().so3() * pKF->GetVelocity();
            else
                std::cerr<< "GBA velocity empty!! "<< pKF->mnId <<std::endl;
            pKF->mnBAGlobalForKF = GBAid;
            pKF->mBiasGBA = pKF->GetImuBias();
        }

        pKF->mTcwBefGBA = pKF->GetPose();
        pKF->SetPose(pKF->mTcwGBA);
        if(pKF->bImu)
        {
            pKF->mVwbBefGBA = pKF->GetVelocity();
            pKF->SetVelocity(pKF->mVwbGBA);
            pKF->SetNewBias(pKF->mBiasGBA);
        }
        else
            cerr << " GBA no inertial!! "<< pKF->mnId<<std::endl;
    }

    // Correct MapPoints
    const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        if(pMP->mnBAGlobalForKF==GBAid)
        {
            // If optimized by Global BA, just update
            pMP->SetWorldPos(pMP->mPosGBA);
        }
        else
        {
            // Update according to the correction of its reference keyframe
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

            if(pRefKF->mnBAGlobalForKF!=GBAid)
                continue;

            // Map to non-corrected camera
            Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

            // Backproject using corrected camera
            pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
        }
    }
    // remove bad colinearity edges
    vector<MapEdge*> vpMEs = mpMap->GetAllMapEdges();
    for(MapEdge* pME : vpMEs)
    {
        if(!pME || pME->isBad() || !pME->mbValid)
            continue;
        pME->checkValid();
    }
    for(MapPoint* pMP : vpMPs)
    {
        if(pMP == nullptr || pMP->isBad())
            continue;
        pMP->removeColineOutliers();
    }

    std::cout<<"Map updated!"<<std::endl;

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        mpMap->EraseKeyFrame((*lit));
        delete *lit;
    }
    mlNewKeyFrames.clear();

    MSTracking::get().mState=OK;
    bInitializing = false;

    mpMap->InfoMapChange();

    return;
}

void MSLocalMapping::ScaleRefinement()
{
    if (mbResetRequested)
        return;
    // Retrieve all keyframes in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();

    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;

    Optimizer::InertialOptimization(mpMap, mRwg, mScale);

    if (mScale<1e-1) // 1e-1
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }
    
    Sophus::SO3d so3wg(mRwg);
    // Before this line we are not changing the map
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    if ((fabs(mScale-1.f)>0.002))
    {
        Sophus::SE3f Tgw(mRwg.cast<float>().transpose(),Eigen::Vector3f::Zero());
        mpMap->ApplyScaledRotation(Tgw,mScale,true);
        MSTracking::get().UpdateFrameIMU(mScale,mpCurrentKeyFrame->GetImuBias(),mpCurrentKeyFrame);
    }
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        mpMap->EraseKeyFrame(pKF);
        delete *lit;
    }
    mlNewKeyFrames.clear();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpMap->InfoMapChange();

    return;
}

double MSLocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

KeyFrame* MSLocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

void MSLocalMapping::increMap(KeyFrame* pnewKF)
{
    InsertKeyFrame(pnewKF);

    mpCurrentKeyFrame = pnewKF;

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP && !pMP->isBad())
        {
            pMP->AddObservation(mpCurrentKeyFrame, i);
            pMP->UpdateNormalAndDepth();
            pMP->ComputeDistinctiveDescriptors();
        }
    }
    // update Edge observation
    for(unsigned int lid_cur=0; lid_cur<mpCurrentKeyFrame->mvKeyEdges.size(); lid_cur++)
    {
        MapPoint* pMP1 = mpCurrentKeyFrame->GetMapPoint(mpCurrentKeyFrame->mvKeyEdges[lid_cur].startIdx);
        MapPoint* pMP2 = mpCurrentKeyFrame->GetMapPoint(mpCurrentKeyFrame->mvKeyEdges[lid_cur].endIdx);
        if(pMP1 == nullptr || pMP2 == nullptr || pMP1->isBad() || pMP2->isBad())
            continue;
        Eigen::Vector3f v_ = (pMP1->GetWorldPos() - pMP2->GetWorldPos()).normalized();
        Eigen::Vector3f v1_ = (mpCurrentKeyFrame->GetCameraCenter() - pMP1->GetWorldPos()).normalized();
        Eigen::Vector3f v2_ = (mpCurrentKeyFrame->GetCameraCenter() - pMP1->GetWorldPos()).normalized();
        if(std::fabs(v_.dot(v1_))>0.95 || std::fabs(v_.dot(v2_))>0.95)
            continue;
        MapEdge *pME = pMP1->getEdge(pMP2);
        if(pME && !pME->isBad())
        {
            mpCurrentKeyFrame->AddMapEdge(pME, lid_cur);
            pME->addObservation(mpCurrentKeyFrame, lid_cur);
            pME->checkValid();
        }
    }
    // update coline observation
    for(unsigned int pid_cur=0; pid_cur<mpCurrentKeyFrame->mvKeysUn.size(); pid_cur++)
    {
        MapPoint* pMP = mpCurrentKeyFrame->GetMapPoint(pid_cur);
        if(pMP == nullptr || pMP->isBad())
            continue;   
        const KeyPointEx &kp_cur = mpCurrentKeyFrame->mvKeysUn[pid_cur];
        for(auto cp_cur : kp_cur.mvColine)
        {
            MapPoint* pMPs = mpCurrentKeyFrame->GetMapPoint(cp_cur.first);
            MapPoint* pMPe = mpCurrentKeyFrame->GetMapPoint(cp_cur.second);
            if(pMPs == nullptr || pMPe == nullptr || pMPs->isBad() || pMPe->isBad())
                continue;
            MapColine* pMC = pMP->addColine(pMPs, pMPe, mpCurrentKeyFrame);
            if(pMC)
                mpMap->AddMapColine(pMC);
        }
    }
    
    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int borrar = mlpRecentAddedMapPoints.size();

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
            lit = mlpRecentAddedMapPoints.erase(lit);
        else if(pMP->GetFoundRatio()<0.25f)
        {
            pMP->SetBadFlag();
            mpMap->EraseMapPoint(pMP);
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=2)
        {
            pMP->SetBadFlag();
            mpMap->EraseMapPoint(pMP);
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
    KeyFrame* pKF = mpCurrentKeyFrame;
    int count=0; 
    while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn))
    {
        vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
        if(it==vpNeighKFs.end())
            vpNeighKFs.push_back(pKF->mPrevKF);
        pKF = pKF->mPrevKF;
    }
    float th = 0.6f;
    Matcher matcher(mpMap->mpCamera, th);

    Sophus::SE3<float> sophTcw1 = mpCurrentKeyFrame->GetPose();
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4();
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0);
    Eigen::Matrix<float,3,3> Rwc1 = Rcw1.transpose();
    Eigen::Vector3f tcw1 = sophTcw1.translation();
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        GeometricCamera* pCamera = mpMap->mpCamera;

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        bool bCoarse = MSTracking::get().mState==RECENTLY_LOST && mpMap->GetIniertialBA2();

        
        matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices, bCoarse);

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

            const KeyPointEx &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
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
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
        for( unsigned int lid_cur=0 ; lid_cur < mpCurrentKeyFrame->mvKeyEdges.size(); lid_cur++)
        {
            MapEdge* pME = mpCurrentKeyFrame->GetMapEdge(lid_cur);
            if(pME && !pME->isBad())
                continue;
            KeyEdge ke_cur = mpCurrentKeyFrame->mvKeyEdges[lid_cur];
            MapPoint *pMP1 = mpCurrentKeyFrame->GetMapPoint(ke_cur.startIdx);
            MapPoint *pMP2 = mpCurrentKeyFrame->GetMapPoint(ke_cur.endIdx);
            if(pMP1 == nullptr || pMP2 == nullptr || pMP1->isBad() || pMP2->isBad())
                continue;
            Eigen::Vector3f v_ = (pMP1->GetWorldPos() - pMP2->GetWorldPos()).normalized();
            Eigen::Vector3f v1_ = (mpCurrentKeyFrame->GetCameraCenter() - pMP1->GetWorldPos()).normalized();
            Eigen::Vector3f v2_ = (mpCurrentKeyFrame->GetCameraCenter() - pMP2->GetWorldPos()).normalized();
            if(std::fabs(v_.dot(v1_))>0.95 || std::fabs(v_.dot(v2_))>0.95)
                continue;
            pME = pMP1->getEdge(pMP2);
            if(pME && !pME->isBad())
            {
                mpCurrentKeyFrame->AddMapEdge(pME, lid_cur);
                pME->addObservation(mpCurrentKeyFrame, lid_cur);
                continue;
            }
            pME = new MapEdge(pMP1, pMP2, mpMap);
            mpCurrentKeyFrame->AddMapEdge(pME, lid_cur);
            pME->addObservation(mpCurrentKeyFrame, lid_cur);
            mpMap->AddMapEdge(pME);
        }
        // add colines
        for(unsigned int pid_cur=0; pid_cur<mpCurrentKeyFrame->mvKeysUn.size(); pid_cur++)
        {
            MapPoint* pMP = mpCurrentKeyFrame->GetMapPoint(pid_cur);
            if(pMP == nullptr || pMP->isBad())
                continue;   
            const KeyPointEx &kp_cur = mpCurrentKeyFrame->mvKeysUn[pid_cur];
            for(auto cp_cur : kp_cur.mvColine)
            {
                MapPoint* pMP1 = mpCurrentKeyFrame->GetMapPoint(cp_cur.first);
                MapPoint* pMP2 = mpCurrentKeyFrame->GetMapPoint(cp_cur.second);
                if(pMP1 == nullptr || pMP2 == nullptr || pMP1->isBad() || pMP2->isBad())
                    continue;
                MapColine* pMC = pMP->addColine(pMP1, pMP2, mpCurrentKeyFrame);
                if(pMC)
                    mpMap->AddMapColine(pMC);
            }
        }
    }    
}