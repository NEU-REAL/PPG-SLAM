#include "Tracking.h"
#include "Matcher.h"
#include "G2oVertex.h"
#include "G2oEdge.h"
#include "Optimizer.h"
#include "GeometricCamera.h"
#include "MLPnPsolver.h"
#include "Frame.h"

#include <iostream>

#include <mutex>
#include <chrono>

using namespace std;

void MSTracking::Launch(Map* pMap, const string &strNet)
{
    mState = NO_IMAGES_YET;
    mLastProcessedState = NO_IMAGES_YET;
    mpExtractor = nullptr;
    mpReferenceKF = nullptr;
    mpMap = pMap;
    mpLastKeyFrame = nullptr;
    mnLastRelocFrameId = 0;
    mTimeStampLost = 0;
    mbMapUpdated = false;
    mbReadyToInitializate = false;

    // Initialize IMU variables
    mScale = 1.0;
    infoInertial = Eigen::MatrixXd::Zero(9,9);
    mFirstTs = 0.f;
    bInitializing = false;
    mTinit = 0.f;

    mpCamera = pMap->mpCamera;
    mpImuCalib = pMap->mpImuCalib;

    mpExtractor = new PPGExtractor(mpCamera, strNet);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), mpImuCalib);
}


SE3f MSTracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    mImGray = im;
    mCurrentFrame = Frame(mImGray, timestamp, mpExtractor, mpCamera, mpImuCalib, &mLastFrame);
    Track();
    return mCurrentFrame.GetPose();
}

void MSTracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void MSTracking::PreintegrateIMU()
{
    if (!mCurrentFrame.mpPrevFrame)
    {
        std::cerr<< "non prev frame "<<std::endl;
        mCurrentFrame.mbImuPreintegrated = true;
        return;
    }
    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if (mlQueueImuData.size() == 0)
    {
        std::cerr<<"Not IMU data in mlQueueImuData!!" <<std::endl;
        mCurrentFrame.mbImuPreintegrated = true;
        return;
    }
    while (true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if (!mlQueueImuData.empty())
            {
                IMU::Point *m = &mlQueueImuData.front();
                cout.precision(17);
                if (m->t < mCurrentFrame.mpPrevFrame->mTimeStamp - mpImuCalib->mImuPer)
                {
                    mlQueueImuData.pop_front();
                }
                else if (m->t < mCurrentFrame.mTimeStamp - mpImuCalib->mImuPer)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if (bSleep)
            usleep(500);
    }
    const int n = mvImuFromLastFrame.size() - 1;
    if (n == 0)
    {
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }
    IMU::Preintegrated *pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias, mpImuCalib);
    for (int i = 0; i < n; i++)
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if ((i == 0) && (i < (n - 1)))
        {
            float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
                   (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tini / tab)) *
                  0.5f;
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                      (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tini / tab)) *
                     0.5f;
            tstep = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if (i < (n - 1))
        {
            acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a) * 0.5f;
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w) * 0.5f;
            tstep = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
        }
        else if ((i > 0) && (i == (n - 1)))
        {
            float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
                   (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tend / tab)) *
                  0.5f;
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                      (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tend / tab)) *
                     0.5f;
            tstep = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;
        }
        else if ((i == 0) && (i == (n - 1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc, angVel, tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
    }
    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;
    mCurrentFrame.mbImuPreintegrated = true;
}

bool MSTracking::PredictStateIMU()
{
    if (!mCurrentFrame.mpPrevFrame)
    {
        std::cerr<< "No last frame" <<std::endl;
        return false;
    }
    if (mbMapUpdated && mpLastKeyFrame)
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        return true;
    }
    else if (!mbMapUpdated)
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

        mCurrentFrame.mImuBias = mLastFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;
    return false;
}

void MSTracking::ResetFrameIMU()
{
    // TODO To implement...
}

void MSTracking::Track()
{
    if (MSViewing::get().mbStepByStep)
    {
        std::cout << "Tracking: Waiting to the next step" << std::endl;
        while (!MSViewing::get().mbStep && MSViewing::get().mbStepByStep)
            usleep(500);
        MSViewing::get().mbStep = false;
    }
    if (mState != NO_IMAGES_YET && (mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp || mCurrentFrame.mTimeStamp > mLastFrame.mTimeStamp + 1.0))
    {
        cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
        cerr << "ERROR: Timestamp jump detected!" << endl;
        cerr << "Last frame timestamp: " << mLastFrame.mTimeStamp << endl;
        cerr << "Current frame timestamp: " << mCurrentFrame.mTimeStamp << endl;
        unique_lock<mutex> lock(mMutexImuQueue);
        mlQueueImuData.clear();
        Reset();
        return;
    }
    if (mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());
    if (mState == NO_IMAGES_YET)
        mState = NOT_INITIALIZED;
    mLastProcessedState = mState;
    PreintegrateIMU();

    // initialize
    if (mState == NOT_INITIALIZED)
    {
        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        MonocularInitialization();
        MSViewing::get().UpdateFrame(mCurrentFrame);
        if (mState != OK) // If rightly initialized, mState=OK
            mLastFrame = Frame(mCurrentFrame);
        else
        {
            SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        }
        return;
    }

    // Update time for IMU initialization if IMU is already initialized
    if(mpMap->isImuInitialized() && mpLastKeyFrame && mpLastKeyFrame->mPrevKF)
    {
        float dist = (mpLastKeyFrame->mPrevKF->GetCameraCenter() - mpLastKeyFrame->GetCameraCenter()).norm();
        if(dist>0.05)
            mTinit += mpLastKeyFrame->mTimeStamp - mpLastKeyFrame->mPrevKF->mTimeStamp;
    }

    // Initialize IMU here
    if(!mpMap->isImuInitialized())
        InitializeIMU(1e2, 1e10, true);
    else 
    {
        if (!mpMap->GetInertialBA() && mTinit>Map::imuIniTm) // TODO:imu initialization time, 10for euroc ,5 for uma, 10 for tum \\ warning IMU 初始化时间对结果影响很大
        {
            cout << "start visual inertial BA" << endl;
            mpMap->SetInertialBA();
            InitializeIMU(1.f, 1e5, true);
            cout << "end visual inertial BA" << endl;
        }
        // scale refinement
        if (((mpMap->KeyFramesInMap())<=100) && mpMap->KeyFramesInMap() %20 == 0)
            ScaleRefinement();
    }

    mbMapUpdated = mpMap->CheckMapChanged();
    // track reference
    bool bOK(false);
    {
        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        // Local Mapping might have changed some MapPoints tracked in last frame
        CheckReplacedInLastFrame();
        if (!mpMap->isImuInitialized())
        {
            if(mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                bOK = TrackReferenceKeyFrame();
            else
            {
                bOK = TrackWithMotionModel();
                if (!bOK)
                    bOK = TrackReferenceKeyFrame();
            }
        }
        else
            bOK = PredictStateIMU();
        if (!bOK)
        {
            std::cerr<< "Track Reference KF Lost, Reseting current map..." <<std::endl;
            mState = LOST;
            std::cerr<< "done" <<std::endl;
            return;
        }
        // track local map
        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if (mState !=LOST && bOK)
            bOK = TrackLocalMap();

        if (mState !=LOST && bOK)
        {
            mTimeStampLost = mCurrentFrame.mTimeStamp;
            mState = OK;
        }
        else if(mpMap->isImuInitialized())
        {
            mState = RECENTLY_LOST;
            cout << "Fail to track local map! IMU only ..." << endl;
            if(mCurrentFrame.mTimeStamp - mTimeStampLost < 5.)
            {
                bOK = true;
                PredictStateIMU();
            }else
            {
                std::cerr<< "IMU takes too long, reseting current map..."<<std::endl;
                mState = LOST;
                std::cerr<< "done" <<std::endl;
                return;
            }
        }
        else
        {
            std::cerr<< "Fail to track local map, reseting current map..." <<std::endl;
            mState = LOST;
            std::cerr<< "done" <<std::endl;
            return;
        }

        if(mState!= LOST)
        {
            if (mpMap->isImuInitialized())
            {
                if (bOK)
                {
                    if (mCurrentFrame.mnId == (mnLastRelocFrameId + mpCamera->mfFps))
                    {
                        cout << "RESETING FRAME!!!" << endl;
                        ResetFrameIMU();
                    }
                    else if (mCurrentFrame.mnId > (mnLastRelocFrameId + 30))
                        mLastBias = mCurrentFrame.mImuBias;
                }
            }
                
            if (bOK || mState == RECENTLY_LOST)
            {
                // Update motion model
                if (mLastFrame.HasPose() && mCurrentFrame.HasPose())
                {
                    SE3f LastTwc = mLastFrame.GetPose().inverse();
                    mVelocity = mCurrentFrame.GetPose() * LastTwc;
                }
                // Clean VO matches
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1)
                        {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }

                bool bNeedKF = NeedNewKeyFrame();
                // Check if we need to insert a new keyframe
                // if(bNeedKF && bOK)
                if (bNeedKF && (bOK || (mInsertKFsLost && mState == RECENTLY_LOST)))
                    CreateNewKeyFrame();
                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame. Only has effect if lastframe is tracked
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                // Store frame pose information to retrieve the complete camera trajectory afterwards.
                if (mCurrentFrame.HasPose())
                {
                    SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
                    mlRelativeFramePoses.push_back(Tcr_);
                    mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
                    mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                    mlbLost.push_back(mState == LOST);
                }
                else
                {
                    // This can happen if tracking is lost
                    mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
                    mlpReferences.push_back(mlpReferences.back());
                    mlFrameTimes.push_back(mlFrameTimes.back());
                    mlbLost.push_back(mState == LOST);
                }
            }

            // Update drawer
            MSViewing::get().UpdateFrame(mCurrentFrame);
            if (mCurrentFrame.HasPose())
                MSViewing::get().SetCurrentCameraPose(mCurrentFrame.GetPose());

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
        }
    }
    // Reset if the camera get lost soon after initialization
    if (mState == LOST)
    {
        if (mpMap->KeyFramesInMap() <= 10 || !mpMap->isImuInitialized())
        {
            Reset();
            return;
        }
    }
}

void MSTracking::MonocularInitialization()
{

    if (!mbReadyToInitializate)
    {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size() > 50)
        {

            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = cv::Point2f(mCurrentFrame.mvKeysUn[i].mPos[0],mCurrentFrame.mvKeysUn[i].mPos[1]);

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            if (mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            
            mbReadyToInitializate = true;

            return;
        }
    }
    else
    {
        if ( (int)mCurrentFrame.mvKeys.size() <= 50 || (mLastFrame.mTimeStamp - mInitialFrame.mTimeStamp) > 1.0)
        {
            mbReadyToInitializate = false;
            return;
        }

        // Find correspondences
        Matcher matcher(mpMap->mpCamera, 0.9);
        int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 50);

        // Check if there are enough correspondences
        if (nmatches < 50)
        {
            mbReadyToInitializate = false;
            return;
        }

        SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if (mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn, mCurrentFrame.mvKeysUn, mvIniMatches, Tcw, mvIniP3D, vbTriangulated))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(SE3f());
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void MSTracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame *pKFini = mInitialFrame.buildKeyFrame(mpMap);
    KeyFrame *pKFcur = mCurrentFrame.buildKeyFrame(mpMap);
    pKFini->mpImuPreintegrated = (IMU::Preintegrated *)(NULL);
    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);
    // map points
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
            continue;
        // Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint *pMP = new MapPoint(worldPos, pKFcur);
        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);
        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);
        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();
        // Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;
        // Add to Map
        mpMap->AddMapPoint(pMP);
    }
    // map edges
    for( unsigned int lid_cur=0 ; lid_cur < pKFcur->mvKeyEdges.size(); lid_cur++)
    {
        KeyEdge kl_cur = pKFcur->mvKeyEdges[lid_cur];
        MapPoint *pMP1 = pKFcur->GetMapPoint(kl_cur.startIdx);
        MapPoint *pMP2 = pKFcur->GetMapPoint(kl_cur.endIdx);
        if(pMP1 == nullptr || pMP2 == nullptr || pMP1->isBad() || pMP2->isBad())
            continue;
        Eigen::Vector3f v_ = (pMP1->GetWorldPos() - pMP2->GetWorldPos()).normalized();
        Eigen::Vector3f v1_ = (pKFcur->GetCameraCenter() - pMP1->GetWorldPos()).normalized();
        Eigen::Vector3f v2_ = (pKFcur->GetCameraCenter() - pMP2->GetWorldPos()).normalized();
        if(std::fabs(v_.dot(v1_))>MapEdge::viewCosTh || std::fabs(v_.dot(v2_))>MapEdge::viewCosTh)
            continue;
        MapEdge * pME = new MapEdge(pMP1, pMP2);
        pME->addObservation(pKFcur, lid_cur);
        pKFcur->AddMapEdge(pME, lid_cur);
        mpMap->AddMapEdge(pME);
    }
    // map colines
    for( unsigned int pid_cur=0 ; pid_cur < pKFcur->mvKeysUn.size(); pid_cur++)
    {
        const KeyPointEx &kp_cur = pKFcur->mvKeysUn[pid_cur];
        MapPoint *pMP_cur = pKFcur->GetMapPoint(pid_cur);
        if(pMP_cur == nullptr || pMP_cur->isBad())
            continue;
        for(auto cp_cur : kp_cur.mvColine)
        {
            MapPoint *pMPs = pKFcur->GetMapPoint(cp_cur.first);
            MapPoint *pMPe = pKFcur->GetMapPoint(cp_cur.second);
            if(pMPs == nullptr || pMPe == nullptr || pMPs->isBad() || pMPe->isBad())
                continue;
            MapColine* pMC = pMP_cur->addColine(pMPs,pMPe, pKFcur);
            if(pMC)
                mpMap->AddMapColine(pMC);
        }
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint *> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    std::cout<< "New Map created with " << to_string(mpMap->MapPointsInMap()) << " points"<< std::endl;
    Optimizer::GlobalBundleAdjustment(mpMap, 20);
    // Compute scene median depth inline
    float medianDepth = -1.0f;
    if (pKFini->N > 0) {
        std::vector<float> vDepths;
        vDepths.reserve(pKFini->N);
        
        const Eigen::Matrix<float,1,3> Rcw2 = pKFini->GetRotation().row(2);
        const float zcw = pKFini->GetTranslation()(2);
        
        const std::vector<MapPoint*> vpMapPoints = pKFini->GetMapPointMatches();
        for (int i = 0; i < pKFini->N; i++) {
            if (vpMapPoints[i]) {
                const Eigen::Vector3f x3Dw = vpMapPoints[i]->GetWorldPos();
                vDepths.push_back(Rcw2.dot(x3Dw) + zcw);
            }
        }
        
        if (!vDepths.empty()) {
            std::sort(vDepths.begin(), vDepths.end());
            medianDepth = vDepths[(vDepths.size()-1)/2];  // q=2 means median
        }
    }
    
    float invMedianDepth = 4.0f / medianDepth; // 4.0f

    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) // TODO Check, originally 100 tracks
    {
        std::cerr<<"Wrong initialization, reseting..."<<std::endl;
        Reset();
        return;
    }

    // Scale initial baseline
    SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            MapPoint *pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }
    // remove bad colinearity edges
    vector<MapEdge*> vpMEs = mpMap->GetAllMapEdges();
    for(MapEdge* pME : vpMEs)
    {
        if(pME== nullptr || pME->isBad())
            continue;
        pME->checkValid();
    }
    for(MapPoint* pMP : vpAllMapPoints)
    {
        if(pMP == nullptr || pMP->isBad())
            continue;
        pMP->removeColineOutliers();
    }

    pKFcur->mPrevKF = pKFini;
    pKFini->mNextKF = pKFcur;
    pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(), mpImuCalib);

    MSLocalMapping::get().InsertKeyFrame(pKFini);
    MSLocalMapping::get().InsertKeyFrame(pKFcur);
    MSLocalMapping::get().mFirstTs = pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mpLastKeyFrame = pKFcur;
    // mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    MSViewing::get().SetCurrentCameraPose(pKFcur->GetPose());

    mState = OK;
}

void MSTracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (!pMP)
            continue;
        if (pMP->mpReplaced)
            mLastFrame.mvpMapPoints[i] = pMP->mpReplaced;
    }
}

bool MSTracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW(mpMap);

    // We perform first an matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    Matcher matcher(mpMap->mpCamera, 0.7);
    vector<MapPoint *> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
    {
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    Optimizer::PoseOptimization(&mCurrentFrame);

    // std::cerr<< "TrackReferenceKeyFrame " <<mCurrentFrame.mvKeysUn.size()<<" "<<nmatches<<" "<<nOptimize<<std::endl;
    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
            {
                nmatchesMap++;
                mCurrentFrame.mvpMapPoints[i]->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                mCurrentFrame.mvpMapPoints[i]->mnTrackedbyFrame = mCurrentFrame.mnId;
            }
        }
    }
    return true;
}

bool MSTracking::TrackWithMotionModel()
{
    Matcher matcher(mpMap->mpCamera, 0.9);
    mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());

    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

    // Project points seen in previous frame
    int th = 15;

    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th);

    // If few matches, uses a wider window search
    if (nmatches < 20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th);
    }

    if (nmatches < 20)
    {
        std::cerr<<"Not enough matches!!"<<std::endl;
        return true;
    }

    // Optimize frame pose with all matches
    int nOptimize = Optimizer::PoseOptimization(&mCurrentFrame);

    // std::cerr<< "TrackMotionModel " <<mCurrentFrame.mvKeysUn.size()<<" "<<nmatches<<" "<<nOptimize<<std::endl;
    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
            {
                nmatchesMap++;
                mCurrentFrame.mvpMapPoints[i]->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                mCurrentFrame.mvpMapPoints[i]->mnTrackedbyFrame = mCurrentFrame.mnId;
            }
        }
    }
    return true;
}

bool MSTracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    UpdateLocalMap();
    SearchLocalPoints();

    if (!mpMap->isImuInitialized())
        Optimizer::PoseOptimization(&mCurrentFrame);
    else
    {
        if (mCurrentFrame.mnId <= mnLastRelocFrameId + mpCamera->mfFps)
            Optimizer::PoseOptimization(&mCurrentFrame);
        else
        {
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
            if (!mbMapUpdated) //  && (mnMatchesInliers>30))
                Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame, mpMap);
            else
                Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame, mpMap);
        }
    }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    mnMatchesInliers++;
            }
        }
    }

    // std::cerr<<"track local map "<<mCurrentFrame.mvKeysUn.size()<<" "<<mnMatchesInliers<<std::endl;
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mpCamera->mfFps && mnMatchesInliers < 20)
        return false;

    if ((mnMatchesInliers > 10) && (mState == RECENTLY_LOST))
        return true;

    if ((mnMatchesInliers < 5 && mpMap->isImuInitialized()) || (mnMatchesInliers < 20 && !mpMap->isImuInitialized()))
        return false;
    else
        return true;
}

bool MSTracking::NeedNewKeyFrame()
{
    if(MSLocalMapping::get().CheckNewKeyFrames() || !MSLocalMapping::get().mbLocalMappingIdle)
        return false;

    if (!mpMap->isImuInitialized())
    {
        if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.1)
            return true;
        else
            return false;
    }

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (MSLocalMapping::get().isStopped() || MSLocalMapping::get().stopRequested())
        return false;

    if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.1)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (MSLocalMapping::get().mbLocalMappingIdle)
            return true;
        else
            return false;
    }
    else
        return false;
}

void MSTracking::CreateNewKeyFrame()
{

    KeyFrame *pNewKF = mCurrentFrame.buildKeyFrame(mpMap); 

    if (mpMap->isImuInitialized())
        pNewKF->bImu = true;

    pNewKF->SetNewBias(mCurrentFrame.mImuBias);
    mpReferenceKF = pNewKF;
    mCurrentFrame.mpReferenceKF = pNewKF;

    if (mpLastKeyFrame)
    {
        pNewKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pNewKF;
    }
    mpMap->IncreMap(pNewKF);

    MSLocalMapping::get().InsertKeyFrame(pNewKF);
    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pNewKF->GetImuBias(), mpImuCalib);
    mpLastKeyFrame = pNewKF;
}

void MSTracking::SearchLocalPoints()
{
    // Project points in frame and check its visibility
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        mCurrentFrame.CheckInFrustum(pMP, 0.5);
    }
    
    Matcher matcher(mpMap->mpCamera, 0.8);
    int th = 10;
    if (mpMap->isImuInitialized())
    {
        if (mpMap->GetInertialBA())
            th = 3;
        else
            th = 6;
    }
    if (mCurrentFrame.mnId < mnLastRelocFrameId + 2) // relocalised recently
        th = 5;
    if (mState == LOST || mState == RECENTLY_LOST) // Lost for less than 1 second
        th = 15;

    matcher.ExtendMapMatches(mCurrentFrame, mvpLocalMapPoints, th);
}

void MSTracking::UpdateLocalMap()
{
    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void MSTracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    for (vector<KeyFrame *>::const_reverse_iterator itKF = mvpLocalKeyFrames.rbegin(), itEndKF = mvpLocalKeyFrames.rend(); itKF != itEndKF; ++itKF)
    {
        KeyFrame *pKF = *itKF;
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {

            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if (!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

void MSTracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame *, int> keyframeCounter;
    if (!mpMap->isImuInitialized() || (mCurrentFrame.mnId < mnLastRelocFrameId + 2))
    {
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP)
            {
                if (!pMP->isBad())
                {
                    const map<KeyFrame *, int> observations = pMP->GetObservations();
                    for (map<KeyFrame *, int>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                    mCurrentFrame.mvpMapPoints[i] = NULL;
            }
        }
    }
    else
    {
        for (int i = 0; i < mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            if (mLastFrame.mvpMapPoints[i])
            {
                MapPoint *pMP = mLastFrame.mvpMapPoints[i];
                if (!pMP)
                    continue;
                if (!pMP->isBad())
                {
                    const map<KeyFrame *, int> observations = pMP->GetObservations();
                    for (map<KeyFrame *, int>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                    mLastFrame.mvpMapPoints[i] = NULL; // MODIFICATION
            }
        }
    }

    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        KeyFrame *pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max)
        {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80) // 80
            break;

        KeyFrame *pKF = *itKF;

        const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if (mvpLocalKeyFrames.size() < 80)
    {
        KeyFrame *tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for (int i = 0; i < Nd; i++)
        {
            if (!tempKeyFrame)
                break;
            if (tempKeyFrame->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                tempKeyFrame = tempKeyFrame->mPrevKF;
            }
        }
    }

    if (pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool MSTracking::Relocalization()
{
    std::cout<<"Starting relocalization"<<std::endl;
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW(mpMap);

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame *> vpCandidateKFs = mpMap->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
    {
        std::cout<<"There are not candidates"<<std::endl;
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an matching with each candidate
    // If enough matches are found we setup a PnP solver
    Matcher matcher(mpMap->mpCamera, 0.75);

    vector<MLPnPsolver *> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint *>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i < nKFs; i++)
    {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                MLPnPsolver *pSolver = new MLPnPsolver(mCurrentFrame, mpCamera, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 6, 0.5, 5.991); // This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    Matcher matcher2(mpMap->mpCamera, 0.9);

    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver *pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers, eigTcw);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (bTcw)
            {
                SE3f Tcw(eigTcw);
                mCurrentFrame.SetPose(Tcw);
                // Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint *> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if (nGood < 10)
                    continue;

                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 0.5);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if (nGood >= 50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch)
        return false;
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }
}

void MSTracking::Reset()
{
    std::cerr<<"System Reseting" <<std::endl;
    // Reset Local Mapping
    MSLocalMapping::get().RequestReset();

    // Reset Loop Closing
    MSLoopClosing::get().RequestReset();
    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    mnLastRelocFrameId = 0;
    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;
    mbReadyToInitializate = false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame *>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
    mvIniMatches.clear();
    std::cerr<<"   End reseting! "<<std::endl;
}

KeyFrame* MSTracking::GetLastKeyFrame()
{
    return mpLastKeyFrame;
}

void MSTracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame)
{
    list<KeyFrame *>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for (auto lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lbL++)
    {
        if (*lbL)
            continue;
        (*lit).translation() *= s;
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while (!mCurrentFrame.mbImuPreintegrated)
    {
        usleep(500);
    }

    if (mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz * t12 + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                         twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                         Vwb1 + Gz * t12 + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }
}

int MSTracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

void MSTracking::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    if(mpMap->KeyFramesInMap()<10)
        return;

    // while(MSLocalMapping::get().CheckNewKeyFrames() || !MSLocalMapping::get().mbLocalMappingIdle)
    //     usleep(500);

    // Retrieve all keyframe in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpLastKeyFrame;
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
    if(mpLastKeyFrame->mTimeStamp-mFirstTs< 2.0)
        return;

    MSLocalMapping::get().RequestStop();
    while(!MSLocalMapping::get().isStopped())
        usleep(500);

    bInitializing = true;

    std::cerr<< " initialize imu"<<std::endl;

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
        
        // Safety check for vzg before calling SO3::exp
        if (!vzg.allFinite() || nv < 1e-8) {
            std::cout << "Warning: Invalid rotation vector in Tracking (nv=" << nv << ")" << std::endl;
            Rwg = Eigen::Matrix3f::Identity();
        } else {
            try {
                Rwg = SO3f::exp(vzg).matrix();
            } catch (const std::exception& e) {
                std::cout << "Warning: SO3::exp failed in Tracking: " << e.what() << std::endl;
                Rwg = Eigen::Matrix3f::Identity();
            }
        }
        mRwg = Rwg.cast<double>();
        mFirstTs = mpLastKeyFrame->mTimeStamp-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = mpLastKeyFrame->GetGyroBias().cast<double>();
        mba = mpLastKeyFrame->GetAccBias().cast<double>();
    }

    mScale=1.0;

    Optimizer::InertialOptimization(mpMap, mRwg, mScale, mbg, mba, false, priorG, priorA);

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
            SE3f Twg(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());
            mpMap->ApplyScaledRotation(Twg, mScale, true);
            UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpLastKeyFrame);
        }

        // Check if initialization OK
        if (!mpMap->isImuInitialized())
            for (int i = 0; i < N; i++) {
                KeyFrame *pKF2 = vpKF[i];
                pKF2->bImu = true;
            }
    }

    UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpLastKeyFrame);
    if (!mpMap->isImuInitialized())
    {
        mpMap->SetImuInitialized();
        mpLastKeyFrame->bImu = true;
    }

    if (bFIBA)
    {
        if (priorA!=0.f)
            Optimizer::FullInertialBA(mpMap, 100, mpLastKeyFrame->mnId, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpMap, 100, mpLastKeyFrame->mnId, NULL, false);
    }

    std::cout << "Global Bundle Adjustment finished" << std::endl << "Updating map ..."<< std::endl;

    // Get Map Mutex
    // unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    unsigned long GBAid = mpLastKeyFrame->mnId;

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
        if(!pME || pME->isBad())
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

    mState=OK;
    bInitializing = false;

    mpMap->InfoMapChange();

    MSLocalMapping::get().Release();
    return;
}

void MSTracking::ScaleRefinement()
{
    // while(MSLocalMapping::get().CheckNewKeyFrames() || !MSLocalMapping::get().mbLocalMappingIdle)
    //     usleep(500);

    // Retrieve all keyframes in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpLastKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;

    Optimizer::InertialOptimization(mpMap, mRwg, mScale);

    if (mScale<1e-1) // 1e-1
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }
    
    SO3d so3wg(mRwg);
    // Before this line we are not changing the map
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    if ((fabs(mScale-1.f)>0.002))
    {
        SE3f Tgw(mRwg.cast<float>().transpose(),Eigen::Vector3f::Zero());
        mpMap->ApplyScaledRotation(Tgw,mScale,true);
        UpdateFrameIMU(mScale,mpLastKeyFrame->GetImuBias(),mpLastKeyFrame);
    }

    // To perform pose-inertial opt w.r.t. last keyframe
    mpMap->InfoMapChange();

    return;
}
