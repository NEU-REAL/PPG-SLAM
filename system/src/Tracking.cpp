#include "Tracking.h"
#include "Matcher.h"
#include "G2oVertex.h"
#include "G2oEdge.h"
#include "Optimizer.h"
#include "GeometricCamera.h"
#include "MLPnPsolver.h"
#include "Frame.h"
#include "System.h"
#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"

#include <iostream>
#include <mutex>
#include <chrono>

using namespace std;

/**
 * @brief Initialize tracking system with map and neural network
 * Sets up feature extractor, IMU calibration, and initial state
 */
void MSTracking::Launch(Map* pMap, const string &strNet)
{
    // Initialize tracking state
    mState = NO_IMAGES_YET;
    mLastProcessedState = NO_IMAGES_YET;
    
    // Reset pointers
    mpExtractor = nullptr;
    mpReferenceKF = nullptr;
    mpLastKeyFrame = nullptr;
    
    // Set map and camera
    mpMap = pMap;
    mpCamera = pMap->mpCamera;
    mpImuCalib = pMap->mpImuCalib;
    
    // Initialize tracking flags
    mnLastRelocFrameId = 0;
    mTimeStampLost = 0;
    mbMapUpdated = false;
    mbReadyToInitializate = false;

    // Initialize IMU parameters
    mTinit = 0.f;

    // Create feature extractor and IMU preintegrator
    mpExtractor = new PPGExtractor(mpCamera, strNet);
    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), mpImuCalib);
}

/**
 * @brief Process monocular image and return camera pose
 * Main entry point for tracking pipeline
 */
cv::Mat MSTracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;
    mCurrentFrame = Frame(mImGray, timestamp, mpExtractor, mpCamera, mpImuCalib, &mLastFrame);
    Track();

    // draw frame
    cv::Mat showMat;
    cv::cvtColor(mImGray,showMat, cv::COLOR_GRAY2BGR);

    std::vector<KeyPointEx> &vCurrentKeys = mCurrentFrame.mvKeys;
    std::vector<KeyEdge> &vCurrentEdges = mCurrentFrame.mvKeyEdges;
    std::vector<MapPoint*> &vpMapPoints = mCurrentFrame.mvpMapPoints;
    for(KeyPointEx kp : vCurrentKeys)
    {
        for(std::pair<unsigned int, unsigned int> cpt : kp.mvColine)
        {
            cv::Point2f pt1,pt2;
            pt1 = cv::Point2f(vCurrentKeys[cpt.first].mPos[0],vCurrentKeys[cpt.first].mPos[1]);
            pt2 = cv::Point2f(vCurrentKeys[cpt.second].mPos[0],vCurrentKeys[cpt.second].mPos[1]);
            cv::line(showMat,pt1, pt2, cv::Scalar(20,20,255),2);
        }
    }
    for(KeyEdge ke : vCurrentEdges)
    {
        const KeyPointEx &kp1 = vCurrentKeys[ke.startIdx];
        const KeyPointEx &kp2 = vCurrentKeys[ke.endIdx];
        cv::line(showMat, cv::Point2f(kp1.mPos[0],kp1.mPos[1]), cv::Point2f(kp2.mPos[0],kp2.mPos[1]), cv::Scalar(0,255,0),1);
        cv::circle(showMat, cv::Point2f(kp1.mPos[0],kp1.mPos[1]), 3, cv::Scalar(0,255,0), -1);
        cv::circle(showMat, cv::Point2f(kp2.mPos[0],kp2.mPos[1]), 3, cv::Scalar(0,255,0), -1);
    }
    for(unsigned int i=0; i<vCurrentKeys.size(); i++)
    {
        MapPoint * pMP = vpMapPoints[i];
        if(pMP == nullptr || pMP->isBad())
            continue;
        cv::circle(showMat, cv::Point2f(vCurrentKeys[i].mPos[0],vCurrentKeys[i].mPos[1]), 3, cv::Scalar(125,255,0),1);
    }
    return showMat;
}

/**
 * @brief Add IMU measurement to processing queue
 * Thread-safe insertion of IMU data for preintegration
 */
void MSTracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

/**
 * @brief Preintegrate IMU measurements between consecutive frames
 * Extracts IMU data from queue and performs preintegration using trapezoidal rule
 */
void MSTracking::PreintegrateIMU()
{
    if (!mCurrentFrame.mpPrevFrame)
    {
        mCurrentFrame.mbImuPreintegrated = true;
        return;
    }
    
    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    
    if (mlQueueImuData.size() == 0)
    {
        mCurrentFrame.mbImuPreintegrated = true;
        return;
    }

    // Extract IMU measurements between frames
    while (true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if (!mlQueueImuData.empty())
            {
                IMU::Point *m = &mlQueueImuData.front();
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
        std::cerr << "Empty IMU measurements vector!" << std::endl;
        return;
    }

    // Perform IMU preintegration using trapezoidal rule
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
                   (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tini / tab)) * 0.5f;
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                      (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tini / tab)) * 0.5f;
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
                   (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tend / tab)) * 0.5f;
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                      (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tend / tab)) * 0.5f;
            tstep = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;
        }
        else if ((i == 0) && (i == (n - 1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        // Integrate measurements
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc, angVel, tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
    }
    
    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;
    mCurrentFrame.mbImuPreintegrated = true;
}

/**
 * @brief Predict camera pose using IMU measurements
 * Uses IMU preintegration to predict frame pose from last keyframe
 */
bool MSTracking::PredictStateIMU()
{
    if (!mCurrentFrame.mpPrevFrame)
    {
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
    
    return false;
}

/**
 * @brief Reset frame IMU information
 * Placeholder for IMU frame reset functionality
 */
void MSTracking::ResetFrameIMU()
{
    // Reset IMU frame state (implementation placeholder)
}

/**
 * @brief Main tracking function implementing the SLAM pipeline
 * Handles initialization, tracking, and keyframe creation
 */
void MSTracking::Track()
{
    // Validate timestamp consistency
    if (mState != NO_IMAGES_YET && (mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp || 
        mCurrentFrame.mTimeStamp > mLastFrame.mTimeStamp + 1.0))
    {
        cerr << "ERROR: Timestamp inconsistency detected!" << endl;
        unique_lock<mutex> lock(mMutexImuQueue);
        mlQueueImuData.clear();
        Reset();
        return;
    }
    
    // Set IMU bias and state
    if (mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());
    
    if (mState == NO_IMAGES_YET)
        mState = NOT_INITIALIZED;
    
    mLastProcessedState = mState;
    PreintegrateIMU();

    // Monocular initialization
    if (mState == NOT_INITIALIZED)
    {
        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        MonocularInitialization();
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

    // Initialize IMU here
    if(!mpMap->isImuInitialized())
        InitializeIMU(1e2, 1e10, true);
    else 
    {
        // Update time for IMU initialization if IMU is already initialized
        if(mpLastKeyFrame && mpLastKeyFrame->mPrevKF)
        {
            float dist = (mpLastKeyFrame->mPrevKF->GetCameraCenter() - mpLastKeyFrame->GetCameraCenter()).norm();
            if(dist>0.05)
                mTinit += mpLastKeyFrame->mTimeStamp - mpLastKeyFrame->mPrevKF->mTimeStamp;
        }
        
        // Check IMU initialization timing criteria
        if (!mpMap->GetInertialBA() && mTinit > Map::imuIniTm)
        {
            std::cout << "Starting visual inertial BA" << std::endl;
            mpMap->SetInertialBA();
            InitializeIMU(1.f, 1e5, true);
            std::cout << "Visual inertial BA completed" << std::endl;
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
            std::cout << "Failed to track local map! Using IMU-only tracking..." << std::endl;
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
                        std::cout << "Resetting frame after relocalization" << std::endl;
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
                            mCurrentFrame.mvpMapPoints[i] = nullptr;
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
                // Clear outlier map points from frame
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = nullptr;
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

/**
 * @brief Initialize monocular SLAM from two frames
 * Attempts to reconstruct initial map from first two frames with sufficient features
 */
void MSTracking::MonocularInitialization()
{
    if (!mbReadyToInitializate)
    {
        // Initialize first frame if enough features
        if (mCurrentFrame.mvKeys.size() > 50)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = cv::Point2f(mCurrentFrame.mvKeysUn[i].mPos[0], mCurrentFrame.mvKeysUn[i].mPos[1]);

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            // Reset IMU preintegration
            if (mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            
            mbReadyToInitializate = true;
        }
        return;
    }
    
    // Check second frame validity
    if ((int)mCurrentFrame.mvKeys.size() <= 50 || (mLastFrame.mTimeStamp - mInitialFrame.mTimeStamp) > 1.0)
    {
        mbReadyToInitializate = false;
        return;
    }

    // Find correspondences between frames
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

/**
 * @brief Create initial map from successful two-frame initialization
 * Builds first two keyframes and triangulated map points
 */
void MSTracking::CreateInitialMapMonocular()
{
    // Create initial keyframes
    KeyFrame *pKFini = mInitialFrame.buildKeyFrame(mpMap);
    KeyFrame *pKFcur = mCurrentFrame.buildKeyFrame(mpMap);
    pKFini->mpImuPreintegrated = (IMU::Preintegrated *)(NULL);
    
    // Insert keyframes into map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);
    
    // Create map points from triangulated features
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
            continue;
            
        // Create MapPoint from triangulated 3D position
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint *pMP = new MapPoint(worldPos, pKFcur);
        
        // Associate map point with keyframes
        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);
        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);
        
        // Update map point properties
        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();
        
        // Update current frame
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;
        
        // Add to global map
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
    std::cout << "New map created with " << mpMap->MapPointsInMap() << " points" << std::endl;
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

    // Check initialization quality thresholds
    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50)
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

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mState = OK;
}

/**
 * @brief Update map points in last frame that have been replaced
 * Handles map point replacement during local mapping optimization
 */
void MSTracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (!pMP)
            continue;
            
        // Replace with updated map point if available
        if (pMP->mpReplaced)
            mLastFrame.mvpMapPoints[i] = pMP->mpReplaced;
    }
}

/**
 * @brief Track current frame against reference keyframe
 * Uses BoW matching followed by PnP pose optimization
 */
bool MSTracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words for feature matching
    mCurrentFrame.ComputeBoW(mpMap);

    // Match features with reference keyframe using BoW
    Matcher matcher(mpMap->mpCamera, 0.7);
    vector<MapPoint *> vpMapPointMatches;
    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
    {
        return false;
    }

    // Set matched map points and initial pose
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    // Optimize pose using PnP
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Remove outliers and count valid matches
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i] = nullptr;
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                nmatches--;
            }
        }
    }
    
    return nmatchesMap >= 10;
}

/**
 * @brief Track current frame using constant velocity motion model
 * Predicts pose using velocity and matches features by projection
 */
bool MSTracking::TrackWithMotionModel()
{
    Matcher matcher(mpMap->mpCamera, 0.9);
    
    // Predict current pose using velocity model
    mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);

    // Match map points by projection
    int th = 15;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th);

    // Use wider search if few matches found
    if (nmatches < 20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th);
    }

    if (nmatches < 20)
        return false;

    // Optimize pose using matched points
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Remove outliers and count valid matches
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i] = nullptr;
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
    
    return nmatchesMap >= 10;
}

/**
 * @brief Track current frame against local map
 * Updates local map and searches for additional matches
 */
bool MSTracking::TrackLocalMap()
{
    // Update local map and find additional matches
    UpdateLocalMap();
    SearchLocalPoints();

    // Optimize pose (different approach for IMU vs visual-only)
    if (!mpMap->isImuInitialized())
        Optimizer::PoseOptimization(&mCurrentFrame);
    else
    {
        // IMU-based optimization for post-initialization frames
        if (mCurrentFrame.mnId <= mnLastRelocFrameId + mpCamera->mfFps)
            Optimizer::PoseOptimization(&mCurrentFrame);
        else
        {
            if (!mbMapUpdated)
                Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame, mpMap);
            else
                Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame, mpMap);
        }
    }

    mnMatchesInliers = 0;

    // Count inlier matches and update map point statistics
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

    // Check tracking quality thresholds
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mpCamera->mfFps && mnMatchesInliers < 20)
        return false;

    if ((mnMatchesInliers > 10) && (mState == RECENTLY_LOST))
        return true;

    if ((mnMatchesInliers < 5 && mpMap->isImuInitialized()) || 
        (mnMatchesInliers < 20 && !mpMap->isImuInitialized()))
        return false;
    else
        return true;
}

/**
 * @brief Determine if new keyframe is needed based on tracking quality
 * Considers timing, tracking quality, and local mapping availability
 */
bool MSTracking::NeedNewKeyFrame()
{
    // Don't insert keyframes if local mapping is busy
    if (MSLocalMapping::get().CheckNewKeyFrames() || !MSLocalMapping::get().mbLocalMappingIdle)
        return false;

    // Simple timing check for non-IMU systems
    if (!mpMap->isImuInitialized())
    {
        return (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.1;
    }

    // Don't insert if local mapping is stopped
    if (MSLocalMapping::get().isStopped() || MSLocalMapping::get().stopRequested())
        return false;

    // Time-based keyframe insertion for IMU systems
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

/**
 * @brief Create new keyframe from current frame
 * Handles IMU preintegration and inserts into local mapping
 */
void MSTracking::CreateNewKeyFrame()
{
    // Create keyframe from current frame
    KeyFrame *pNewKF = mCurrentFrame.buildKeyFrame(mpMap); 

    // Set IMU status and bias
    if (mpMap->isImuInitialized())
        pNewKF->bImu = true;
    pNewKF->SetNewBias(mCurrentFrame.mImuBias);
    
    // Update reference keyframe
    mpReferenceKF = pNewKF;
    mCurrentFrame.mpReferenceKF = pNewKF;

    // Handle IMU preintegration chain
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

/**
 * @brief Search for local map points in current frame
 * Projects local map points and matches with current frame features
 */
void MSTracking::SearchLocalPoints()
{
    // Project local map points into current frame
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP->isBad())
            continue;
            
        // Check if point is in current frame frustum
        mCurrentFrame.CheckInFrustum(pMP, 0.5);
    }
    
    // Match local points with frame features
    Matcher matcher(mpMap->mpCamera, 0.8);
    int th = 10;  // Default search radius
    
    // Adjust search radius based on IMU initialization and BA state
    if (mpMap->isImuInitialized())
    {
        th = mpMap->GetInertialBA() ? 3 : 6;
    }
    
    // Adjust for recent relocalization or tracking state
    if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
        th = 5;
    if (mState == LOST || mState == RECENTLY_LOST)
        th = 15;

    matcher.ExtendMapMatches(mCurrentFrame, mvpLocalMapPoints, th);
}

/**
 * @brief Update local map (keyframes and points)
 * Refreshes local keyframes and associated map points
 */
void MSTracking::UpdateLocalMap()
{
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

/**
 * @brief Update local map points from local keyframes
 * Collects all map points observed by local keyframes
 */
void MSTracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    // Collect map points from all local keyframes
    for (vector<KeyFrame *>::const_reverse_iterator itKF = mvpLocalKeyFrames.rbegin(), 
         itEndKF = mvpLocalKeyFrames.rend(); itKF != itEndKF; ++itKF)
    {
        KeyFrame *pKF = *itKF;
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), 
             itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            MapPoint *pMP = *itMP;
            if (!pMP || pMP->isBad())
                continue;
                
            // Avoid duplicates
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
                
            mvpLocalMapPoints.push_back(pMP);
            pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }
    }
}

/**
 * @brief Update local keyframes for tracking
 * Selects keyframes observing current frame map points plus neighbors
 */
void MSTracking::UpdateLocalKeyFrames()
{
    // Count observations of current frame map points by keyframes
    map<KeyFrame *, int> keyframeCounter;
    
    if (!mpMap->isImuInitialized() || (mCurrentFrame.mnId < mnLastRelocFrameId + 2))
    {
        // Use current frame map points
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP && !pMP->isBad())
            {
                const map<KeyFrame *, int> observations = pMP->GetObservations();
                for (map<KeyFrame *, int>::const_iterator it = observations.begin(), 
                     itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
            else if (pMP)
            {
                mCurrentFrame.mvpMapPoints[i] = NULL;
            }
        }
    }
    else
    {
        // Use last frame map points (current frame not matched yet)
        for (int i = 0; i < mLastFrame.N; i++)
        {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (pMP && !pMP->isBad())
            {
                const map<KeyFrame *, int> observations = pMP->GetObservations();
                for (map<KeyFrame *, int>::const_iterator it = observations.begin(), 
                     itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
            else if (pMP)
            {
                mLastFrame.mvpMapPoints[i] = NULL;
            }
        }
    }

    // Find keyframe with most shared map points
    int max = 0;
    KeyFrame *pKFmax = nullptr;

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // Add all keyframes that observe current map points
    for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), 
         itEnd = keyframeCounter.end(); it != itEnd; it++)
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

    // Add covisible neighbors of selected keyframes
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), 
         itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit total number of local keyframes
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame *pKF = *itKF;
        const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), 
             itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad() && pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pNeighKF);
                pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }

    // Add recent temporal keyframes (important for IMU)
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
            }
            tempKeyFrame = tempKeyFrame->mPrevKF;
        }
    }

    if (pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

/**
 * @brief Relocalize lost tracking using keyframe database
 * Matches current frame with candidate keyframes and estimates pose
 */
bool MSTracking::Relocalization()
{
    std::cout << "Starting relocalization" << std::endl;
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW(mpMap);

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame *> vpCandidateKFs = mpMap->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
    {
        std::cout << "No relocalization candidates found" << std::endl;
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
                        mCurrentFrame.mvpMapPoints[io] = nullptr;

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
                // If pose is supported by enough inliers, relocalization succeeds
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
        std::cout << "Relocalization successful!" << std::endl;
        return true;
    }
}

/**
 * @brief Reset tracking system to initial state
 * Clears all tracking data and resets mapping/loop closing
 */
void MSTracking::Reset()
{
    std::cerr << "System resetting..." << std::endl;
    
    // Reset subsystems
    MSLocalMapping::get().RequestReset();
    MSLoopClosing::get().RequestReset();
    
    // Clear map data
    mpMap->clear();

    // Reset tracking state
    mnLastRelocFrameId = 0;
    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;
    mbReadyToInitializate = false;
    mTinit = 0.0f;  // Reset IMU initialization timer

    // Clear tracking history
    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    
    // Reset frame and keyframe references
    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = nullptr;
    mpLastKeyFrame = nullptr;
    mvIniMatches.clear();
    
    std::cerr << "Reset complete!" << std::endl;
}

/**
 * @brief Get last created keyframe
 * @return Pointer to last keyframe
 */
KeyFrame* MSTracking::GetLastKeyFrame()
{
    return mpLastKeyFrame;
}

/**
 * @brief Update frame poses after IMU initialization with scale correction
 * @param s Scale factor for translation correction
 * @param b Updated IMU bias
 * @param pCurrentKeyFrame Reference keyframe for update
 */
void MSTracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame)
{
    // Scale all relative frame poses by scale factor
    list<KeyFrame *>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for (auto lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end(); 
         lit != lend; lit++, lRit++, lbL++)
    {
        if (*lbL)
            continue;
        (*lit).translation() *= s;
    }

    // Update bias and keyframe reference
    mLastBias = b;
    mpLastKeyFrame = pCurrentKeyFrame;
    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    // Wait for IMU preintegration to complete
    while (!mCurrentFrame.mbImuPreintegrated)
    {
        usleep(500);
    }

    // Update last frame IMU state
    if (mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        // Predict from keyframe to last frame
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(
            IMU::NormalizeRotation(Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
            twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
            Vwb1 + Gz * t12 + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    // Update current frame IMU state if preintegration available
    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(
            IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
            twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
            Vwb1 + Gz * t12 + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }
}

/**
 * @brief Get number of inlier matches in current frame
 * @return Number of inlier matches
 */
int MSTracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

/**
 * @brief Initialize IMU with prior noise parameters
 * @param priorG Gyroscope noise prior
 * @param priorA Accelerometer noise prior  
 * @param bFIBA Whether to perform full inertial BA
 */
void MSTracking::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    if(mpMap->KeyFramesInMap()<10)
        return;

    // while(MSLocalMapping::get().CheckNewKeyFrames() || !MSLocalMapping::get().mbLocalMappingIdle)
    //     usleep(500);

    // Retrieve all keyframe in temporal order
    // Collect keyframes chronologically for IMU initialization
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpLastKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    // Need at least 10 keyframes for reliable initialization
    if(vpKF.size() < 10)
        return;

    if(mpLastKeyFrame->mTimeStamp - vpKF.front()->mTimeStamp < 2.0)
        return;

    // Stop local mapping during initialization
    MSLocalMapping::get().RequestStop();
    while(!MSLocalMapping::get().isStopped())
        usleep(500);

    std::cerr << "Initializing IMU..." << std::endl;

    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0);
    Eigen::Matrix3d tmpRwg;  // Gravity direction

    // Compute gravity direction and keyframe velocities
    if (!mpMap->isImuInitialized())
    {
        Eigen::Matrix3f Rwg;
        Eigen::Vector3f dirG;
        dirG.setZero();
        
        for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF != vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated || !(*itKF)->mPrevKF)
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
        tmpRwg = Rwg.cast<double>();
    }
    else
    {
        tmpRwg = Eigen::Matrix3d::Identity();
    }

    double tmpScale = 1.0;
    Eigen::Vector3d tmpBg = mpLastKeyFrame->GetGyroBias().cast<double>();
    Eigen::Vector3d tmpBa = mpLastKeyFrame->GetAccBias().cast<double>();

    Optimizer::InertialOptimization(mpMap, tmpRwg, tmpScale, tmpBg, tmpBa, false, priorG, priorA);

    if (tmpScale<1e-1)
    {
        std::cout << "Scale too small during initialization" << std::endl;
        return;
    }

    // Before this line we are not changing the map
    {
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        if ((fabs(tmpScale - 1.f) > 0.00001)) {
            SE3f Twg(tmpRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());
            mpMap->ApplyScaledRotation(Twg, tmpScale, true);
            UpdateFrameIMU(tmpScale, vpKF[0]->GetImuBias(), mpLastKeyFrame);
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

    std::cout << "Map updated successfully!" << std::endl;

    mState=OK;

    mpMap->InfoMapChange();

    MSLocalMapping::get().Release();
    return;
}

/**
 * @brief Refine scale and gravity estimation using inertial optimization
 * Performs scale and gravity refinement for visual-inertial initialization
 */
void MSTracking::ScaleRefinement()
{
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

    // Initialize scale and gravity rotation
    Eigen::Matrix3d tmpRwg = Eigen::Matrix3d::Identity();
    double tmpScale = 1.0;

    // Perform inertial optimization to refine scale and gravity
    Optimizer::InertialOptimization(mpMap, tmpRwg, tmpScale);

    // Check if scale is reasonable
    if (tmpScale < 1e-1)
    {
        std::cout << "Scale too small, initialization failed" << std::endl;
        return;
    }
    
    SO3d so3wg(tmpRwg);
    // Before this line we are not changing the map
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    if ((fabs(tmpScale-1.f)>0.002))
    {
        SE3f Tgw(tmpRwg.cast<float>().transpose(),Eigen::Vector3f::Zero());
        mpMap->ApplyScaledRotation(Tgw,tmpScale,true);
        UpdateFrameIMU(tmpScale,mpLastKeyFrame->GetImuBias(),mpLastKeyFrame);
    }

    // To perform pose-inertial opt w.r.t. last keyframe
    mpMap->InfoMapChange();

    return;
}
