/**
 * @file Frame.cpp
 * @brief Frame class implementation
 */

#include <thread>
#include "Frame.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "PPGExtractor.h"
#include "KannalaBrandt8.h"
#include "Pinhole.h"
#include "Map.h"

// ==================== STATIC MEMBERS ====================

long unsigned int Frame::nNextId = 0;

// ==================== CONSTRUCTORS ====================

Frame::Frame(): 
    mnId(0), mTimeStamp(0.0), N(0), mbHasPose(false), mVw(Eigen::Vector3f::Zero()), mbHasVelocity(false), mImuBias(), 
    mbImuPreintegrated(false), mpcpi(nullptr), mpExtractor(nullptr), mpImuPreintegrated(nullptr), mpImuPreintegratedFrame(nullptr), 
    mpPrevFrame(nullptr), mpLastKeyFrame(nullptr), mpReferenceKF(nullptr), mpCamera(nullptr), mpImuCalib(nullptr)
{}

Frame::Frame(const Frame &frame) : 
    mnId(frame.mnId), mTimeStamp(frame.mTimeStamp), N(frame.N), mvKeys(frame.mvKeys), mvKeysUn(frame.mvKeysUn), mvKeyEdges(frame.mvKeyEdges),
    mDescriptors(frame.mDescriptors.clone()), mvpMapPoints(frame.mvpMapPoints), mvpMapEdges(frame.mvpMapEdges), mvbOutlier(frame.mvbOutlier), 
    mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec), mTcw(frame.mTcw), mRwc(frame.mRwc), mOw(frame.mOw), mRcw(frame.mRcw), mtcw(frame.mtcw),
    mbHasPose(false), mVw(frame.mVw), mbHasVelocity(false), mImuBias(frame.mImuBias), mbImuPreintegrated(frame.mbImuPreintegrated), mpcpi(frame.mpcpi), 
    mpExtractor(frame.mpExtractor), mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame),
    mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame), mpReferenceKF(frame.mpReferenceKF), mpCamera(frame.mpCamera), 
    mpImuCalib(frame.mpImuCalib)
{
    srcMat = frame.srcMat.clone();
    
    // Copy grid
    for(int i = 0; i < GeometricCamera::FRAME_GRID_COLS; i++)
        for(int j = 0; j < GeometricCamera::FRAME_GRID_ROWS; j++)
            mGrid[i][j] = frame.mGrid[i][j];

    if(frame.mbHasPose)
        SetPose(frame.GetPose());

    if(frame.HasVelocity())
        SetVelocity(frame.GetVelocity());
}

Frame::Frame(const cv::Mat &imGray, const double &timeStamp, PPGExtractor* pExt, GeometricCamera* pCam, IMU::Calib *pImu, Frame* pPrevF)
    : mnId(nNextId++), mTimeStamp(timeStamp), N(0),
      mbHasPose(false), mVw(Eigen::Vector3f::Zero()), mbHasVelocity(false), 
      mImuBias(), mbImuPreintegrated(false),
      mpcpi(nullptr), mpExtractor(pExt), 
      mpImuPreintegrated(nullptr), mpImuPreintegratedFrame(nullptr),
      mpPrevFrame(pPrevF), mpLastKeyFrame(nullptr), mpReferenceKF(nullptr),
      mpCamera(pCam), mpImuCalib(pImu)
{
    srcMat = imGray.clone();

    // Extract features
    mpExtractor->run(imGray, mvKeys, mvKeysUn, mvKeyEdges, mDescriptors); 
    N = mvKeys.size();
    
    if(mvKeys.empty())
        return;

    // Initialize containers
    mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(nullptr));
    mvpMapEdges = vector<MapEdge*>(mvKeyEdges.size(), static_cast<MapEdge*>(nullptr));
    mvbOutlier = vector<bool>(N, false);

    AssignFeaturesToGrid();

    // Set velocity from previous frame
    if(pPrevF && pPrevF->HasVelocity())
        SetVelocity(pPrevF->GetVelocity());
}

// ==================== KEYFRAME CREATION ====================

KeyFrame* Frame::buildKeyFrame(Map* pMap)
{
    KeyFrame* ret = new KeyFrame();
    ret->bImu = pMap->isImuInitialized();
    ret->mnFrameId = mnId;  
    ret->mTimeStamp = mTimeStamp;

    ret->N = N;
    ret->mvKeys = mvKeys;
    ret->mvKeysUn = mvKeysUn;
    ret->mvKeyEdges = mvKeyEdges;
    ret->mDescriptors = mDescriptors.clone();
    ret->mBowVec = mBowVec;
    ret->mFeatVec = mFeatVec;
    ret->mpImuPreintegrated = mpImuPreintegrated;
    ret->mpImuCalib = mpImuCalib;
    ret->mpCamera = mpCamera;
    ret->mvpMapPoints = mvpMapPoints;
    ret->mvpMapEdges = mvpMapEdges;
    ret->srcMat = srcMat.clone();

    // Copy grid
    ret->mGrid.resize(GeometricCamera::FRAME_GRID_COLS);
    for(int i = 0; i < GeometricCamera::FRAME_GRID_COLS; i++)
    {
        ret->mGrid[i].resize(GeometricCamera::FRAME_GRID_ROWS);
        for(int j = 0; j < GeometricCamera::FRAME_GRID_ROWS; j++)
            ret->mGrid[i][j] = mGrid[i][j];
    }

    // Set velocity and bias
    if(HasVelocity()) 
    {
        ret->mVw = GetVelocity();
        ret->mbHasVelocity = true;
    }
    else
    {
        ret->mVw.setZero();
        ret->mbHasVelocity = false;
    }
    
    ret->mImuBias = mImuBias;
    ret->SetPose(GetPose());

    // Compute BoW
    vector<cv::Mat> vCurrentDesc(mDescriptors.rows, cv::Mat());
    for(int j = 0; j < mDescriptors.rows; j++)
        vCurrentDesc[j] = mDescriptors.row(j);
    pMap->mpVoc->transform(vCurrentDesc, ret->mBowVec, ret->mFeatVec, 4);

    return ret;
}

// ==================== GRID MANAGEMENT ====================

void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells = GeometricCamera::FRAME_GRID_COLS*GeometricCamera::FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);

    for(unsigned int i=0; i<GeometricCamera::FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<GeometricCamera::FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const KeyPointEx &kp = mvKeysUn[i];
        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

// ==================== POSE MANAGEMENT ====================

void Frame::SetPose(const Sophus::SE3<float> &Tcw) {
    mTcw = Tcw;
    UpdatePoseMatrices();
    mbHasPose = true;
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const
{
    return mVw;
}

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mpImuCalib->mTcb * Tbw;

    UpdatePoseMatrices();
    mbHasPose = true;
}

// ==================== INTERNAL POSE COMPUTATION ====================

void Frame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();
}

Eigen::Matrix<float,3,1> Frame::GetImuPosition() const {
    return mRwc * mpImuCalib->mTcb.translation() + mOw;
}

Eigen::Matrix<float,3,3> Frame::GetImuRotation() {
    return mRwc * mpImuCalib->mTcb.rotationMatrix();
}

Sophus::SE3<float> Frame::GetImuPose() {
    return mTcw.inverse() * mpImuCalib->mTcb;
}

// ==================== FEATURE MANAGEMENT ====================

void Frame::CheckInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;
    pMP->mTrackProjX = -1;
    pMP->mTrackProjY = -1;
    pMP->mTrackDepth = -1;
    // 3D in absolute coordinates
    Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();
    // 3D in camera coordinates
    const Eigen::Matrix<float,3,1> Pc = mRcw * P + mtcw;
    // Check positive depth
    if(Pc[2]<0.0f)
        return;
    // check if projected in image
    const Eigen::Vector2f uv = mpCamera->project(Pc);
    if(!mpCamera->IsInImage(uv(0), uv(1)))
        return;
    // Check distance to camera
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = P - mOw;
    const float dist = PO.norm();
    if(dist<minDistance || dist>maxDistance)
        return;
    // Check viewing angle
    Eigen::Vector3f Pn = pMP->GetNormal();
    const float viewCos = PO.dot(Pn)/dist;
    if(viewCos<viewingCosLimit)
        return;

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = uv(0);
    pMP->mTrackProjY = uv(1);
    pMP->mTrackDepth = dist;
    pMP->mTrackViewCos = viewCos;
    pMP->IncreaseVisible();
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mpCamera->mnMinX-factorX)*mpCamera->mfGridElementWidthInv));
    if(nMinCellX>=GeometricCamera::FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)GeometricCamera::FRAME_GRID_COLS-1,(int)ceil((x-mpCamera->mnMinX+factorX)*mpCamera->mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mpCamera->mnMinY-factorY)*mpCamera->mfGridElementHeightInv));
    if(nMinCellY>=GeometricCamera::FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)GeometricCamera::FRAME_GRID_ROWS-1,(int)ceil((y-mpCamera->mnMinY+factorY)*mpCamera->mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const KeyPointEx &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.mPos[0]-x;
                const float disty = kpUn.mPos[1]-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const KeyPointEx &kp, int &posX, int &posY)
{
    posX = round((kp.mPos[0]-mpCamera->mnMinX)*mpCamera->mfGridElementWidthInv);
    posY = round((kp.mPos[1]-mpCamera->mnMinY)*mpCamera->mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=GeometricCamera::FRAME_GRID_COLS || posY<0 || posY>=GeometricCamera::FRAME_GRID_ROWS)
        return false;

    return true;
}

// ==================== BAG OF WORDS ====================

void Frame::ComputeBoW(Map* pMap)
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc(mDescriptors.rows, cv::Mat());
        for (int j=0;j<mDescriptors.rows;j++)
            vCurrentDesc[j] = mDescriptors.row(j);
        pMap->mpVoc->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

Eigen::Vector2f Frame::ProjectPoint(MapPoint* pMP)
{
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();
    // 3D in camera coordinates
    Eigen::Vector3f Pc = mRcw * P + mtcw;
    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    uv = mpCamera->project(Pc);
    return uv;
}