#include <limits.h>
#include <opencv2/core/core.hpp>
#include "DBoW3/DBoW3.h"
#include <stdint-gcc.h>

#include "Matcher.h"
#include "KannalaBrandt8.h"
#include "Pinhole.h"

using namespace std;

const float Matcher::TH_HIGH = 0.8;
const float Matcher::TH_LOW = 0.7;

Matcher::Matcher(GeometricCamera* pCam, float nnratio) : mpCamera(pCam), mfNNratio(nnratio)
{
}

int Matcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th)
{
    int nmatches = 0;

    const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    const Eigen::Vector3f twc = Tcw.inverse().translation();
    const Sophus::SE3f Tlw = LastFrame.GetPose();

    for (int i = 0; i < LastFrame.N; i++)
    {
        MapPoint *pMP = LastFrame.mvpMapPoints[i];
        if (pMP)
        {
            if (!LastFrame.mvbOutlier[i])
            {
                // Project
                Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                Eigen::Vector3f x3Dc = Tcw * x3Dw;
                const float invzc = 1.0 / x3Dc(2);

                if (invzc < 0)
                    continue;

                Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                if (uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX)
                    continue;
                if (uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
                    continue;

                vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), th);
                if (vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();
                float bestDist = 1e6;
                int bestIdx2 = -1;
                for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                {
                    const size_t i2 = *vit;
                    if (CurrentFrame.mvpMapPoints[i2])
                        if (CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
                            continue;
                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);
                    const float dist = DescriptorDistance(dMP, d);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }
                if (bestDist <= TH_HIGH)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                    nmatches++;
                }
            }
        }
    }
    return nmatches;
}

int Matcher::SearchByProjection(Frame &F, const vector<MapPoint *> &vpMapPoints, const float th)
{

    std::vector<MapPoint*> candidateMapPoints;
    candidateMapPoints.reserve(vpMapPoints.size());

    for (MapPoint *pMP  : vpMapPoints)
    {
        if (pMP->isBad())
            continue;
        if (!pMP->mbTrackInView)
            continue;
        candidateMapPoints.push_back(pMP);
    }
    
    int nmatches = 0;

    const bool bFactor = th != 1.0;

    for (size_t iMP = 0; iMP < candidateMapPoints.size(); iMP++)
    {
        MapPoint *pMP = candidateMapPoints[iMP];

        if(pMP->mnTrackedbyFrame == F.mnId)
            continue; // already matched in this frame

        if (pMP->isBad())
            continue;

        if (pMP->mbTrackInView)
        {
            // The size of the window will depend on the viewing direction
            float r(4.0f);
            
            if (pMP->mTrackViewCos > 0.998)
                r = 2.5;

            if (bFactor)
                r *= th;

            const vector<size_t> vIndices = F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r);

            if (!vIndices.empty())
            {
                const cv::Mat MPdescriptor = pMP->GetDescriptor();

                float bestDist = 1e6;
                float bestDist2 = 1e6;
                int bestIdx = -1;

                // Get best and second matches with near keypoints
                for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
                {
                    const size_t idx = *vit;

                    if (F.mvpMapPoints[idx])
                        if (F.mvpMapPoints[idx]->Observations() > 0)
                            continue;

                    const cv::Mat &d = F.mDescriptors.row(idx);

                    const float dist = DescriptorDistance(MPdescriptor, d);

                    if (dist < bestDist)
                    {
                        bestDist2 = bestDist;
                        bestDist = dist;
                        bestIdx = idx;
                    }
                    else if (dist < bestDist2)
                        bestDist2 = dist;
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if (bestDist <= TH_HIGH)
                {
                    if (bestDist > mfNNratio * bestDist2)
                        continue;

                    F.mvpMapPoints[bestIdx] = pMP;
                    nmatches++;
                }
            }
        }
    }
    return nmatches;
}



int Matcher::ExtendMapMatches(Frame &F, const vector<MapPoint *> &vpMapPoints, const float th)
{
    int nmatches = 0;
    std::vector<MapPoint*> candidateMapPoints;
    candidateMapPoints.reserve(vpMapPoints.size());

    for (MapPoint *pMP  : vpMapPoints)
    {
        if (pMP->isBad())
            continue;
        if (!pMP->mbTrackInView)
            continue;
        candidateMapPoints.push_back(pMP);
    }

    std::sort(candidateMapPoints.begin(), candidateMapPoints.end(), 
        [](MapPoint *a, MapPoint *b) {return a->getEdges().size() > b->getEdges().size();});

    for (MapPoint *pMP  : candidateMapPoints)
    {
        if(pMP->mnTrackedbyFrame == F.mnId)
            continue; // already matched in this frame
        if(pMP->isBad())
            continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor();
        float bestDist = 1e6;
        float bestDist2 = 1e6;
        int bestIdx = -1;
        // Get best and second matches with near keypoints
        float r = th;
        if (pMP->mTrackViewCos > 0.998)
            r *= 2.5;
        else
            r *= 4.0;
        const vector<size_t> vIndices = F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r);
        if (vIndices.empty()) 
            continue;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;
            if (F.mvpMapPoints[idx] && F.mvpMapPoints[idx]->Observations() > 0)
                continue;
            const cv::Mat &d = F.mDescriptors.row(idx);
            const float dist = DescriptorDistance(MPdescriptor, d);
            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx = idx;
            }
            else if (dist < bestDist2)
                bestDist2 = dist;
        }
        if (bestDist > TH_HIGH && bestDist > mfNNratio * bestDist2)
            continue;
        F.mvpMapPoints[bestIdx] = pMP;
        pMP->mnTrackedbyFrame = F.mnId;
 
        // seed growing
        std::deque<unsigned int> matchSeed;
        matchSeed.push_back(bestIdx);
        while (matchSeed.size() > 0)
        {
            unsigned int keyID = matchSeed.front();
            matchSeed.pop_front();
            std::vector<MapEdge*> mapEdge_set = pMP->getEdges();
            const std::vector<unsigned int> &keyEdge_set = F.mvKeysUn[keyID].mvConnected;
            if(mapEdge_set.empty() || keyEdge_set.empty())
                continue;

            Eigen::MatrixXf weight(mapEdge_set.size(), keyEdge_set.size());
            weight.setConstant(1e6);
            std::vector<unsigned int> lx;
            std::vector<unsigned int> ly;
            for(unsigned int i=0;i< mapEdge_set.size(); i++)
            {
                if(mapEdge_set[i]->isBad() || !mapEdge_set[i]->mbValid || mapEdge_set[i]->theOtherPt(pMP) == nullptr)
                    continue;
                lx.push_back(i);
            }
            for (unsigned int j=0; j<keyEdge_set.size(); j++)
                ly.push_back(j);

            for(unsigned int i : lx)
            {
                for(unsigned int j : ly)
                {
                    MapPoint *pMP_o = mapEdge_set[i]->theOtherPt(pMP);
                    unsigned int keyID_o = F.mvKeyEdges[keyEdge_set[j]].theOtherPid(keyID);
                    if(pMP_o == F.mvpMapPoints[keyID_o])
                        weight(i, j) = -1;
                    else
                    {
                        cv::Mat D1 = pMP_o->GetDescriptor();
                        cv::Mat D2 = F.mDescriptors.row(keyID_o);
                        float score = DescriptorDistance(D1, D2);
                        weight(i, j) = score;
                    }
                }
            }
            // search
            while (!lx.empty() && !ly.empty())
            {
                unsigned int minlx(0),minly(0);
                float minWeight = 1e6;
                for(unsigned int i=0; i < lx.size(); i++)
                {
                    for(unsigned int j=0; j < ly.size(); j++)
                    {
                        if(weight(lx[i],ly[j]) < minWeight)
                        {
                            minWeight = weight(lx[i],ly[j]);
                            minlx = i;
                            minly = j;
                        }
                    }
                }
                if(minWeight > TH_HIGH)
                    break; // no more matches
    
                unsigned int mapEdge_set_id = lx[minlx];
                unsigned int keyEdge_set_id = ly[minly];
                lx.erase(lx.begin() + minlx);
                ly.erase(ly.begin() + minly);
                
                MapEdge* pME = mapEdge_set[mapEdge_set_id];
                unsigned int keyEdgeID = keyEdge_set[keyEdge_set_id];

                MapPoint *pMP_o = pME->theOtherPt(pMP);
                unsigned int keyID_o = F.mvKeyEdges[keyEdgeID].theOtherPid(keyID);
                if(pMP_o == nullptr || pMP_o->isBad() || pMP_o->mnTrackedbyFrame == F.mnId)
                    continue; 
                F.mvpMapPoints[keyID_o] = pMP_o;
                F.mvpMapEdges[keyEdgeID] = pME;
                pMP_o->mnTrackedbyFrame = F.mnId;
                matchSeed.push_back(keyID_o);
            }
        }
        nmatches++;
    }
    return nmatches;
}

int Matcher::SearchByBoW(KeyFrame *pKF, Frame &F, vector<MapPoint *> &vpMapPointMatches)
{
    const vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

    const DBoW3::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches = 0;

    // We perform the matching over vocabulary that belong to the same vocabulary node (at a certain level)
    DBoW3::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW3::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW3::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW3::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while (KFit != KFend && Fit != Fend)
    {
        if (KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint *pMP = vpMapPointsKF[realIdxKF];

                if (!pMP)
                    continue;

                if (pMP->isBad())
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);

                float bestDist1 = 1e6;
                int bestIdxF = -1;
                float bestDist2 = 1e6;

                for (size_t iF = 0; iF < vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];
                    if (vpMapPointMatches[realIdxF])
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);
                    const float dist = DescriptorDistance(dKF, dF);

                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdxF = realIdxF;
                    }
                    else if (dist < bestDist2)
                        bestDist2 = dist;
                }

                if (bestDist1 <= TH_LOW)
                {
                    if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF] = pMP;
                        const KeyPointEx &kp = pKF->mvKeysUn[realIdxKF];
                        nmatches++;
                    }
                }
            }

            KFit++;
            Fit++;
        }
        else if (KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }
    return nmatches;
}

int Matcher::SearchByProjection(KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints,
                                   vector<MapPoint *> &vpMatched, int th, float ratioHamming)
{
    // Get Calibration Parameters for later projection
    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint *> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint *>(NULL));

    int nmatches = 0;

    // For each Candidate MapPoint Project and Match
    for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++)
    {
        MapPoint *pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if (pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if (p3Dc(2) < 0.0)
            continue;

        // Project into Image
        const Eigen::Vector2f uv = mpCamera->project(p3Dc);

        // Point must be inside the image
        if (!pKF->IsInImage(uv(0), uv(1)))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw - Ow;
        const float dist = PO.norm();

        if (dist < minDistance || dist > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if (PO.dot(Pn) < 0.5 * dist)
            continue;

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0), uv(1), th);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = 1e6;
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;
            if (vpMatched[idx])
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const float dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_LOW * ratioHamming)
        {
            vpMatched[bestIdx] = pMP;
            nmatches++;
        }
    }

    return nmatches;
}

int Matcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches = 0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);

    vector<int> vMatchedDistance(F2.mvKeysUn.size(), INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(), -1);

    for (size_t i1 = 0, iend1 = F1.mvKeysUn.size(); i1 < iend1; i1++)
    {
        KeyPointEx kp1 = F1.mvKeysUn[i1];
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize);

        if (vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        float bestDist = 1e6;
        float bestDist2 = 1e6;
        int bestIdx2 = -1;

        for (vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            float dist = DescriptorDistance(d1, d2);

            if (vMatchedDistance[i2] <= dist)
                continue;

            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            }
            else if (dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        if (bestDist <= TH_LOW)
        {
            if (bestDist < (float)bestDist2 * mfNNratio)
            {
                if (vnMatches21[bestIdx2] >= 0)
                {
                    vnMatches12[vnMatches21[bestIdx2]] = -1;
                    nmatches--;
                }
                vnMatches12[i1] = bestIdx2;
                vnMatches21[bestIdx2] = i1;
                vMatchedDistance[bestIdx2] = bestDist;
                nmatches++;
            }
        }
    }
    // Update prev matched
    for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
        if (vnMatches12[i1] >= 0)
            vbPrevMatched[i1] = cv::Point2f( F2.mvKeysUn[vnMatches12[i1]].mPos[0],F2.mvKeysUn[vnMatches12[i1]].mPos[1]);

    return nmatches;
}

int Matcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint *>(vpMapPoints1.size(), static_cast<MapPoint *>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(), false);

    int nmatches = 0;

    DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                MapPoint *pMP1 = vpMapPoints1[idx1];
                if (!pMP1)
                    continue;
                if (pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                float bestDist1 = 1e6;
                int bestIdx2 = -1;
                float bestDist2 = 1e6;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint *pMP2 = vpMapPoints2[idx2];

                    if (vbMatched2[idx2] || !pMP2)
                        continue;

                    if (pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    float dist = DescriptorDistance(d1, d2);

                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx2 = idx2;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }

                if (bestDist1 < TH_LOW)
                {
                    if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1] = vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2] = true;
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }
    return nmatches;
}

int Matcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, vector<pair<size_t, size_t>> &vMatchedPairs, const bool bCoarse)
{
    const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    // Compute epipole in second image
    Sophus::SE3f T1w = pKF1->GetPose();
    Sophus::SE3f T2w = pKF2->GetPose();
    Sophus::SE3f Tw2 = pKF2->GetPoseInverse(); // for convenience
    Eigen::Vector3f Cw = pKF1->GetCameraCenter();
    Eigen::Vector3f C2 = T2w * Cw;

    Eigen::Vector2f ep = mpCamera->project(C2);
    Sophus::SE3f T12;
    Sophus::SE3f Tll, Tlr, Trl, Trr;
    Eigen::Matrix3f R12; // for fastest computation
    Eigen::Vector3f t12; // for fastest computation

    T12 = T1w * Tw2;
    R12 = T12.rotationMatrix();
    t12 = T12.translation();

    // Find matches between not tracked keypoints
    // Matching speed-up by Vocabulary
    // Compare only Vocabulary that share the same node
    int nmatches = 0;
    vector<bool> vbMatched2(pKF2->N, false);
    vector<int> vMatches12(pKF1->N, -1);

    DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint *pMP1 = pKF1->GetMapPoint(idx1);

                // If there is already a MapPoint skip
                if (pMP1)
                {
                    continue;
                }

                const KeyPointEx &kp1 = pKF1->mvKeysUn[idx1];

                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                float bestDist = TH_LOW;
                int bestIdx2 = -1;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    MapPoint *pMP2 = pKF2->GetMapPoint(idx2);

                    // If we have already matched or there is a MapPoint skip
                    if (vbMatched2[idx2] || pMP2)
                        continue;

                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                    const float dist = DescriptorDistance(d1, d2);

                    if (dist > TH_LOW || dist > bestDist)
                        continue;

                    const KeyPointEx &kp2 = pKF2->mvKeysUn[idx2];

                    if ((ep - kp2.mPos).norm() < 10.0f) // 10 pixels
                        continue;

                    if (mpCamera->epipolarConstrain(kp1, kp2, R12, t12)) // MODIFICATION_2
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                if (bestIdx2 >= 0)
                {
                    vMatches12[idx1] = bestIdx2;
                    nmatches++;
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
    {
        if (vMatches12[i] < 0)
            continue;
        vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
    }

    return nmatches;
}

int Matcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    Sophus::SE3f Tcw;
    Eigen::Vector3f Ow;

    Tcw = pKF->GetPose();
    Ow = pKF->GetCameraCenter();

    int nFused = 0;
    const int nMPs = vpMapPoints.size();

    // For debbuging
    int count_notMP = 0, count_bad = 0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal = 0, count_notidx = 0, count_thcheck = 0;
    for (int i = 0; i < nMPs; i++)
    {
        MapPoint *pMP = vpMapPoints[i];

        if (!pMP)
        {
            count_notMP++;
            continue;
        }

        if (pMP->isBad())
        {
            count_bad++;
            continue;
        }
        else if (pMP->IsInKeyFrame(pKF))
        {
            count_isinKF++;
            continue;
        }

        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if (p3Dc(2) < 0.0f)
        {
            count_negdepth++;
            continue;
        }

        const float invz = 1 / p3Dc(2);

        const Eigen::Vector2f uv = mpCamera->project(p3Dc);

        // Point must be inside the image
        if (!pKF->IsInImage(uv(0), uv(1)))
        {
            count_notinim++;
            continue;
        }

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw - Ow;
        const float dist3D = PO.norm();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
        {
            count_dist++;
            continue;
        }

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if (PO.dot(Pn) < 0.5 * dist3D)
        {
            count_normal++;
            continue;
        }

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0), uv(1), th);

        if (vIndices.empty())
        {
            count_notidx++;
            continue;
        }

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = 1e6;
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            size_t idx = *vit;
            const KeyPointEx &kp = pKF->mvKeysUn[idx];

            const float &kpx = kp.mPos[0];
            const float &kpy = kp.mPos[1];
            const float ex = uv(0) - kpx;
            const float ey = uv(1) - kpy;
            const float e2 = ex * ex + ey * ey;

            if (e2  > 5.99)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const float dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if (bestDist <= TH_LOW)
        {
            MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
            if (pMPinKF)
            {
                if (!pMPinKF->isBad())
                {
                    if (pMPinKF->Observations() > pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }
            nFused++;
        }
        else
            count_thcheck++;
    }

    return nFused;
}

int Matcher::Fuse(KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Decompose Scw
    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint *> spAlreadyFound = pKF->GetMapPoints();

    int nFused = 0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for (int iMP = 0; iMP < nPoints; iMP++)
    {
        MapPoint *pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if (pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if (p3Dc(2) < 0.0f)
            continue;

        // Project into Image
        const Eigen::Vector2f uv = mpCamera->project(p3Dc);

        // Point must be inside the image
        if (!pKF->IsInImage(uv(0), uv(1)))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw - Ow;
        const float dist3D = PO.norm();

        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if (PO.dot(Pn) < 0.5 * dist3D)
            continue;

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0), uv(1), th);
        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = 1e6;
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(); vit != vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            float dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if (bestDist <= TH_LOW)
        {
            MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
            if (pMPinKF)
            {
                if (!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int Matcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th)
{
    // Camera 1 & 2 from world
    Sophus::SE3f T1w = pKF1->GetPose();
    Sophus::SE3f T2w = pKF2->GetPose();

    // Transformation between cameras
    Sophus::Sim3f S21 = S12.inverse();

    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1, false);
    vector<bool> vbAlreadyMatched2(N2, false);

    for (int i = 0; i < N1; i++)
    {
        MapPoint *pMP = vpMatches12[i];
        if (pMP)
        {
            vbAlreadyMatched1[i] = true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if (idx2 >= 0 && idx2 < N2)
                vbAlreadyMatched2[idx2] = true;
        }
    }

    vector<int> vnMatch1(N1, -1);
    vector<int> vnMatch2(N2, -1);

    // Transform from KF1 to KF2 and search
    for (int i1 = 0; i1 < N1; i1++)
    {
        MapPoint *pMP = vpMapPoints1[i1];

        if (!pMP || vbAlreadyMatched1[i1])
            continue;

        if (pMP->isBad())
            continue;

        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        Eigen::Vector3f p3Dc1 = T1w * p3Dw;
        Eigen::Vector3f p3Dc2 = S21 * p3Dc1;

        // Depth must be positive
        if (p3Dc2(2) < 0.0)
            continue;

        const Eigen::Vector2f uv = mpCamera->project(p3Dc2);

        // Point must be inside the image
        if (!pKF2->IsInImage(uv[0], uv[1]))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = p3Dc2.norm();

        // Depth must be inside the scale invariance region
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(uv[0], uv[1], th);
        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = 1e6;
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const float dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_HIGH)
        {
            vnMatch1[i1] = bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for (int i2 = 0; i2 < N2; i2++)
    {
        MapPoint *pMP = vpMapPoints2[i2];

        if (!pMP || vbAlreadyMatched2[i2])
            continue;

        if (pMP->isBad())
            continue;

        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        Eigen::Vector3f p3Dc2 = T2w * p3Dw;
        Eigen::Vector3f p3Dc1 = S12 * p3Dc2;

        // Depth must be positive
        if (p3Dc1(2) < 0.0)
            continue;

        const Eigen::Vector2f uv = mpCamera->project(p3Dc1);

        // Point must be inside the image
        if (!pKF1->IsInImage(uv[0], uv[1]))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = p3Dc1.norm();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(uv[0], uv[1], th);
        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = 1e6;
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const float dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_HIGH)
        {
            vnMatch2[i2] = bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for (int i1 = 0; i1 < N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if (idx2 >= 0)
        {
            int idx1 = vnMatch2[idx2];
            if (idx1 == i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

int Matcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint *> &sAlreadyFound, const float th, const float descDist)
{
    int nmatches = 0;

    const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMPs[i];

        if (pMP)
        {
            if (!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                // Project
                Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                Eigen::Vector3f x3Dc = Tcw * x3Dw;

                const Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                if (uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX)
                    continue;
                if (uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                Eigen::Vector3f PO = x3Dw - Ow;
                float dist3D = PO.norm();

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if (dist3D < minDistance || dist3D > maxDistance)
                    continue;

                // Search in a window
                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), th);

                if (vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                float bestDist = 1e6;
                int bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if (CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const float dist = DescriptorDistance(dMP, d);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }

                if (bestDist <= descDist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                    nmatches++;
                }
            }
        }
    }
    return nmatches;
}
