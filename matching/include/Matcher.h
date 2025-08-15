/**
 * @file Matcher.h
 * @brief Feature matching for PPG-SLAM
 */

#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "sophus/sim3.hpp"
#include "MapPoint.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "PPGGraph.h"

/**
 * @brief Feature matcher for SLAM
 */
class Matcher
{
public:
    /** Constructor */
    Matcher(GeometricCamera* pCam, float nnratio = 0.6);

    // Projection matching
    /** Search by projecting MapPoints */
    int SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th = 3);
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th);
    int SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const std::set<MapPoint *> &sAlreadyFound, const float th, const float descDist);
    int SearchByProjection(KeyFrame* pKF, Sophus::Sim3f &Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th, float ratioHamming=1.0);

    // BoW matching
    /** Search using Bag-of-Words */
    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches);
    int SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12);

    // Special matching
    /** Search for initialization (monocular) */
    int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize = 10);
    
    /** Search for triangulation */
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<pair<size_t, size_t>> &vMatchedPairs, const bool bCoarse = false);
    
    /** Search using Sim3 transformation */
    int SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th);

    // Map operations
    /** Fuse MapPoints to remove duplicates */
    int Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th = 3.0);
    int Fuse(KeyFrame *pKF, Sophus::Sim3f &Scw, const std::vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint);
    
    /** Extend map matches */
    int ExtendMapMatches(Frame &F, const vector<MapPoint *> &vpMapPoints, const float th);
    
public:
    static const float TH_LOW;
    static const float TH_HIGH;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    GeometricCamera* mpCamera;
    float mfNNratio;
};
