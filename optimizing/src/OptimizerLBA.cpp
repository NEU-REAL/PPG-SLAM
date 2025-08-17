#include "Optimizer.h"
#include <iostream>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include "G2oEdge.h"
#include "G2oVertex.h"
#include <mutex>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>

using namespace std;

/**
 * @brief Local Bundle Adjustment for visual SLAM optimization
 * 
 * Algorithm principle:
 * 1. Select local keyframes: current keyframe and its covisible neighbors
 * 2. Select local map points: points observed by local keyframes
 * 3. Select fixed keyframes: frames that observe local points but are not local
 * 4. Build graph optimization problem with pose and landmark vertices
 * 5. Add visual projection constraints and colinearity constraints
 * 6. Optimize poses and landmarks jointly using Levenberg-Marquardt
 * 7. Remove outlier observations and update optimized results
 * 
 * @param pKF         Current keyframe (center of local optimization)
 * @param pbStopFlag  Flag to stop optimization early if needed
 * @param pMap        Map containing keyframes and landmarks
 */


void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{
    // 1. Select local keyframes: current keyframe and covisible neighbors
    list<KeyFrame*> lLocalKeyFrames;
    
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // 2. Select local map points observed by local keyframes
    int num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        if (pKFi->mnId == pMap->GetOriginKF()->mnId)
        {
            num_fixedKF = 1;
        }
        
        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for (vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint* pMP = *vit;
            if (pMP && !pMP->isBad())
            {
                if (pMP->mnBALocalForKF != pKF->mnId)
                {
                    lLocalMapPoints.push_back(pMP);
                    pMP->mnBALocalForKF = pKF->mnId;
                }
            }
        }
    }

    // 3. Select fixed keyframes: frames that observe local points but are not local
    list<KeyFrame*> lFixedCameras;
    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map<KeyFrame*, int> observations = (*lit)->GetObservations();
        for (map<KeyFrame*, int>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    
    num_fixedKF = lFixedCameras.size() + num_fixedKF;
    if (num_fixedKF == 0)
    {
        cerr << "LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted" << endl;
        return;
    }

    // 4. Setup optimizer using Levenberg-Marquardt algorithm
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

    solver->setUserLambdaInit(100.0);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // 5. Add local keyframe vertices (optimizable poses)
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId == pMap->GetOriginKF()->mnId);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // 6. Add fixed keyframe vertices (fixed poses for stability)
    for (list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // 7. Add map point vertices and visual projection edges
    vector<EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve((lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size());

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve((lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size());

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve((lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size());

    // Cauchy kernel threshold (more robust to outliers than Huber)
    const float thCauchyMono = sqrt(5.991);

    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexPointXYZ* vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(false);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*, int> observations = pMP->GetObservations();

        // Create visual projection edges
        for (map<KeyFrame*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if (!pKFi->isBad())
            {
                const int index = mit->second;
                // Monocular observation
                if (index != -1)
                {
                    const KeyPointEx &kpUn = pKFi->mvKeysUn[index];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.mPos[0], kpUn.mPos[1];
                    
                    EdgeSE3ProjectXYZ* e = new EdgeSE3ProjectXYZ();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity());
                    
                    // Use Cauchy robust kernel for stronger outlier rejection
                    g2o::RobustKernelCauchy* rk = new g2o::RobustKernelCauchy;
                    e->setRobustKernel(rk);
                    rk->setDelta(thCauchyMono);
                    e->pCamera = pMap->mpCamera;
                    
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
            }
        }
    }

    // 8. Add colinearity constraints for geometric consistency
    for (MapPoint* pMP : lLocalMapPoints)
    {
        if (optimizer.vertex(pMP->mnId + maxKFid + 1) == NULL)
            continue;
            
        vector<MapColine*> vMCs = pMP->getColinearity();
        for (auto pMC : vMCs)
        {
            if (pMC->isBad() || !pMC->mbValid)
                continue;
                
            MapPoint* pMPs = pMC->mpMPs;
            MapPoint* pMPe = pMC->mpMPe;
            
            if (optimizer.vertex(pMPs->mnId + maxKFid + 1) == NULL ||
                optimizer.vertex(pMPe->mnId + maxKFid + 1) == NULL)
                continue;
                
            EdgeColine* e = new EdgeColine();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPs->mnId + maxKFid + 1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMP->mnId + maxKFid + 1)));
            e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPe->mnId + maxKFid + 1)));
            e->setInformation(Eigen::Matrix3d::Identity() * pMC->aveWeight());
            
            // Add Huber robust kernel for moderate outlier rejection in colinearity
            g2o::RobustKernelHuber* rkColine = new g2o::RobustKernelHuber;
            e->setRobustKernel(rkColine);
            rkColine->setDelta(sqrt(7.815)); // Chi-squared threshold for 3 DOF (95% confidence)
            
            optimizer.addEdge(e);
        }
    }

    // 9. Execute optimization
    if (pbStopFlag && *pbStopFlag)
        return;

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // 10. Remove outlier observations based on chi-squared test
    vector<pair<KeyFrame*, MapPoint*>> vToErase;
    vToErase.reserve(vpEdgesMono.size());

    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // 11. Update map with optimized results
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if (!vToErase.empty())
    {
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
        
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            MapPoint* pMPi = vToErase[i].second;
            if (pMPi->isBad())
                pMap->EraseMapPoint(pMPi);
        }
    }

    // 12. Recover optimized keyframe poses
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->SetPose(Tiw);
    }

    // 13. Recover optimized map point positions
    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    // 14. Clean up bad colinearity edges
    for (MapPoint* pMP : lLocalMapPoints)
    {
        vector<MapEdge*> vpMEs = pMP->getEdges();
        for (MapEdge* pME : vpMEs)
        {
            if (!pME || pME->isBad() || pME->mnBALocalForKF == pKF->mnId)
                continue;
            pME->mnBALocalForKF = pKF->mnId;
            pME->checkValid();
        }
    }
    
    for (MapPoint* pMP : lLocalMapPoints)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;
        vector<MapColine*> vCLs = pMP->removeColineOutliers();
        for (MapColine* pMC : vCLs)
            pMap->EraseMapColine(pMC);
    }
    
    pMap->InfoMapChange();
}

/**
 * @brief Local Bundle Adjustment with Inertial constraints
 * 
 * This function performs local bundle adjustment combining visual and inertial measurements.
 * It optimizes poses, velocities, biases of local keyframes and 3D positions of map points
 * using both visual reprojection errors and IMU preintegration constraints.
 * 
 * Algorithm:
 * 1. Select temporal optimization window (recent keyframes)
 * 2. Add covisible keyframes for visual constraints
 * 3. Build optimization graph with pose, velocity, bias vertices
 * 4. Add inertial edges between consecutive keyframes
 * 5. Add visual projection edges for map points
 * 6. Add colinearity constraints for line features
 * 7. Optimize using Levenberg-Marquardt algorithm
 * 8. Remove outliers and update optimized values
 * 
 * @param pKF Current keyframe (center of optimization window)
 * @param pbStopFlag External stop signal for early termination
 * @param pMap Map containing keyframes and map points
 * @param bLarge Use larger lambda initialization for robust optimization
 * @param bRecInit Recovery mode with relaxed constraints
 */
void Optimizer::LocalInertialBA(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, bool bLarge, bool bRecInit)
{
    // Optimization parameters
    const int maxOpt = 50;       // Maximum number of keyframes in temporal window
    const int opt_it = 10;       // Number of optimization iterations
    
    // Calculate temporal window size
    const int Nd = std::min(static_cast<int>(pMap->KeyFramesInMap()) - 2, maxOpt);
    const unsigned long maxKFid = pKF->mnId;

    // Initialize containers for optimization window
    vector<KeyFrame*> vpOptimizableKFs;
    const vector<KeyFrame*> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
    list<KeyFrame*> lpOptVisKFs;

    // 1. Build temporal keyframe optimization window
    vpOptimizableKFs.reserve(Nd);
    vpOptimizableKFs.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    
    // Add previous keyframes to temporal window
    for (int i = 1; i < Nd; i++)
    {
        if (vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
        }
        else
            break;
    }

    int N = vpOptimizableKFs.size();

    // 2. Collect map points observed by temporal keyframes
    list<MapPoint*> lLocalMapPoints;
    for (int i = 0; i < N; i++)
    {
        vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for (vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint* pMP = *vit;
            if (pMP && !pMP->isBad() && pMP->mnBALocalForKF != pKF->mnId)
            {
                lLocalMapPoints.push_back(pMP);
                pMP->mnBALocalForKF = pKF->mnId;
            }
        }
    }

    // 3. Set fixed keyframe (first frame before optimization window)
    list<KeyFrame*> lFixedKeyFrames;
    if (vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF = pKF->mnId;
    }
    else
    {
        // If no previous keyframe, fix the last one in optimization window
        vpOptimizableKFs.back()->mnBALocalForKF = 0;
        vpOptimizableKFs.back()->mnBAFixedForKF = pKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // 4. Add covisible keyframes for visual constraints (currently disabled)
    // This feature is disabled for local inertial BA to improve performance

    // 5. Add fixed keyframes that observe local map points
    const int maxFixKF = 200;
    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map<KeyFrame*, int> observations = (*lit)->GetObservations();
        for (map<KeyFrame*, int>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                {
                    lFixedKeyFrames.push_back(pKFi);
                    break;
                }
            }
        }
        if (lFixedKeyFrames.size() >= maxFixKF)
            break;
    }

    // 6. Setup g2o optimizer with Levenberg-Marquardt algorithm
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

    // Set lambda initialization based on problem scale
    if (bLarge)
        solver->setUserLambdaInit(1e-2);  // Smaller lambda for large problems
    else
        solver->setUserLambdaInit(1e0);   // Standard lambda

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // 7. Add temporal keyframe vertices (pose, velocity, bias)
    N = vpOptimizableKFs.size();
    for (int i = 0; i < N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        // Add pose vertex
        VertexPose *VP = new VertexPose(pKFi, pMap->mpCamera);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        // Add IMU state vertices if keyframe has IMU measurements
        if (pKFi->bImu)
        {
            // Velocity vertex
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            
            // Gyroscope bias vertex
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            
            // Accelerometer bias vertex
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // 8. Add visual keyframe vertices (only pose)
    for (list<KeyFrame*>::iterator it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; it++)
    {
        KeyFrame* pKFi = *it;
        VertexPose *VP = new VertexPose(pKFi, pMap->mpCamera);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);
    }

    // 9. Add fixed keyframe vertices
    for (list<KeyFrame*>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        VertexPose *VP = new VertexPose(pKFi, pMap->mpCamera);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        // Add fixed IMU vertices (only for keyframe before temporal window)
        if (pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // 10. Create inertial constraints between consecutive keyframes
    vector<EdgeInertial*> vei(N, (EdgeInertial*)NULL);
    vector<EdgeGyroRW*> vegr(N, (EdgeGyroRW*)NULL);
    vector<EdgeAccRW*> vear(N, (EdgeAccRW*)NULL);

    for (int i = 0; i < N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        if (!pKFi->mPrevKF)
        {
            cout << "ERROR: No inertial link to previous frame!" << endl;
            continue;
        }

        // Add inertial edge if both keyframes have IMU measurements
        if (pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            // Update bias for preintegration
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            
            // Get vertices for both keyframes
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 3);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3);

            if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error creating inertial edge - missing vertices" << endl;
                continue;
            }

            // Create inertial preintegration edge
            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);
            vei[i]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vei[i]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vei[i]->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vei[i]->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vei[i]->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vei[i]->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            // Apply robust kernel for boundary edges or recovery mode
            if (i == N - 1 || bRecInit)
            {
                // Use Huber kernel for IMU constraints (more stable for inertial data)
                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                vei[i]->setRobustKernel(rki);
                if (i == N - 1)
                    vei[i]->setInformation(vei[i]->information() * 1e-2);
                rki->setDelta(sqrt(16.92));
            }
            optimizer.addEdge(vei[i]);

            // Create gyroscope bias random walk edge
            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vegr[i]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG2));
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            // Create accelerometer bias random walk edge
            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vear[i]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA2));
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);
            optimizer.addEdge(vear[i]);
        }
        else
        {
            cout << "ERROR: Cannot build inertial edge for keyframe " << pKFi->mnId << endl;
        }
    }

    // 11. Set up map point vertices and visual projection edges
    // Visual projection edges (monocular)
    vector<EdgeMono*> vpEdgesMono;
    vpEdgesMono.reserve((N + lFixedKeyFrames.size()) * lLocalMapPoints.size());

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve((N + lFixedKeyFrames.size()) * lLocalMapPoints.size());

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve((N + lFixedKeyFrames.size()) * lLocalMapPoints.size());

    // Robust kernel thresholds - Cauchy provides stronger outlier rejection
    const float thCauchyMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;

    const unsigned long iniMPid = maxKFid * 5;

    // Create map point vertices and visual constraints
    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexPointXYZ* vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());

        unsigned long id = pMP->mnId + iniMPid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(false);
        optimizer.addVertex(vPoint);
        
        const map<KeyFrame*, int> observations = pMP->GetObservations();

        // Create visual projection constraints
        for (map<KeyFrame*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            // Skip if keyframe is not in optimization or fixed set
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                continue;

            if (!pKFi->isBad())
            {
                const int index = mit->second;
                
                // Create monocular observation edge
                if (index != -1)
                {
                    const KeyPointEx &kpUn = pKFi->mvKeysUn[index];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.mPos[0], kpUn.mPos[1];
                    
                    EdgeMono* e = new EdgeMono();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity());
                    
                    // Add Cauchy robust kernel for stronger outlier rejection
                    g2o::RobustKernelCauchy* rk = new g2o::RobustKernelCauchy;
                    e->setRobustKernel(rk);
                    rk->setDelta(thCauchyMono);
                    
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
            }
        }
    }

    // 12. Add colinearity constraints for line features
    for (MapPoint* pMP : lLocalMapPoints)
    {
        if (optimizer.vertex(pMP->mnId + iniMPid + 1) == NULL)
            continue;
            
        vector<MapColine*> vMCs = pMP->getColinearity();
        for (auto pMC : vMCs)
        {
            if (pMC->isBad() || !pMC->mbValid)
                continue;
                
            MapPoint* pMPs = pMC->mpMPs;
            MapPoint* pMPe = pMC->mpMPe;
            
            // Check if all three points have vertices in optimizer
            if (optimizer.vertex(pMPs->mnId + iniMPid + 1) == NULL ||
                optimizer.vertex(pMPe->mnId + iniMPid + 1) == NULL)
                continue;
                
            // Create colinearity edge connecting three points
            EdgeColine* e = new EdgeColine();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPs->mnId + iniMPid + 1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMP->mnId + iniMPid + 1)));
            e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPe->mnId + iniMPid + 1)));
            e->setInformation(Eigen::Matrix3d::Identity() * pMC->aveWeight());
            
            // Add Huber robust kernel for moderate outlier rejection in line features
            g2o::RobustKernelHuber* rkColine = new g2o::RobustKernelHuber;
            e->setRobustKernel(rkColine);
            rkColine->setDelta(sqrt(7.815)); // Chi-squared threshold for 3 DOF (95% confidence)
            
            optimizer.addEdge(e);
        }
    }

    // 13. Execute optimization
    optimizer.initializeOptimization();
    optimizer.optimize(opt_it);
    
    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    // 14. Collect outlier observations for removal
    vector<pair<KeyFrame*, MapPoint*>> vToErase;
    vToErase.reserve(vpEdgesMono.size());

    // Check monocular visual edges for outliers
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        EdgeMono* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        bool bClose = pMP->mTrackDepth < 10.f;

        if (pMP->isBad())
            continue;

        // Apply different thresholds for close and far points
        if ((e->chi2() > chi2Mono2 && !bClose) || 
            (e->chi2() > 1.5f * chi2Mono2 && bClose) || 
            !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get map mutex for thread-safe operations
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // 15. Remove outlier observations
    if (!vToErase.empty())
    {
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
        
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            MapPoint* pMPi = vToErase[i].second;
            if (pMPi->isBad())
                pMap->EraseMapPoint(pMPi);
        }
    }

    // Reset fixed keyframe flags
    for (list<KeyFrame*>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
        (*lit)->mnBAFixedForKF = 0;

    // 16. Recover optimized temporal keyframe states
    N = vpOptimizableKFs.size();
    for (int i = 0; i < N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        // Recover optimized pose
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        SE3f Tcw(VP->estimate().Rcw.cast<float>(), VP->estimate().tcw.cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF = 0;

        // Recover optimized IMU states if available
        if (pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]));
        }
    }

    // 17. Recover optimized visual keyframe poses
    for (list<KeyFrame*>::iterator it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; it++)
    {
        KeyFrame* pKFi = *it;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        SE3f Tcw(VP->estimate().Rcw.cast<float>(), VP->estimate().tcw.cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF = 0;
    }

    // 18. Recover optimized map point positions
    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(pMP->mnId + iniMPid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    // 20. Clean up colinearity constraints
    for (MapPoint* pMP : lLocalMapPoints)
    {
        vector<MapEdge*> vpMEs = pMP->getEdges();
        for (MapEdge* pME : vpMEs)
        {
            if (!pME || pME->isBad() || pME->mnBALocalForKF == pKF->mnId)
                continue;
            pME->mnBALocalForKF = pKF->mnId;
            pME->checkValid();
        }
    }
    
    for (MapPoint* pMP : lLocalMapPoints)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;
        vector<MapColine*> vCLs = pMP->removeColineOutliers();
        for (MapColine* pMC : vCLs)
            pMap->EraseMapColine(pMC);
    }

    pMap->InfoMapChange();
}