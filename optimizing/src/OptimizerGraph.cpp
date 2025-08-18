#include "Optimizer.h"
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include "G2oEdge.h"
#include "G2oVertex.h"
#include<mutex>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

/**
 * @brief Essential Graph Optimization for loop closure correction
 * @details Optimizes the pose graph using Sim3 poses to correct accumulated drift
 *          after loop closure detection. The essential graph contains only keyframes
 *          and their relative constraints, making optimization efficient.
 *          Core algorithm: minimize relative pose errors in Sim3 space.
 * 
 * @param pMap               Pointer to the map containing keyframes and map points
 * @param pLoopKF            Loop closure keyframe
 * @param pCurKF             Current keyframe that closed the loop
 * @param NonCorrectedSim3   Original Sim3 poses before loop correction
 * @param CorrectedSim3      Corrected Sim3 poses from loop detection
 * @param LoopConnections    Loop closure connections between keyframes
 * @param bFixScale          Whether to fix the scale (true for monocular SLAM)
 */
void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const KeyFrameAndPose &NonCorrectedSim3,
                                       const KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{   
    // ========== OPTIMIZER SETUP ==========
    // Create Levenberg-Marquardt optimizer with Eigen linear solver for 7-DOF Sim3 optimization
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<g2o::BlockSolver_7_3>(std::move(linear_solver)));

    solver->setUserLambdaInit(1e-16);  // Very small initial damping for better convergence

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(solver);

    // Get all map elements
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    // Pre-allocate vectors for Sim3 poses and vertices
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid + 1);

    const int minFeat = 100;  // Minimum features for valid covisibility connection

    // ========== SIM3 VERTICES ==========
    // Add Sim3 pose vertices for each keyframe
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
            
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();
        const int nIDi = pKF->mnId;

        // Check if this keyframe has a corrected pose from loop detection
        KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);
        if (it != CorrectedSim3.end())
        {
            // Use corrected Sim3 pose from loop closure
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            // Convert SE3 pose to Sim3 (with unit scale)
            SE3d Tcw = pKF->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        // Fix the origin keyframe to prevent gauge freedom
        if (pKF->mnId == pMap->GetOriginKF()->mnId)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;  // Fix scale for monocular SLAM

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;
    }

    // ========== LOOP CLOSURE EDGES ==========
    // Track inserted edges to avoid duplicates
    set<pair<long unsigned int, long unsigned int>> sInsertedEdges;
    
    // Information matrix for Sim3 constraints (7x7 for [R|t|s])
    const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

    // Add loop closure edges
    int count_loop = 0;
    for (map<KeyFrame*, set<KeyFrame*>>::const_iterator mit = LoopConnections.begin(), 
         mend = LoopConnections.end(); mit != mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*>& spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for (set<KeyFrame*>::const_iterator sit = spConnections.begin(), 
             send = spConnections.end(); sit != send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            
            // Skip weak connections except for current-loop pair
            if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && 
                pKF->GetWeight(*sit) < minFeat)
                continue;

            // Compute relative Sim3 transformation
            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            // Create Sim3 edge constraint
            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);
            e->information() = matLambda;

            optimizer.addEdge(e);
            count_loop++;
            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
        }
    }

    // ========== COVISIBILITY EDGES ==========
    // Add edges for keyframes in covisibility graph
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];
        const int nIDi = pKF->mnId;

        // Get inverse pose (use non-corrected if available)
        g2o::Sim3 Swi;
        KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);
        if (iti != NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        // Add explicit loop edges (detected by loop detector)
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for (set<KeyFrame*>::const_iterator sit = sLoopEdges.begin(), 
             send = sLoopEdges.end(); sit != send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if (pLKF->mnId < pKF->mnId)  // Avoid duplicate edges
            {
                // Get loop keyframe pose
                g2o::Sim3 Slw;
                KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);
                if (itl != NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                // Compute relative transformation
                g2o::Sim3 Sli = Slw * Swi;
                
                // Create loop edge
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Add covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFrame*>::const_iterator vit = vpConnectedKFs.begin(); 
             vit != vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if (pKFn && !pKFn->isBad() && pKFn->mnId < pKF->mnId)
            {
                // Skip if edge already inserted
                pair<long unsigned int, long unsigned int> edge_pair = 
                    make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId));
                if (sInsertedEdges.count(edge_pair))
                    continue;

                // Get neighbor keyframe pose
                g2o::Sim3 Snw;
                KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);
                if (itn != NonCorrectedSim3.end())
                    Snw = itn->second;
                else
                    Snw = vScw[pKFn->mnId];

                // Compute relative transformation
                g2o::Sim3 Sni = Snw * Swi;

                // Create covisibility edge
                g2o::EdgeSim3* en = new g2o::EdgeSim3();
                en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                en->setMeasurement(Sni);
                en->information() = matLambda;
                optimizer.addEdge(en);
            }
        }

        // ========== INERTIAL EDGES ==========
        // Add inertial constraints between consecutive keyframes
        if (pKF->bImu && pKF->mPrevKF)
        {
            // Get previous keyframe pose
            g2o::Sim3 Spw;
            KeyFrameAndPose::const_iterator itp = NonCorrectedSim3.find(pKF->mPrevKF);
            if (itp != NonCorrectedSim3.end())
                Spw = itp->second;
            else
                Spw = vScw[pKF->mPrevKF->mnId];

            // Compute relative transformation
            g2o::Sim3 Spi = Spw * Swi;
            
            // Create inertial edge
            g2o::EdgeSim3* ep = new g2o::EdgeSim3();
            ep->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mPrevKF->mnId)));
            ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            ep->setMeasurement(Spi);
            ep->information() = matLambda;
            optimizer.addEdge(ep);
        }
    }

    // ========== OPTIMIZATION ==========
    // Initialize and run essential graph optimization
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);  // 20 iterations for pose graph optimization
    optimizer.computeActiveErrors();
    
    // Lock map for updating poses
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // ========== POSE RECOVERY ==========
    // Convert optimized Sim3 poses back to SE3 format
    // SE3 Pose Recovering: Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw = VSim3->estimate();
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
        double s = CorrectedSiw.scale();

        // Extract scale and apply to keyframe pose
        SE3f Tiw(CorrectedSiw.rotation().cast<float>(), 
                  CorrectedSiw.translation().cast<float>() / s);
        pKFi->SetPose(Tiw);
    }

    // ========== MAP POINT CORRECTION ==========
    // Correct map points using optimized keyframe poses
    // Transform to "non-optimized" reference keyframe pose and back with optimized pose
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        MapPoint* pMP = vpMPs[i];
        if (pMP->isBad())
            continue;

        // Determine reference keyframe for this map point
        int nIDr;
        if (pMP->mnCorrectedByKF == pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        // Apply pose correction to map point
        g2o::Sim3 Srw = vScw[nIDr];                    // Original reference pose
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];  // Corrected reference pose

        Eigen::Matrix<double, 3, 1> eigP3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        
        pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    // ========== MAP EDGE VALIDATION ==========
    // Remove bad colinearity edges after pose optimization
    vector<MapEdge*> vpMEs = pMap->GetAllMapEdges();
    for (MapEdge* pME : vpMEs)
    {
        if (!pME || pME->isBad())
            continue;
        pME->checkValid();  // Validate edge geometry after pose update
    }
    
    // ========== COLINEARITY CLEANUP ==========
    // Remove outlier colinear point associations after optimization
    for (MapPoint* pMP : vpMPs)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;
        std::vector<MapColine*> vCLs = pMP->removeColineOutliers();
        for (MapColine* pMC : vCLs)
            pMap->EraseMapColine(pMC);
    }
    
    // Notify map of changes for consistency
    pMap->InfoMapChange();
}

/**
 * @brief Sim3 Pose Optimization for loop closure detection
 * @details Optimizes the Sim3 transformation between two keyframes using 3D-2D correspondences.
 *          This function estimates the 7-DOF similarity transformation (rotation, translation, scale)
 *          that best aligns matched features between two keyframes.
 *          Core algorithm: Minimize reprojection errors in both image frames using robust optimization.
 * 
 * @param pMap               Pointer to the map
 * @param pKF1               First keyframe
 * @param pKF2               Second keyframe
 * @param vpMatches1         Matched map points from KF1 to KF2
 * @param g2oS12             Input/Output Sim3 transformation from KF1 to KF2
 * @param th2                Chi-square threshold for outlier rejection
 * @param bFixScale          Whether to fix the scale parameter
 * @param mAcumHessian       Accumulated Hessian matrix for uncertainty estimation
 * @param bAllPoints         Whether to use all points or only those in KF2
 * @return                   Number of inlier correspondences
 */
int Optimizer::OptimizeSim3(Map* pMap, KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2,
                            const bool bFixScale, Eigen::Matrix<double,7,7> &mAcumHessian, const bool bAllPoints)
{
    // ========== OPTIMIZER SETUP ==========
    // Create dense linear solver for 7-DOF Sim3 optimization
    auto linear_solver = std::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // ========== KEYFRAME POSES ==========
    // Extract camera poses for both keyframes
    const Eigen::Matrix3f R1w = pKF1->GetRotation();
    const Eigen::Vector3f t1w = pKF1->GetTranslation();
    const Eigen::Matrix3f R2w = pKF2->GetRotation();
    const Eigen::Vector3f t2w = pKF2->GetTranslation();

    // ========== SIM3 VERTEX ==========
    // Set up the Sim3 transformation vertex (7-DOF: rotation + translation + scale)
    VertexSim3Expmap* vSim3 = new VertexSim3Expmap();
    vSim3->_fix_scale = bFixScale;           // Fix scale for stereo/RGB-D, optimize for monocular
    vSim3->setEstimate(g2oS12);              // Initial Sim3 estimate
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->pCamera1 = pMap->mpCamera;        // Camera parameters for projection
    vSim3->pCamera2 = pMap->mpCamera;
    optimizer.addVertex(vSim3);

    // ========== MAP POINT VERTICES AND EDGES ==========
    // Prepare data structures for 3D points and projection edges
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    
    vector<EdgeSim3ProjectXYZ*> vpEdges12;          // Edges: KF2 point -> KF1 image
    vector<EdgeInverseSim3ProjectXYZ*> vpEdges21;   // Edges: KF1 point -> KF2 image
    vector<size_t> vnIndexEdge;                     // Edge indices for correspondence tracking
    vector<bool> vbIsInKF2;                         // Whether point is observed in KF2

    // Reserve memory for efficiency
    vnIndexEdge.reserve(2 * N);
    vpEdges12.reserve(2 * N);
    vpEdges21.reserve(2 * N);
    vbIsInKF2.reserve(2 * N);

    const float deltaHuber = sqrt(th2);             // Huber kernel threshold

    // Correspondence statistics
    int nCorrespondences = 0;

    vector<int> vIdsOnlyInKF2;

    // ========== CORRESPONDENCE PROCESSING ==========
    for (int i = 0; i < N; i++)
    {
        if (!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2 * i + 1;                 // Unique vertex ID for point from KF1
        const int id2 = 2 * (i + 1);               // Unique vertex ID for point from KF2

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);  // Feature index in KF2

        Eigen::Vector3f P3D1c;                      // 3D point in KF1 camera frame
        Eigen::Vector3f P3D2c;                      // 3D point in KF2 camera frame

        // Case 1: Both map points are valid
        if (pMP1 && pMP2)
        {
            if (!pMP1->isBad() && !pMP2->isBad())
            {
                // Add 3D point vertex for KF1
                g2o::VertexPointXYZ* vPoint1 = new g2o::VertexPointXYZ();
                Eigen::Vector3f P3D1w = pMP1->GetWorldPos();
                P3D1c = R1w * P3D1w + t1w;             // Transform to camera frame
                vPoint1->setEstimate(P3D1c.cast<double>());
                vPoint1->setId(id1);
                vPoint1->setFixed(true);                // Fix 3D points during optimization
                optimizer.addVertex(vPoint1);

                // Add 3D point vertex for KF2
                g2o::VertexPointXYZ* vPoint2 = new g2o::VertexPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w * P3D2w + t2w;             // Transform to camera frame
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);                // Fix 3D points during optimization
                optimizer.addVertex(vPoint2);
            }
            else
            {
                continue;
            }
        }
        // Case 2: Only pMP2 is available (no corresponding 3D point in KF1)
        else
        {
            // Handle case where 3D position in KF1 doesn't exist
            if (!pMP2->isBad())
            {
                g2o::VertexPointXYZ* vPoint2 = new g2o::VertexPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w * P3D2w + t2w;             // Transform to KF2 camera frame
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);

                vIdsOnlyInKF2.push_back(id2);           // Track points only in KF2
            }
            continue;
        }

        // Skip invalid correspondences
        if (i2 < 0 && !bAllPoints)                      // Point not observed in KF2
            continue;
        if (P3D2c(2) < 0)                               // Point behind camera
            continue;

        nCorrespondences++;

        // ========== PROJECTION EDGES ==========
        // Create bidirectional projection edges for robust Sim3 estimation

        // Edge 1: Project KF2 point to KF1 image (x1 = S12 * X2)
        Eigen::Matrix<double, 2, 1> obs1;
        const KeyPointEx& kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.mPos[0], kpUn1.mPos[1];

        EdgeSim3ProjectXYZ* e12 = new EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));  // 3D point
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));   // Sim3 pose
        e12->setMeasurement(obs1);
        e12->setInformation(Eigen::Matrix2d::Identity());

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);                      // Huber threshold for outlier rejection
        optimizer.addEdge(e12);

        // Edge 2: Project KF1 point to KF2 image (x2 = S21 * X1)
        Eigen::Matrix<double, 2, 1> obs2;
        KeyPointEx kpUn2;
        bool inKF2;
        
        if (i2 >= 0)
        {
            // Point is observed in KF2, use actual keypoint coordinates
            kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.mPos[0], kpUn2.mPos[1];
            inKF2 = true;
        }
        else
        {
            // Point not observed in KF2, project 3D point to image plane
            float invz = 1 / P3D2c(2);
            float x = P3D2c(0) * invz;
            float y = P3D2c(1) * invz;

            obs2 << x, y;
            kpUn2 = KeyPointEx(x, y, 0);
            inKF2 = false;
        }

        EdgeInverseSim3ProjectXYZ* e21 = new EdgeInverseSim3ProjectXYZ();
        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));  // 3D point
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));   // Sim3 pose
        e21->setMeasurement(obs2);
        e21->setInformation(Eigen::Matrix2d::Identity());

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);                      // Huber threshold for outlier rejection
        optimizer.addEdge(e21);

        // Store edges and metadata for outlier detection
        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
        vbIsInKF2.push_back(inKF2);
    }

    // ========== INITIAL OPTIMIZATION ==========
    // Run initial optimization with robust kernels
    optimizer.initializeOptimization();
    optimizer.optimize(5);                              // 5 iterations with robust kernels

    // ========== OUTLIER DETECTION ==========
    // Identify and remove outlier correspondences based on chi-square test
    int nBad = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        // Check if reprojection errors exceed threshold
        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint*>(NULL);  // Mark as outlier
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i] = static_cast<EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i] = static_cast<EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;

            continue;
        }

        // Remove robust kernels for final optimization (more precise)
        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }

    // ========== FINAL OPTIMIZATION ==========
    // Determine number of additional iterations based on outlier count
    int nMoreIterations;
    if (nBad > 0)
        nMoreIterations = 10;                           // More iterations if outliers found
    else
        nMoreIterations = 5;                            // Fewer iterations if clean

    if (nCorrespondences - nBad < 10)                   // Insufficient correspondences
        return 0;

    // Optimize again with inliers only (no robust kernels)
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    // ========== FINAL INLIER COUNT ==========
    // Count final inliers and mark remaining outliers
    int nIn = 0;
    mAcumHessian = Eigen::MatrixXd::Zero(7, 7);         // Reset Hessian accumulation
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();

        // Final outlier check after optimization
        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint*>(NULL);  // Mark as outlier
        }
        else
        {
            nIn++;                                      // Count inlier correspondence
        }
    }

    // ========== RESULT RECOVERY ==========
    // Extract optimized Sim3 transformation
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12 = vSim3_recov->estimate();                   // Update output Sim3

    return nIn;                                         // Return number of inlier correspondences
}

/**
 * @brief 4-DOF Essential Graph Optimization for visual-inertial SLAM
 * @details Optimizes the essential graph using 4-DOF pose vertices (x, y, z, yaw) while
 *          constraining roll and pitch from IMU gravity direction. This function is designed
 *          for visual-inertial systems where gravity provides absolute attitude reference.
 *          Core algorithm: minimize pose graph errors in SE(2)×R representation.
 * 
 * @param pMap               Pointer to the map containing keyframes and map points
 * @param pLoopKF            Loop closure keyframe (fixed as anchor)
 * @param pCurKF             Current keyframe that detected the loop
 * @param NonCorrectedSim3   Original Sim3 poses before loop correction
 * @param CorrectedSim3      Corrected Sim3 poses from loop detection
 * @param LoopConnections    Loop closure connections between keyframes
 */
void Optimizer::OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const KeyFrameAndPose &NonCorrectedSim3,
                                       const KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections)
{
    // ========== OPTIMIZER SETUP ==========
    // Create Levenberg-Marquardt optimizer with Eigen linear solver for 4-DOF optimization
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(solver);

    // Get all map elements
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    // Pre-allocate vectors for Sim3 poses and 4-DOF vertices
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
    vector<VertexPose4DoF*> vpVertices(nMaxKFid + 1);

    const int minFeat = 100;  // Minimum features for valid covisibility connection

    // ========== 4-DOF POSE VERTICES ==========
    // Set up 4-DOF pose vertices for each keyframe (x, y, z, yaw)
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;

        VertexPose4DoF* V4DoF;

        const int nIDi = pKF->mnId;

        // Check if this keyframe has a corrected pose from loop detection
        KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);
        if (it != CorrectedSim3.end())
        {
            // Use corrected Sim3 pose from loop closure
            vScw[nIDi] = it->second;
            const g2o::Sim3 Swc = it->second.inverse();
            Eigen::Matrix3d Rwc = Swc.rotation().toRotationMatrix();
            Eigen::Vector3d twc = Swc.translation();
            V4DoF = new VertexPose4DoF(Rwc, twc, pKF, pMap->mpCamera);
        }
        else
        {
            // Convert SE3 pose to Sim3 (with unit scale) for consistency
            SE3d Tcw = pKF->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);
            vScw[nIDi] = Siw;
            V4DoF = new VertexPose4DoF(pKF, pMap->mpCamera);
        }

        // Fix the loop keyframe as anchor to prevent gauge freedom
        if (pKF == pLoopKF)
            V4DoF->setFixed(true);

        V4DoF->setId(nIDi);
        V4DoF->setMarginalized(false);

        optimizer.addVertex(V4DoF);
        vpVertices[nIDi] = V4DoF;
    }

    // ========== EDGE SETUP ==========
    // Track inserted edges to avoid duplicates
    set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

    // Information matrix for 6-DOF pose constraints (4-DOF vertices but 6-DOF edges)
    // Note: Roll and pitch are constrained by gravity, so we give high weight to rotational components
    Eigen::Matrix<double, 6, 6> matLambda = Eigen::Matrix<double, 6, 6>::Identity();
    matLambda(0, 0) = 1e3;  // High weight for roll constraint
    matLambda(1, 1) = 1e3;  // High weight for pitch constraint
    matLambda(2, 2) = 1e3;  // High weight for yaw (this should be index 2, not 0 again)

    // ========== LOOP CLOSURE EDGES ==========
    // Add loop closure edges detected by the loop detector
    for (map<KeyFrame*, set<KeyFrame*>>::const_iterator mit = LoopConnections.begin(), 
         mend = LoopConnections.end(); mit != mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*>& spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];

        for (set<KeyFrame*>::const_iterator sit = spConnections.begin(), 
             send = spConnections.end(); sit != send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            
            // Skip weak connections except for current-loop pair
            if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && 
                pKF->GetWeight(*sit) < minFeat)
                continue;

            // Compute relative transformation from Sim3 poses
            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sij = Siw * Sjw.inverse();
            
            // Convert to 4x4 transformation matrix
            Eigen::Matrix4d Tij;
            Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
            Tij.block<3, 1>(0, 3) = Sij.translation();
            
            // Create 4-DOF edge constraint
            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->information() = matLambda;
            
            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
        }
    }

    // ========== COVISIBILITY AND SEQUENTIAL EDGES ==========
    // Add edges for keyframes in covisibility graph and sequential connections
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];
        const int nIDi = pKF->mnId;

        // Get pose (use non-corrected if available for consistency)
        g2o::Sim3 Siw;
        KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);
        if (iti != NonCorrectedSim3.end())
            Siw = iti->second;
        else
            Siw = vScw[nIDi];

        // ========== SEQUENTIAL INERTIAL EDGES ==========
        // Add edges between consecutive keyframes (important for trajectory smoothness)
        KeyFrame* prevKF = pKF->mPrevKF;
        if (prevKF)
        {
            int nIDj = prevKF->mnId;

            // Get previous keyframe pose
            g2o::Sim3 Swj;
            KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(prevKF);
            if (itj != NonCorrectedSim3.end())
                Swj = (itj->second).inverse();
            else
                Swj = vScw[nIDj].inverse();

            // Compute relative transformation
            g2o::Sim3 Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
            Tij.block<3, 1>(0, 3) = Sij.translation();
            Tij(3, 3) = 1.0;

            // Create sequential edge
            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // ========== EXPLICIT LOOP EDGES ==========
        // Add edges for explicit loop closures detected by the loop detector
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for (set<KeyFrame*>::const_iterator sit = sLoopEdges.begin(), 
             send = sLoopEdges.end(); sit != send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if (pLKF->mnId < pKF->mnId)  // Avoid duplicate edges
            {
                // Get loop keyframe pose
                g2o::Sim3 Swl;
                KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);
                if (itl != NonCorrectedSim3.end())
                    Swl = itl->second.inverse();
                else
                    Swl = vScw[pLKF->mnId].inverse();

                // Compute relative transformation
                g2o::Sim3 Sil = Siw * Swl;
                Eigen::Matrix4d Til;
                Til.block<3, 3>(0, 0) = Sil.rotation().toRotationMatrix();
                Til.block<3, 1>(0, 3) = Sil.translation();
                Til(3, 3) = 1.0;

                // Create loop edge
                Edge4DoF* e = new Edge4DoF(Til);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                e->information() = matLambda;
                optimizer.addEdge(e);
            }
        }

        // ========== COVISIBILITY GRAPH EDGES ==========
        // Add edges for keyframes connected through visual covisibility
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFrame*>::const_iterator vit = vpConnectedKFs.begin(); 
             vit != vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            
            // Skip invalid connections and avoid redundant edges
            if (pKFn && pKFn != prevKF && pKFn != pKF->mNextKF && !sLoopEdges.count(pKFn))
            {
                if (!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                {
                    // Skip if edge already inserted
                    pair<long unsigned int, long unsigned int> edge_pair = 
                        make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId));
                    if (sInsertedEdges.count(edge_pair))
                        continue;

                    // Get neighbor keyframe pose
                    g2o::Sim3 Swn;
                    KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);
                    if (itn != NonCorrectedSim3.end())
                        Swn = itn->second.inverse();
                    else
                        Swn = vScw[pKFn->mnId].inverse();

                    // Compute relative transformation
                    g2o::Sim3 Sin = Siw * Swn;
                    Eigen::Matrix4d Tin;
                    Tin.block<3, 3>(0, 0) = Sin.rotation().toRotationMatrix();
                    Tin.block<3, 1>(0, 3) = Sin.translation();
                    Tin(3, 3) = 1.0;
                    
                    // Create covisibility edge
                    Edge4DoF* e = new Edge4DoF(Tin);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    e->information() = matLambda;
                    optimizer.addEdge(e);
                }
            }
        }
    }

    // ========== OPTIMIZATION ==========
    // Initialize and run 4-DOF pose graph optimization
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);  // 20 iterations for pose graph optimization

    // Lock map for updating poses
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // ========== POSE RECOVERY ==========
    // Convert optimized 4-DOF poses back to SE3 format
    // Extract rotation and translation from 4-DOF vertices
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        // Extract optimized 4-DOF pose (x, y, z, yaw with fixed roll, pitch)
        VertexPose4DoF* Vi = static_cast<VertexPose4DoF*>(optimizer.vertex(nIDi));
        Eigen::Matrix3d Ri = Vi->estimate().Rcw;
        Eigen::Vector3d ti = Vi->estimate().tcw;

        // Convert to Sim3 with unit scale for consistency
        g2o::Sim3 CorrectedSiw = g2o::Sim3(Ri, ti, 1.0);
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();

        // Update keyframe pose with optimized result
        SE3d Tiw(CorrectedSiw.rotation(), CorrectedSiw.translation());
        pKFi->SetPose(Tiw.cast<float>());
    }

    // ========== MAP POINT CORRECTION ==========
    // Correct map points using optimized keyframe poses
    // Transform to "non-optimized" reference keyframe pose and back with optimized pose
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        MapPoint* pMP = vpMPs[i];
        if (pMP->isBad())
            continue;

        // Get reference keyframe for this map point
        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
        int nIDr = pRefKF->mnId;

        // Apply pose correction to map point coordinates
        g2o::Sim3 Srw = vScw[nIDr];                    // Original reference pose
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];  // Corrected reference pose

        Eigen::Matrix<double, 3, 1> eigP3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        
        pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    // ========== MAP EDGE VALIDATION ==========
    // Remove bad colinearity edges after pose optimization
    vector<MapEdge*> vpMEs = pMap->GetAllMapEdges();
    for (MapEdge* pME : vpMEs)
    {
        if (!pME || pME->isBad())
            continue;
        pME->checkValid();  // Validate edge geometry after pose update
    }

    // ========== COLINEARITY CLEANUP ==========
    // Remove outlier colinear point associations after optimization
    for (MapPoint* pMP : vpMPs)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;
        std::vector<MapColine*> vCLs = pMP->removeColineOutliers();
        for (MapColine* pMC : vCLs)
            pMap->EraseMapColine(pMC);
    }
    
    // Notify map of changes for consistency
    pMap->InfoMapChange();
}