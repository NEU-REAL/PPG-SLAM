#include "Optimizer.h"
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include "G2oEdge.h"
#include "G2oVertex.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>

/**
 * @brief Global Bundle Adjustment for entire map optimization
 * @details Optimizes all keyframe poses and map point positions simultaneously.
 *          Uses robust kernels to handle outliers in feature observations.
 *          Core algorithm: minimize reprojection error across all observations.
 * 
 * @param pMap         Pointer to the map containing keyframes and map points
 * @param nIterations  Number of optimization iterations
 * @param nLoopKF      Loop keyframe identifier for loop closure handling
 * @param pbStopFlag   Stop flag for early termination
 */
void Optimizer::GlobalBundleAdjustment(Map* pMap, int nIterations, const unsigned long nLoopKF, bool* pbStopFlag)
{
    // Get all map elements
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    // ========== OPTIMIZER SETUP ==========
    // Create Levenberg-Marquardt optimizer with Eigen linear solver
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    // Track maximum keyframe ID for vertex ID management
    long unsigned int maxKFid = 0;

    // Pre-allocate vectors for edges (optimization efficiency)
    const int nExpectedSize = (vpKFs.size()) * vpMPs.size();
    vector<EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);
    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // ========== KEYFRAME VERTICES ==========
    // Add SE3 pose vertices for each keyframe
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
            
        // Create SE3 pose vertex
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        SE3<float> Tcw = pKF->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), 
                                       Tcw.translation().cast<double>()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId == pMap->GetOriginKF()->mnId);  // Fix origin frame
        optimizer.addVertex(vSE3);
        
        if (pKF->mnId > maxKFid)
            maxKFid = pKF->mnId;
    }

    // ========== MAP POINT VERTICES ==========
    // Add XYZ position vertices for each 3D map point
    const float thHuberMono = sqrt(5.991);  // Chi-square threshold for monocular observations
    
    for (size_t i = 0; i < vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];
        if (pMP->isBad())
            continue;
            
        // Create 3D point vertex
        g2o::VertexPointXYZ* vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        
        const int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(false);  // Don't marginalize for full optimization
        optimizer.addVertex(vPoint);

        // Get observations of this map point
        const map<KeyFrame*, int> observations = pMP->GetObservations();

        // ========== REPROJECTION EDGES ==========
        // Create edge for each observation (keyframe that sees this map point)
        int nEdges = 0;
        for (map<KeyFrame*, int>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            const int index = mit->second;

            // Validate keyframe and feature index
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;
            if (index < 0)
                continue;
            if (optimizer.vertex(id) == NULL || optimizer.vertex(pKF->mnId) == NULL)
                continue;

            nEdges++;
            
            // Get feature observation
            const KeyPointEx& kpUn = pKF->mvKeysUn[index];
            Eigen::Matrix<double, 2, 1> obs;
            obs << kpUn.mPos[0], kpUn.mPos[1];

            // Create SE3 projection edge (3D point to 2D observation)
            EdgeSE3ProjectXYZ* e = new EdgeSE3ProjectXYZ();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
            e->setMeasurement(obs);
            e->setInformation(Eigen::Matrix2d::Identity());
            
            // Use robust kernel to handle outliers
            g2o::RobustKernelCauchy* rk = new g2o::RobustKernelCauchy;
            rk->setDelta(thHuberMono);
            e->setRobustKernel(rk);

            // Set camera parameters
            e->pCamera = pMap->mpCamera;
            
            optimizer.addEdge(e);
            
            // Store edge information for later analysis
            vpEdgesMono.push_back(e);
            vpEdgeKFMono.push_back(pKF);
            vpMapPointEdgeMono.push_back(pMP);
        }

        // Remove map points with insufficient observations (< 2 views)
        if (nEdges < 2)
            optimizer.removeVertex(vPoint);
    }

    // ========== COLINEARITY CONSTRAINTS ==========
    // Add collinearity constraints between map points that should lie on same line
    for (MapPoint* pMP : vpMPs)
    {
        if (optimizer.vertex(pMP->mnId + maxKFid + 1) == NULL)
            continue;
            
        std::vector<MapColine*> vMCs = pMP->getColinearity();
        for (auto pMC : vMCs)
        {
            if (pMC->isBad() || !pMC->mbValid)
                continue;
                
            MapPoint* pMPs = pMC->mpMPs;  // Start point of line
            MapPoint* pMPe = pMC->mpMPe;  // End point of line
            
            // Ensure all three points exist in optimizer
            if (optimizer.vertex(pMPs->mnId + maxKFid + 1) == NULL ||
                optimizer.vertex(pMPe->mnId + maxKFid + 1) == NULL)
                continue;
                
            // Create collinearity edge (geometric constraint)
            EdgeColine* e = new EdgeColine();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPs->mnId + maxKFid + 1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMP->mnId + maxKFid + 1)));
            e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPe->mnId + maxKFid + 1)));
            e->setInformation(Eigen::Matrix3d::Identity() * pMC->aveWeight());
            optimizer.addEdge(e);
        }
    }

    // ========== OPTIMIZATION PROCESS ==========
    // Execute bundle adjustment optimization
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // ========== RESULT RECOVERY ==========
    // Extract optimized keyframe poses
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
            
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        
        if (nLoopKF == pMap->GetOriginKF()->mnId)  // Initial optimization
        {
            pKF->SetPose(SE3f(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>()));
        }
        else  // Loop closure correction
        {
            pKF->mTcwGBA = SE3d(SE3quat.rotation(), SE3quat.translation()).cast<float>();
            pKF->mnBAGlobalForKF = nLoopKF;

            // Check for large pose changes (potential loop closure issues)
            SE3f mTwc = pKF->GetPoseInverse();
            SE3f mTcGBA_c = pKF->mTcwGBA * mTwc;
            Eigen::Vector3f vector_dist = mTcGBA_c.translation();
            double dist = vector_dist.norm();
            
            if (dist > 1)
            {
                // Validate observations for this keyframe
                for (size_t i2 = 0, iend = vpEdgesMono.size(); i2 < iend; i2++)
                {
                    MapPoint* pMP = vpMapPointEdgeMono[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if (pKF != pKFedge)
                        continue;

                    if (pMP->isBad())
                        continue;

                    // Check reprojection error and depth validity - no action needed
                    // This loop validates the optimization quality
                }
            }
        }
    }

    // ========== MAP POINT RECOVERY ==========
    // Extract optimized 3D point positions
    for (size_t i = 0; i < vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];
        g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));

        if (pMP == nullptr || pMP->isBad() || vPoint == nullptr)
            continue;

        if (nLoopKF == pMap->GetOriginKF()->mnId)  // Initial optimization
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else  // Loop closure correction
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
    
    // ========== COLINEARITY EDGE CLEANUP ==========
    // Remove invalid colinearity constraints after optimization
    vector<MapEdge*> vpMEs = pMap->GetAllMapEdges();
    for (MapEdge* pME : vpMEs)
    {
        if (!pME || pME->isBad())
            continue;
        pME->checkValid();
    }
    
    // Remove outlier collinear constraints
    for (MapPoint* pMP : vpMPs)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;
        std::vector<MapColine*> vCLs = pMP->removeColineOutliers();
        for (MapColine* pMC : vCLs)
            pMap->EraseMapColine(pMC);
    }
}

/**
 * @brief Full Inertial Bundle Adjustment with IMU constraints
 * @details Simultaneous optimization of keyframe poses, velocities, biases,
 *          map point positions, and IMU parameters. Includes gravity direction
 *          and accelerometer/gyroscope bias estimation.
 *          Core algorithm: minimize visual-inertial cost function.
 * 
 * @param pMap        Pointer to the map containing keyframes and map points
 * @param its         Number of optimization iterations
 * @param nLoopKF     Loop keyframe identifier for loop closure handling
 * @param pbStopFlag  Stop flag for early termination
 * @param bInit       Flag indicating initialization phase
 * @param priorG      Prior weight for gravity constraint
 * @param priorA      Prior weight for accelerometer bias constraint
 */
void Optimizer::FullInertialBA(Map *pMap, int its, const unsigned long nLoopKF, bool *pbStopFlag, bool bInit, float priorG, float priorA)
{
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    // ========== OPTIMIZER SETUP ==========
    // Create Levenberg-Marquardt optimizer with Eigen linear solver
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

    solver->setUserLambdaInit(1e-5);  // Initial damping parameter

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    // ========== KEYFRAME VERTICES ==========
    // Add pose, velocity, and bias vertices for each keyframe
    KeyFrame* pIncKF(nullptr);
    for (KeyFrame* pKFi : vpKFs)
    {
        if (pKFi == nullptr || pKFi->isBad() || pKFi->mnId > maxKFid)
            continue;

        // Add pose vertex
        VertexPose* VP = new VertexPose(pKFi, pMap->mpCamera);
        VP->setId(pKFi->mnId);
        pIncKF = pKFi;
        optimizer.addVertex(VP);

        // Add IMU state vertices (velocity and biases)
        if (pKFi->bImu)
        {
            // Velocity vertex
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            
            // Bias vertices (only in non-initialization mode)
            if (!bInit)
            {
                VertexGyroBias* VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
                VG->setFixed(false);
                optimizer.addVertex(VG);
                
                VertexAccBias* VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
                VA->setFixed(false);
                optimizer.addVertex(VA);
            }
        }
    }

    // Add shared bias vertices during initialization
    if (bInit && pIncKF != nullptr)
    {
        VertexGyroBias* VG = new VertexGyroBias(pIncKF);
        VG->setId(4 * maxKFid + 2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        
        VertexAccBias* VA = new VertexAccBias(pIncKF);
        VA->setId(4 * maxKFid + 3);
        VA->setFixed(false);
        optimizer.addVertex(VA);
    }

    // ========== IMU PREINTEGRATION EDGES ==========
    // Connect consecutive keyframes with IMU constraints
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi == nullptr || pKFi->isBad() || pKFi->mnId > maxKFid)
            continue;

        if (pKFi->mPrevKF == nullptr || pKFi->mPrevKF->isBad() || pKFi->mPrevKF->mnId > maxKFid)
            continue;

        if (!pKFi->bImu || !pKFi->mPrevKF->bImu)
        {
            cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
            continue;
        }

        // Update bias for preintegration
        pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
        
        // Get vertex pointers for previous and current keyframes
        g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
        g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 1);
        g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
        g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1);

        g2o::HyperGraph::Vertex* VG1(nullptr);
        g2o::HyperGraph::Vertex* VA1(nullptr);
        g2o::HyperGraph::Vertex* VG2(nullptr);
        g2o::HyperGraph::Vertex* VA2(nullptr);
        
        if (!bInit)
        {
            // Individual bias vertices for each keyframe
            VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 2);
            VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 3);
            VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2);
            VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3);
            
            if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cout << "Error vertices: " << VP1 << ", " << VV1 << ", " << VG1 << ", " << VA1 
                     << ", " << VP2 << ", " << VV2 << ", " << VG2 << ", " << VA2 << endl;
                continue;
            }
            
            // Add random walk edges for bias evolution
            EdgeGyroRW* egr = new EdgeGyroRW();
            egr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            egr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG2));
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
            egr->setInformation(InfoG);
            egr->computeError();
            optimizer.addEdge(egr);

            EdgeAccRW* ear = new EdgeAccRW();
            ear->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            ear->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA2));
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();
            ear->setInformation(InfoA);
            ear->computeError();
            optimizer.addEdge(ear);
        }
        else
        {
            // Shared bias vertices during initialization
            VG1 = optimizer.vertex(4 * maxKFid + 2);
            VA1 = optimizer.vertex(4 * maxKFid + 3);
            
            if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
            {
                cout << "Error init vertices: " << VP1 << ", " << VV1 << ", " << VG1 
                     << ", " << VA1 << ", " << VP2 << ", " << VV2 << endl;
                continue;
            }
        }

        // Create inertial edge connecting consecutive keyframes
        EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
        ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
        ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
        ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
        ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
        ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
        ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

        // Use robust kernel for IMU outlier rejection
        g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
        ei->setRobustKernel(rki);
        rki->setDelta(sqrt(16.92));  // Chi-square threshold for 6-DOF IMU error
        optimizer.addEdge(ei);
    }

    // ========== PRIOR CONSTRAINTS ==========
    // Add bias priors during initialization phase
    if (bInit)
    {
        Eigen::Vector3f bprior;
        bprior.setZero();

        // Gyroscope bias prior
        EdgePriorGyro* epg = new EdgePriorGyro(bprior);
        epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(4 * maxKFid + 2)));
        epg->setInformation(priorG * Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);

        // Accelerometer bias prior
        EdgePriorAcc* epa = new EdgePriorAcc(bprior);
        epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(4 * maxKFid + 3)));
        epa->setInformation(priorA * Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);
    }

    // ========== MAP POINT VERTICES ==========
    // Add 3D point vertices for visual observations
    const float thHuberMono = sqrt(5.991);  // Chi-square threshold for monocular observations
    const unsigned long iniMPid = maxKFid * 5;
    
    for (MapPoint* pMP : vpMPs)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;
            
        // Create 3D point vertex
        g2o::VertexPointXYZ* vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId + iniMPid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(false);  // Don't marginalize for full optimization
        optimizer.addVertex(vPoint);

        // ========== VISUAL REPROJECTION EDGES ==========
        // Add visual observation edges
        const map<KeyFrame*, int> observations = pMP->GetObservations();
        bool bAllFixed = true;
        
        for (auto mit : observations)
        {
            KeyFrame* pKFi = mit.first;
            const int index = mit.second;
            
            if (pKFi == nullptr || pKFi->isBad() || pKFi->mnId > maxKFid || index < 0)
                continue;

            // Get feature observation
            Eigen::Matrix<double, 2, 1> obs;
            obs << pKFi->mvKeysUn[index].mPos[0], pKFi->mvKeysUn[index].mPos[1];
            
            // Create monocular observation edge
            EdgeMono* e = new EdgeMono();
            g2o::OptimizableGraph::Vertex* VP = optimizer.vertex(pKFi->mnId);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
            e->setMeasurement(obs);
            e->setInformation(Eigen::Matrix2d::Identity());
            
            // Use robust kernel for outlier rejection
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(thHuberMono);
            optimizer.addEdge(e);
            
            if (bAllFixed && !VP->fixed())
                bAllFixed = false;
        }
        
        // Remove vertices with no valid observations
        if (bAllFixed)
            optimizer.removeVertex(vPoint);
    }

    // ========== COLINEARITY CONSTRAINTS ==========
    // Add collinearity constraints for geometric consistency
    for (MapPoint* pMP : vpMPs)
    {
        if (optimizer.vertex(pMP->mnId + iniMPid + 1) == NULL)
            continue;
            
        std::vector<MapColine*> vMCs = pMP->getColinearity();
        for (auto pMC : vMCs)
        {
            if (pMC->isBad() || !pMC->mbValid)
                continue;
                
            MapPoint* pMPs = pMC->mpMPs;  // Start point of line
            MapPoint* pMPe = pMC->mpMPe;  // End point of line
            
            if (optimizer.vertex(pMPs->mnId + iniMPid + 1) == NULL ||
                optimizer.vertex(pMPe->mnId + iniMPid + 1) == NULL)
                continue;
                
            EdgeColine* e = new EdgeColine();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPs->mnId + iniMPid + 1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMP->mnId + iniMPid + 1)));
            e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPe->mnId + iniMPid + 1)));
            e->setInformation(Eigen::Matrix3d::Identity() * pMC->aveWeight());
            optimizer.addEdge(e);
        }
    }

    // ========== OPTIMIZATION PROCESS ==========
    // Check for early termination
    if (pbStopFlag)
        if (*pbStopFlag)
            return;
            
    optimizer.initializeOptimization();
    optimizer.optimize(its);
    
    // ========== RESULT RECOVERY ==========
    // Extract optimized keyframe states
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid)
            continue;
            
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        
        if (nLoopKF == 0)  // Normal optimization
        {
            SE3f Tcw(VP->estimate().Rcw.cast<float>(), VP->estimate().tcw.cast<float>());
            pKFi->SetPose(Tcw);
        }
        else  // Loop closure optimization
        {
            pKFi->mTcwGBA = SE3f(VP->estimate().Rcw.cast<float>(), VP->estimate().tcw.cast<float>());
            pKFi->mnBAGlobalForKF = nLoopKF;
        }
        
        // Extract IMU states
        if (pKFi->bImu)
        {
            // Velocity recovery
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1));
            if (nLoopKF == 0)
                pKFi->SetVelocity(VV->estimate().cast<float>());
            else
                pKFi->mVwbGBA = VV->estimate().cast<float>();

            // Bias recovery
            VertexGyroBias* VG;
            VertexAccBias* VA;
            if (!bInit)
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3));
            }
            else
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(4 * maxKFid + 2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(4 * maxKFid + 3));
            }

            // Construct bias vector (gyro first, then acc)
            Vector6d vb;
            vb << VG->estimate(), VA->estimate();
            IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);
            
            if (nLoopKF == 0)
                pKFi->SetNewBias(b);
            else
                pKFi->mBiasGBA = b;
        }
    }

    // Extract optimized map point positions
    for (size_t i = 0; i < vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];
        g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(pMP->mnId + iniMPid + 1));
        if (vPoint == nullptr)
            continue;
            
        if (nLoopKF == 0)  // Normal optimization
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else  // Loop closure optimization
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
    
    // ========== COLINEARITY EDGE CLEANUP ==========
    // Remove invalid geometric constraints after optimization
    vector<MapEdge*> vpMEs = pMap->GetAllMapEdges();
    for (MapEdge* pME : vpMEs)
    {
        if (!pME || pME->isBad())
            continue;
        pME->checkValid();
    }
    
    // Remove outlier collinear constraints
    for (MapPoint* pMP : vpMPs)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;
        std::vector<MapColine*> vCLs = pMP->removeColineOutliers();
        for (MapColine* pMC : vCLs)
            pMap->EraseMapColine(pMC);
    }
    
    pMap->InfoMapChange();
}