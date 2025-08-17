#include "Optimizer.h"
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include "G2oEdge.h"
#include "G2oVertex.h"
#include <mutex>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>


/**
 * @brief Pose-only Bundle Adjustment for frame tracking optimization
 * 
 * Algorithm principle:
 * 1. Optimize only the camera pose while keeping 3D points fixed
 * 2. Use reprojection errors as optimization constraints
 * 3. Apply robust kernels to handle outliers iteratively
 * 4. Perform multiple rounds with outlier rejection to improve robustness
 * 
 * @param pFrame Input frame with initial pose estimate
 * @return Number of inlier observations after optimization
 */
int Optimizer::PoseOptimization(Frame *pFrame)
{
    // 1. Setup g2o optimizer with dense linear solver for pose-only optimization
    auto linear_solver = std::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver)));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;

    // 2. Add camera pose vertex (SE3 transformation)
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    SE3<float> Tcw = pFrame->GetPose();
    vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
    vSE3->setId(0);
    vSE3->setFixed(false);  // Pose is optimizable
    optimizer.addVertex(vSE3);

    // 3. Setup data structures for monocular edges
    const int N = pFrame->N;
    vector<EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    // Huber kernel threshold (95% confidence for chi-squared distribution with 2 DOF)
    const float deltaMono = sqrt(5.991);

    // 4. Add reprojection error edges for valid map points
    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if (pMP)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                // Create observation measurement
                Eigen::Matrix<double, 2, 1> obs;
                const KeyPointEx &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.mPos[0], kpUn.mPos[1];

                // Create reprojection error edge
                EdgeSE3ProjectXYZOnlyPose* e = new EdgeSE3ProjectXYZOnlyPose();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                e->setInformation(Eigen::Matrix2d::Identity());

                // Add robust kernel for outlier rejection
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                // Set edge parameters
                e->pCamera = pFrame->mpCamera;
                e->Xw = pMP->GetWorldPos().cast<double>();

                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
        }
    }

    // Early exit if insufficient correspondences
    if (nInitialCorrespondences < 3)
        return 0;

    // 5. Iterative optimization with outlier rejection
    // Perform 4 rounds of optimization with consistent chi-squared thresholds
    const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};  // 95% confidence
    const int its[4] = {10, 10, 10, 10};  // Iterations per round

    int nBad = 0;
    for (size_t it = 0; it < 4; it++)
    {
        // Reset pose estimate before each optimization round
        Tcw = pFrame->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));

        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        // 6. Classify observations as inliers/outliers based on chi-squared test
        nBad = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];
            const size_t idx = vnIndexEdgeMono[i];

            // Compute error for outliers to update chi-squared value
            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            // Apply chi-squared test for outlier detection
            if (chi2 > chi2Mono[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);  // Exclude from next optimization
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);  // Include in next optimization
            }

            // Remove robust kernel in final round for better convergence
            if (it == 2)
                e->setRobustKernel(0);
        }

        // Early termination if too few edges remain
        if (optimizer.edges().size() < 10)
            break;
    }    

    // 7. Recover optimized pose and update frame
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    SE3<float> pose(SE3quat_recov.rotation().cast<float>(), SE3quat_recov.translation().cast<float>());
    pFrame->SetPose(pose);

    // Return number of inlier correspondences
    return nInitialCorrespondences - nBad;
}

/**
 * @brief Visual-Inertial Bundle Adjustment with previous frame constraints
 * 
 * Algorithm principle:
 * 1. Jointly optimize current frame pose, velocity, and IMU biases
 * 2. Use IMU preintegration constraints between consecutive frames
 * 3. Apply visual reprojection constraints for robust pose estimation
 * 4. Include prior information from previous frame marginalization
 * 5. Perform outlier rejection with adaptive chi-squared thresholds
 * 
 * @param pFrame Current frame to optimize
 * @param pMap Map containing camera parameters
 * @param bRecInit Whether this is a relocalization/reinitialization
 * @return Number of inlier visual observations after optimization
 */
int Optimizer::PoseInertialOptimizationLastFrame(Frame *pFrame, Map *pMap, bool bRecInit)
{
    // 1. Setup g2o optimizer with Gauss-Newton algorithm for VI-SLAM
    auto linear_solver = std::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int nInitialMonoCorrespondences = 0;
    int nInitialCorrespondences = 0;

    // 2. Add current frame state vertices (pose, velocity, IMU biases)
    VertexPose* VP = new VertexPose(pFrame, pMap->mpCamera);
    VP->setId(0);
    VP->setFixed(false);  // Current pose is optimizable
    optimizer.addVertex(VP);
    
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);  // Current velocity is optimizable
    optimizer.addVertex(VV);
    
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);  // Gyroscope bias is optimizable
    optimizer.addVertex(VG);
    
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);  // Accelerometer bias is optimizable
    optimizer.addVertex(VA);

    // 3. Setup visual reprojection edges for map point observations
    const int N = pFrame->N;
    vector<EdgeMonoOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    // Huber kernel threshold (95% confidence for chi-squared distribution with 2 DOF)
    const float thHuberMono = sqrt(5.991);

    // 4. Add visual reprojection error edges
    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if (pMP)
            {
                nInitialMonoCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                // Create observation measurement
                Eigen::Matrix<double, 2, 1> obs;
                const KeyPointEx &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.mPos[0], kpUn.mPos[1];

                // Create monocular reprojection error edge
                EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos());
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                e->setMeasurement(obs);
                e->setInformation(Eigen::Matrix2d::Identity());

                // Add robust kernel for outlier rejection
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);

                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
        }
    }

    nInitialCorrespondences = nInitialMonoCorrespondences;

    // 5. Add previous frame state vertices
    Frame* pFp = pFrame->mpPrevFrame;

    VertexPose* VPk = new VertexPose(pFp, pMap->mpCamera);
    VPk->setId(4);
    VPk->setFixed(false);  // Previous pose is optimizable
    optimizer.addVertex(VPk);
    
    VertexVelocity* VVk = new VertexVelocity(pFp);
    VVk->setId(5);
    VVk->setFixed(false);  // Previous velocity is optimizable
    optimizer.addVertex(VVk);
    
    VertexGyroBias* VGk = new VertexGyroBias(pFp);
    VGk->setId(6);
    VGk->setFixed(false);  // Previous gyro bias is optimizable
    optimizer.addVertex(VGk);
    
    VertexAccBias* VAk = new VertexAccBias(pFp);
    VAk->setId(7);
    VAk->setFixed(false);  // Previous acc bias is optimizable
    optimizer.addVertex(VAk);

    // 6. Add IMU preintegration constraint between frames
    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);
    ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VPk));  // Previous pose
    ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VVk));  // Previous velocity
    ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGk));  // Previous gyro bias
    ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VAk));  // Previous acc bias
    ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));   // Current pose
    ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV));   // Current velocity
    optimizer.addEdge(ei);

    // 7. Add gyroscope bias random walk constraint
    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGk));
    egr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    // 8. Add accelerometer bias random walk constraint
    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VAk));
    ear->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    // 9. Add prior constraint from previous frame marginalization
    if (!pFp->mpcpi)
        std::cerr << "Warning: Previous frame prior constraint does not exist! Frame ID: " 
                  << to_string(pFp->mnId) << std::endl;

    EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFp->mpcpi);
    ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VPk));
    ep->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VVk));
    ep->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGk));
    ep->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VAk));
    
    // Add robust kernel for prior constraint
    g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
    ep->setRobustKernel(rkp);
    rkp->setDelta(5.0);
    optimizer.addEdge(ep);

    // 10. Iterative optimization with adaptive outlier rejection
    const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};  // Chi-squared thresholds (95% confidence)
    const int its[4] = {10, 10, 10, 10};  // Iterations per round

    int nBad = 0;

    for (size_t it = 0; it < 4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        // Reset counters for this iteration
        nBad = 0;
        
        // Adaptive threshold for close points (more lenient)
        float chi2close = 1.5 * chi2Mono[it];

        // 11. Classify visual observations as inliers/outliers
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];
            const size_t idx = vnIndexEdgeMono[i];
            
            // Check if map point is close to camera (depth < 10m)
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth < 10.0f;

            // Compute error for outliers to update chi-squared value
            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            // Apply adaptive chi-squared test based on point depth and positivity
            if ((chi2 > chi2Mono[it] && !bClose) || 
                (bClose && chi2 > chi2close) || 
                !e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);  // Exclude from next optimization
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);  // Include in next optimization
            }

            // Remove robust kernel in third round for better convergence
            if (it == 2)
                e->setRobustKernel(0);
        }

        // Early termination if too few edges remain
        if (optimizer.edges().size() < 10)
        {
            break;
        }
    }


    // 12. Recovery phase for insufficient inliers (relaxed threshold)
    int nInliers = nInitialCorrespondences - nBad;
    if ((nInliers < 30) && !bRecInit)
    {
        nBad = 0;
        const float chi2MonoOut = 18.0f;  // More lenient threshold for recovery
        
        for (size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            EdgeMonoOnlyPose* e1 = vpEdgesMono[i];
            e1->computeError();
            
            if (e1->chi2() < chi2MonoOut)
                pFrame->mvbOutlier[idx] = false;
            else
                nBad++;
        }
    }

    // 13. Recover optimized pose, velocity and IMU biases
    pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), 
                               VP->estimate().twb.cast<float>(), 
                               VV->estimate().cast<float>());
    
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);

    // 14. Construct Hessian matrix for marginalization
    // Build 30x30 Hessian matrix for [previous_states(15), current_states(15)]
    Eigen::Matrix<double, 30, 30> H;
    H.setZero();

    // Add inertial constraint Hessian (24x24 for poses, velocities, biases)
    H.block<24, 24>(0, 0) += ei->GetHessian();

    // Add gyroscope bias random walk Hessian
    Eigen::Matrix<double, 6, 6> Hgr = egr->GetHessian();
    H.block<3, 3>(9, 9) += Hgr.block<3, 3>(0, 0);
    H.block<3, 3>(9, 24) += Hgr.block<3, 3>(0, 3);
    H.block<3, 3>(24, 9) += Hgr.block<3, 3>(3, 0);
    H.block<3, 3>(24, 24) += Hgr.block<3, 3>(3, 3);

    // Add accelerometer bias random walk Hessian
    Eigen::Matrix<double, 6, 6> Har = ear->GetHessian();
    H.block<3, 3>(12, 12) += Har.block<3, 3>(0, 0);
    H.block<3, 3>(12, 27) += Har.block<3, 3>(0, 3);
    H.block<3, 3>(27, 12) += Har.block<3, 3>(3, 0);
    H.block<3, 3>(27, 27) += Har.block<3, 3>(3, 3);

    // Add prior constraint Hessian
    H.block<15, 15>(0, 0) += ep->GetHessian();

    // Add visual constraint Hessians for inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        EdgeMonoOnlyPose* e = vpEdgesMono[i];
        const size_t idx = vnIndexEdgeMono[i];

        if (!pFrame->mvbOutlier[idx])
        {
            H.block<6, 6>(15, 15) += e->GetHessian();
        }
    }

    // 15. Marginalize previous frame states and create new prior
    H = Marginalize(H, 0, 14);  // Marginalize first 15 variables (previous states)

    // Create new prior constraint for current frame
    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb, VP->estimate().twb, 
                                          VV->estimate(), VG->estimate(), VA->estimate(),
                                          H.block<15, 15>(15, 15));
    
    // Clean up previous frame prior constraint
    delete pFp->mpcpi;
    pFp->mpcpi = NULL;

    return nInitialCorrespondences - nBad;
}


/**
 * @brief Visual-Inertial Bundle Adjustment with fixed keyframe constraints
 * 
 * Algorithm principle:
 * 1. Optimize current frame pose, velocity, and IMU biases against fixed keyframe
 * 2. Use IMU preintegration constraints between keyframe and current frame
 * 3. Apply visual reprojection constraints for robust pose estimation
 * 4. Keyframe states are fixed to provide stable reference frame
 * 5. Perform outlier rejection with progressive chi-squared thresholds
 * 
 * @param pFrame Current frame to optimize
 * @param pMap Map containing camera parameters
 * @param bRecInit Whether this is a relocalization/reinitialization
 * @return Number of inlier visual observations after optimization
 */
int Optimizer::PoseInertialOptimizationLastKeyFrame(Frame *pFrame, Map *pMap, bool bRecInit)
{
    // 1. Setup g2o optimizer with Gauss-Newton algorithm for VI-SLAM
    auto linear_solver = std::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(solver);

    int nInitialMonoCorrespondences = 0;
    int nInitialCorrespondences = 0;

    // 2. Add current frame state vertices (pose, velocity, IMU biases)
    VertexPose* VP = new VertexPose(pFrame, pMap->mpCamera);
    VP->setId(0);
    VP->setFixed(false);  // Current pose is optimizable
    optimizer.addVertex(VP);
    
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);  // Current velocity is optimizable
    optimizer.addVertex(VV);
    
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);  // Gyroscope bias is optimizable
    optimizer.addVertex(VG);
    
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);  // Accelerometer bias is optimizable
    optimizer.addVertex(VA);

    // 3. Setup visual reprojection edges for map point observations
    const int N = pFrame->N;
    vector<EdgeMonoOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    // Huber kernel threshold (95% confidence for chi-squared distribution with 2 DOF)
    const float thHuberMono = sqrt(5.991);

    // 4. Add visual reprojection error edges
    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if (pMP)
            {
                nInitialMonoCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                // Create observation measurement
                Eigen::Matrix<double, 2, 1> obs;
                const KeyPointEx &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.mPos[0], kpUn.mPos[1];

                // Create monocular reprojection error edge
                EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos());
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                e->setMeasurement(obs);
                e->setInformation(Eigen::Matrix2d::Identity());

                // Add robust kernel for outlier rejection
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);

                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
        }
    }
    
    nInitialCorrespondences = nInitialMonoCorrespondences;

    // 5. Add fixed keyframe state vertices (reference frame)
    KeyFrame* pKF = pFrame->mpLastKeyFrame;
    
    VertexPose* VPk = new VertexPose(pKF, pMap->mpCamera);
    VPk->setId(4);
    VPk->setFixed(true);  // Keyframe pose is fixed (provides reference)
    optimizer.addVertex(VPk);
    
    VertexVelocity* VVk = new VertexVelocity(pKF);
    VVk->setId(5);
    VVk->setFixed(true);  // Keyframe velocity is fixed
    optimizer.addVertex(VVk);
    
    VertexGyroBias* VGk = new VertexGyroBias(pKF);
    VGk->setId(6);
    VGk->setFixed(true);  // Keyframe gyro bias is fixed
    optimizer.addVertex(VGk);
    
    VertexAccBias* VAk = new VertexAccBias(pKF);
    VAk->setId(7);
    VAk->setFixed(true);  // Keyframe acc bias is fixed
    optimizer.addVertex(VAk);

    // 6. Add IMU preintegration constraint between keyframe and current frame
    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegrated);
    ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VPk));  // Keyframe pose
    ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VVk));  // Keyframe velocity
    ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGk));  // Keyframe gyro bias
    ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VAk));  // Keyframe acc bias
    ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));   // Current pose
    ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV));   // Current velocity
    optimizer.addEdge(ei);

    // 7. Add gyroscope bias random walk constraint
    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGk));
    egr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    // 8. Add accelerometer bias random walk constraint
    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VAk));
    ear->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    // 9. Iterative optimization with progressive outlier rejection
    // Progressive chi-squared thresholds (more lenient initially, then stricter)
    const float chi2Mono[4] = {12.0f, 7.5f, 5.991f, 5.991f};  // Progressive thresholds
    const int its[4] = {10, 10, 10, 10};  // Iterations per round

    int nBad = 0;

    for (size_t it = 0; it < 4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        // Reset counters for this iteration
        nBad = 0;
        
        // Adaptive threshold for close points (more lenient)
        float chi2close = 1.5f * chi2Mono[it];

        // 10. Classify visual observations as inliers/outliers
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];
            const size_t idx = vnIndexEdgeMono[i];

            // Compute error for outliers to update chi-squared value
            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();
            
            // Check if map point is close to camera (depth < 10m)
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth < 10.0f;

            // Apply adaptive chi-squared test based on point depth and positivity
            if ((chi2 > chi2Mono[it] && !bClose) || 
                (bClose && chi2 > chi2close) || 
                !e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);  // Exclude from next optimization
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);  // Include in next optimization
            }

            // Remove robust kernel in third round for better convergence
            if (it == 2)
                e->setRobustKernel(0);
        }

        // Early termination if too few edges remain
        if (optimizer.edges().size() < 10)
            break;
    }

    // 11. Recovery phase for insufficient inliers (relaxed threshold)
    int nInliers = nInitialCorrespondences - nBad;
    if ((nInliers < 30) && !bRecInit)
    {
        nBad = 0;
        const float chi2MonoOut = 18.0f;  // More lenient threshold for recovery
        
        for (size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            EdgeMonoOnlyPose* e1 = vpEdgesMono[i];
            e1->computeError();
            
            if (e1->chi2() < chi2MonoOut)
                pFrame->mvbOutlier[idx] = false;
            else
                nBad++;
        }
    }

    // 12. Recover optimized pose, velocity and IMU biases
    pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), 
                               VP->estimate().twb.cast<float>(), 
                               VV->estimate().cast<float>());
    
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);

    // 13. Construct simplified Hessian matrix (keyframe-based marginalization)
    // Build 15x15 Hessian matrix for current frame states only
    Eigen::Matrix<double, 15, 15> H;
    H.setZero();

    // Add inertial constraint Hessian (simplified for keyframe reference)
    H.block<9, 9>(0, 0) += ei->GetHessian2();
    H.block<3, 3>(9, 9) += egr->GetHessian2();
    H.block<3, 3>(12, 12) += ear->GetHessian2();

    // Add visual constraint Hessians for inlier observations only
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        EdgeMonoOnlyPose* e = vpEdgesMono[i];
        const size_t idx = vnIndexEdgeMono[i];

        if (!pFrame->mvbOutlier[idx])
        {
            H.block<6, 6>(0, 0) += e->GetHessian();
        }
    }

    // Create new prior constraint for current frame (no marginalization needed)
    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb, VP->estimate().twb, 
                                          VV->estimate(), VG->estimate(), VA->estimate(), H);

    return nInitialCorrespondences - nBad;
}