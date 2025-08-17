#include "Optimizer.h"
#include <iostream>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include "G2oEdge.h"
#include "G2oVertex.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>

using namespace std;

/**
 * @brief Inertial optimization for gravity direction and scale estimation
 * 
 * Algorithm principle:
 * 1. Fix all keyframe poses, velocities and biases, optimize only gravity direction and scale
 * 2. Build graph optimization problem using IMU preintegration constraints
 * 3. Optimize gravity direction Rwg and scale using Gauss-Newton algorithm
 * 4. Used for gravity alignment and scale recovery in SLAM system initialization
 * 
 * @param pMap      Map pointer containing all keyframes
 * @param Rwg       Gravity direction rotation matrix (world to gravity coordinate system)
 * @param scale     Monocular scale factor
 */
void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale)
{
    const int iterations = 10;
    const long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // 1. Configure optimizer - using Gauss-Newton algorithm
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // 2. Add keyframe vertices (all variables are fixed)
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid)
            continue;

        // Pose vertex (fixed)
        VertexPose *VP = new VertexPose(pKFi, pMap->mpCamera);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        // Velocity vertex (fixed)
        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid + 1 + pKFi->mnId);
        VV->setFixed(true);
        optimizer.addVertex(VV);

        // Gyroscope bias vertex (fixed)
        VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
        VG->setId(2 * (maxKFid + 1) + pKFi->mnId);
        VG->setFixed(true);
        optimizer.addVertex(VG);

        // Accelerometer bias vertex (fixed)
        VertexAccBias* VA = new VertexAccBias(vpKFs.front());
        VA->setId(3 * (maxKFid + 1) + pKFi->mnId);
        VA->setFixed(true);
        optimizer.addVertex(VA);
    }

    // 3. Add gravity direction and scale vertices (variables to be optimized)
    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(4 * (maxKFid + 1));
    VGDir->setFixed(false);  // Optimize gravity direction
    optimizer.addVertex(VGDir);

    VertexScale* VS = new VertexScale(scale);
    VS->setId(4 * (maxKFid + 1) + 1);
    VS->setFixed(false);     // Optimize scale
    optimizer.addVertex(VS);

    // 4. Add IMU preintegration edge constraints
    int edge_count = 0;
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid)
        {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;

            // Get all related vertices
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid + 1 + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid + 1 + pKFi->mnId);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(2 * (maxKFid + 1) + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(3 * (maxKFid + 1) + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(4 * (maxKFid + 1));
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(4 * (maxKFid + 1) + 1);

            // Check if vertices exist
            if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cerr << "Error: Missing vertices for edge " << i << endl;
                continue;
            }

            // Create IMU preintegration edge
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            // Add robust kernel function
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            ei->setRobustKernel(rk);
            rk->setDelta(1.0f);
            
            optimizer.addEdge(ei);
            edge_count++;
        }
    }

    // 5. Execute optimization
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    const float error_initial = optimizer.activeRobustChi2();
    
    optimizer.optimize(iterations);
    
    optimizer.computeActiveErrors();
    const float error_final = optimizer.activeRobustChi2();

    // 6. Extract optimization results
    scale = VS->estimate();
    Rwg = VGDir->estimate().Rwg;

    cout << "Inertial optimization: " << edge_count << " edges, "
         << "error: " << error_initial << " -> " << error_final << endl;
}


/**
 * @brief Inertial optimization for complete IMU-visual optimization
 * 
 * Algorithm principle:
 * 1. Jointly optimize gravity direction, scale, IMU biases and keyframe velocities
 * 2. Fix keyframe poses, optimize velocity and IMU bias parameters
 * 3. Use prior information to constrain IMU biases to prevent divergence
 * 4. Employ Levenberg-Marquardt algorithm for nonlinear optimization
 * 5. Used for precise IMU calibration during SLAM system operation
 * 
 * @param pMap          Map pointer
 * @param Rwg           Gravity direction rotation matrix
 * @param scale         Scale factor
 * @param bg            Gyroscope bias output
 * @param ba            Accelerometer bias output
 * @param bFixedVel     Whether to fix velocities
 * @param priorG        Gyroscope bias prior weight
 * @param priorA        Accelerometer bias prior weight
 */
void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, 
                                   Eigen::Vector3d &bg, Eigen::Vector3d &ba, 
                                   bool bFixedVel, float priorG, float priorA)
{
    const int iterations = 200;
    const long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // 1. Configure optimizer - using Levenberg-Marquardt algorithm
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

    // Set initial damping factor
    if (priorG != 0.0f)
        solver->setUserLambdaInit(1e3);

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // 2. Add keyframe vertices (fixed poses, optimizable velocities)
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid)
            continue;

        // Pose vertex (fixed)
        VertexPose *VP = new VertexPose(pKFi, pMap->mpCamera);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        // Velocity vertex (fixed or optimizable based on parameter)
        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid + pKFi->mnId + 1);
        VV->setFixed(bFixedVel);
        optimizer.addVertex(VV);
    }

    // 3. Add IMU bias vertices
    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid * 2 + 2);
    VG->setFixed(bFixedVel);
    optimizer.addVertex(VG);

    VertexAccBias* VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid * 2 + 3);
    VA->setFixed(bFixedVel);
    optimizer.addVertex(VA);

    // 4. Add bias prior constraints
    Eigen::Vector3f bias_prior = Eigen::Vector3f::Zero();

    // Accelerometer bias prior
    EdgePriorAcc* edge_prior_acc = new EdgePriorAcc(bias_prior);
    edge_prior_acc->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    edge_prior_acc->setInformation(priorA * Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge_prior_acc);

    // Gyroscope bias prior
    EdgePriorGyro* edge_prior_gyro = new EdgePriorGyro(bias_prior);
    edge_prior_gyro->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    edge_prior_gyro->setInformation(priorG * Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge_prior_gyro);

    // 5. Add gravity direction and scale vertices
    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(maxKFid * 2 + 4);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);

    VertexScale* VS = new VertexScale(scale);
    VS->setId(maxKFid * 2 + 5);
    VS->setFixed(false);
    optimizer.addVertex(VS);

    // 6. Add IMU preintegration edge constraints
    vector<EdgeInertialGS*> vp_edges;
    vector<pair<KeyFrame*, KeyFrame*>> vp_used_kf;
    vp_edges.reserve(vpKFs.size());
    vp_used_kf.reserve(vpKFs.size());

    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid)
        {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;

            if (!pKFi->mpImuPreintegrated)
            {
                cerr << "Warning: No preintegrated measurement for KF " << pKFi->mnId << endl;
                continue;
            }

            // Update preintegration bias
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());

            // Get all related vertices
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid + pKFi->mPrevKF->mnId + 1);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid + pKFi->mnId + 1);
            g2o::HyperGraph::Vertex* VG_vertex = optimizer.vertex(maxKFid * 2 + 2);
            g2o::HyperGraph::Vertex* VA_vertex = optimizer.vertex(maxKFid * 2 + 3);
            g2o::HyperGraph::Vertex* VGDir_vertex = optimizer.vertex(maxKFid * 2 + 4);
            g2o::HyperGraph::Vertex* VS_vertex = optimizer.vertex(maxKFid * 2 + 5);

            // Check vertex integrity
            if (!VP1 || !VV1 || !VG_vertex || !VA_vertex || !VP2 || !VV2 || !VGDir_vertex || !VS_vertex)
            {
                cerr << "Error: Missing vertices for IMU edge between KF " 
                     << pKFi->mPrevKF->mnId << " and " << pKFi->mnId << endl;
                continue;
            }

            // Create IMU preintegration edge
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG_vertex));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA_vertex));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir_vertex));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS_vertex));

            vp_edges.push_back(ei);
            vp_used_kf.push_back(make_pair(pKFi->mPrevKF, pKFi));
            optimizer.addEdge(ei);
        }
    }

    // 7. Execute optimization
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(iterations);

    // 8. Extract optimization results
    
    // Re-obtain bias vertices (pointers may have changed)
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid * 2 + 2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid * 2 + 3));
    
    // Extract bias estimation values
    Vector6d bias_vector;
    bias_vector << VG->estimate(), VA->estimate();
    bg = VG->estimate();
    ba = VA->estimate();
    scale = VS->estimate();

    // Create IMU bias object
    IMU::Bias optimized_bias(bias_vector[3], bias_vector[4], bias_vector[5],  // acc bias
                           bias_vector[0], bias_vector[1], bias_vector[2]);   // gyro bias
    
    // Update gravity direction
    Rwg = VGDir->estimate().Rwg;

    // 9. Update keyframe velocities and biases
    const size_t num_kf = vpKFs.size();
    for (size_t i = 0; i < num_kf; i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid)
            continue;

        // Update keyframe velocity
        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid + pKFi->mnId + 1));
        Eigen::Vector3d velocity_world = VV->estimate();
        pKFi->SetVelocity(velocity_world.cast<float>());

        // Update bias (reintegrate if change is significant)
        if ((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
        {
            pKFi->SetNewBias(optimized_bias);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
        {
            pKFi->SetNewBias(optimized_bias);
        }
    }

    cout << "Full inertial optimization: " << vp_edges.size() << " IMU edges optimized" << endl;
}


/**
 * @brief Marginalization function - implements Schur complement elimination
 * 
 * Algorithm principle:
 * 1. Rearrange Hessian matrix, move variables to be marginalized to bottom-right corner
 * 2. Apply Schur complement formula: H_new = H_aa - H_ab * H_bb^(-1) * H_ba
 * 3. Use SVD decomposition for inversion to improve numerical stability
 * 4. Eliminate specified variable blocks while preserving constraint information for remaining variables
 * 5. Used for old keyframe marginalization in sliding window optimization
 * 
 * @param H      Original Hessian matrix
 * @param start  Starting index of block to be marginalized
 * @param end    Ending index of block to be marginalized
 * @return       Marginalized Hessian matrix
 */
Eigen::MatrixXd Optimizer::Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
{
    // Calculate matrix block sizes
    const int a = start;                    // Size of preceding block
    const int b = end - start + 1;          // Size of block to marginalize  
    const int c = H.cols() - (end + 1);     // Size of following block

    // 1. Rearrange matrix, move block to be marginalized to bottom-right corner
    // Original layout: [a | ab | ac]    Target layout: [a | ac | ab]
    //                 [ba| b  | bc] -->                 [ca| c  | cb]
    //                 [ca| cb | c ]                     [ba| bc | b ]
    
    Eigen::MatrixXd H_reordered = Eigen::MatrixXd::Zero(H.rows(), H.cols());
    
    // Fill top-left block (a x a)
    if (a > 0)
    {
        H_reordered.block(0, 0, a, a) = H.block(0, 0, a, a);
        H_reordered.block(0, a + c, a, b) = H.block(0, a, a, b);
        H_reordered.block(a + c, 0, b, a) = H.block(a, 0, b, a);
    }
    
    // Fill cross-term blocks
    if (a > 0 && c > 0)
    {
        H_reordered.block(0, a, a, c) = H.block(0, a + b, a, c);
        H_reordered.block(a, 0, c, a) = H.block(a + b, 0, c, a);
    }
    
    // Fill top-right and related blocks
    if (c > 0)
    {
        H_reordered.block(a, a, c, c) = H.block(a + b, a + b, c, c);
        H_reordered.block(a, a + c, c, b) = H.block(a + b, a, c, b);
        H_reordered.block(a + c, a, b, c) = H.block(a, a + b, b, c);
    }
    
    // Fill block to be marginalized (b x b)
    H_reordered.block(a + c, a + c, b, b) = H.block(a, a, b, b);

    // 2. Perform Schur complement marginalization
    // Compute H_bb^(-1) using SVD decomposition for improved numerical stability
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_reordered.block(a + c, a + c, b, b),
                                         Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    Eigen::VectorXd singular_values_inv = svd.singularValues();
    const double threshold = 1e-6;
    
    // Compute pseudo-inverse singular values
    for (int i = 0; i < b; ++i)
    {
        if (singular_values_inv(i) > threshold)
            singular_values_inv(i) = 1.0 / singular_values_inv(i);
        else
            singular_values_inv(i) = 0.0;  // Handle singular cases
    }
    
    // Reconstruct pseudo-inverse matrix
    Eigen::MatrixXd H_bb_inv = svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().transpose();
    
    // Apply Schur complement formula: H_new = H_aa - H_ab * H_bb^(-1) * H_ba
    H_reordered.block(0, 0, a + c, a + c) = H_reordered.block(0, 0, a + c, a + c) - 
        H_reordered.block(0, a + c, a + c, b) * H_bb_inv * H_reordered.block(a + c, 0, b, a + c);
    
    // Zero out blocks related to marginalized variables
    H_reordered.block(a + c, a + c, b, b) = Eigen::MatrixXd::Zero(b, b);
    H_reordered.block(0, a + c, a + c, b) = Eigen::MatrixXd::Zero(a + c, b);
    H_reordered.block(a + c, 0, b, a + c) = Eigen::MatrixXd::Zero(b, a + c);

    // 3. Restore original matrix arrangement
    // Reverse rearrangement: [a | ac | 0]    Target: [a | 0 | ac]
    //                       [ca| c  | 0] -->        [0 | 0 | 0 ]
    //                       [0 | 0  | 0]            [ca| 0 | c ]
    
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(H.rows(), H.cols());
    
    if (a > 0)
    {
        result.block(0, 0, a, a) = H_reordered.block(0, 0, a, a);
        result.block(0, a, a, b) = H_reordered.block(0, a + c, a, b);
        result.block(a, 0, b, a) = H_reordered.block(a + c, 0, b, a);
    }
    
    if (a > 0 && c > 0)
    {
        result.block(0, a + b, a, c) = H_reordered.block(0, a, a, c);
        result.block(a + b, 0, c, a) = H_reordered.block(a, 0, c, a);
    }
    
    if (c > 0)
    {
        result.block(a + b, a + b, c, c) = H_reordered.block(a, a, c, c);
        result.block(a + b, a, c, b) = H_reordered.block(a, a + c, c, b);
        result.block(a, a + b, b, c) = H_reordered.block(a + c, a, b, c);
    }
    
    result.block(a, a, b, b) = H_reordered.block(a + c, a + c, b, b);

    return result;
}