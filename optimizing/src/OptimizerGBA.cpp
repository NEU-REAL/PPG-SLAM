#include "Optimizer.h"
#include <complex>
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


void Optimizer::GlobalBundleAdjustment(Map* pMap, int nIterations, const unsigned long nLoopKF, bool* pbStopFlag)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
    vector<MapColine*> vpCLs = pMap->GetAllMapColines();

    // Setup optimizer
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
                                        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    const int nExpectedSize = (vpKFs.size())*vpMPs.size();

    vector<EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Set KeyFrame vertices
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        SE3<float> Tcw = pKF->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==pMap->GetOriginKF()->mnId);
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    // Set MapPoint vertices
    const float thHuberMono = sqrt(5.991);
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];
        if(pMP->isBad())
            continue;
        g2o::VertexPointXYZ* vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(false);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*, int> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*, int>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            const int index = mit->second;

            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;
            if(index < 0)
                continue;
            if(optimizer.vertex(id) == NULL || optimizer.vertex(pKF->mnId) == NULL)
                continue;

            nEdges++;
            const KeyPointEx &kpUn = pKF->mvKeysUn[index];

            Eigen::Matrix<double,2,1> obs;
            obs << kpUn.mPos[0], kpUn.mPos[1];

            EdgeSE3ProjectXYZ* e = new EdgeSE3ProjectXYZ();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
            e->setMeasurement(obs);
            e->setInformation(Eigen::Matrix2d::Identity());
            // robust kernel
            g2o::RobustKernelCauchy* rk = new g2o::RobustKernelCauchy;
            rk->setDelta(thHuberMono);
            e->setRobustKernel(rk);

            e->pCamera = pMap->mpCamera;
            optimizer.addEdge(e);
            vpEdgesMono.push_back(e);
            vpEdgeKFMono.push_back(pKF);
            vpMapPointEdgeMono.push_back(pMP);
        }

        if(nEdges < 2)
            optimizer.removeVertex(vPoint);
    }

    // coline residual
    for(MapPoint* pMP : vpMPs)
    {
        if(optimizer.vertex(pMP->mnId+maxKFid+1) == NULL)
            continue;
        std::vector<MapColine*> vMCs = pMP->getColinearity();
        for(auto pMC : vMCs)
        {
            if(pMC->isBad() || !pMC->mbValid)
                continue;
            MapPoint* pMPs = pMC->mpMPs;
            MapPoint* pMPe = pMC->mpMPe;
            if( optimizer.vertex(pMPs->mnId+maxKFid+1) == NULL ||
                optimizer.vertex(pMPe->mnId+maxKFid+1) == NULL)
                continue;
            EdgeColine* e = new EdgeColine();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPs->mnId+maxKFid+1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMP->mnId+maxKFid+1)));
            e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPe->mnId+maxKFid+1)));
            e->setInformation(Eigen::Matrix3d::Identity() * pMC->aveWeight());
            optimizer.addEdge(e);
        }
    }

    // Optimize!
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));

        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==pMap->GetOriginKF()->mnId) // for initial optimize
        {
            pKF->SetPose(SE3f(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>()));
        }
        else  // for loop correction
        {
            pKF->mTcwGBA = SE3d(SE3quat.rotation(),SE3quat.translation()).cast<float>();
            pKF->mnBAGlobalForKF = nLoopKF;

            SE3f mTwc = pKF->GetPoseInverse();
            SE3f mTcGBA_c = pKF->mTcwGBA * mTwc;
            Eigen::Vector3f vector_dist =  mTcGBA_c.translation();
            double dist = vector_dist.norm();
            if(dist > 1)
            {
                int numMonoBadPoints = 0, numMonoOptPoints = 0;
                vector<MapPoint*> vpMonoMPsOpt;

                for(size_t i2=0, iend=vpEdgesMono.size(); i2<iend;i2++)
                {
                    EdgeSE3ProjectXYZ* e = vpEdgesMono[i2];
                    MapPoint* pMP = vpMapPointEdgeMono[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if(pKF != pKFedge)
                        continue;

                    if(pMP->isBad())
                        continue;

                    if(e->chi2()>5.991 || !e->isDepthPositive())
                    {
                        numMonoBadPoints++;
                    }
                    else
                    {
                        numMonoOptPoints++;
                        vpMonoMPsOpt.push_back(pMP);
                    }

                }
            }
        }
    }

    //Points
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];
        g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(pMP == nullptr || pMP->isBad() || vPoint == nullptr)
            continue;

        if(nLoopKF==pMap->GetOriginKF()->mnId)  // for initial optimize
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else  // for loop correction
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
    // remove bad colinearity edges
    vector<MapEdge*> vpMEs = pMap->GetAllMapEdges();
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
        std::vector<MapColine*> vCLs = pMP->removeColineOutliers();
        for(MapColine* pMC : vCLs)
            pMap->EraseMapColine(pMC);
    }
}

void Optimizer::FullInertialBA(Map *pMap, int its, const unsigned long nLoopKF, bool *pbStopFlag, bool bInit, float priorG, float priorA)
{
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
    const vector<MapColine*> vpCLs = pMap->GetAllMapColines();

    // Setup optimizer
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
                                        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

    solver->setUserLambdaInit(1e-5);

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    // Set KeyFrame vertices
    KeyFrame* pIncKF(nullptr);
    for(KeyFrame* pKFi : vpKFs)
    {
        if(pKFi == nullptr || pKFi->isBad() || pKFi->mnId>maxKFid)
            continue;

        VertexPose * VP = new VertexPose(pKFi, pMap->mpCamera);
        VP->setId(pKFi->mnId);
        pIncKF = pKFi;

        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            if (!bInit)
            {
                VertexGyroBias* VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid+3*(pKFi->mnId)+2);
                VG->setFixed(false);
                optimizer.addVertex(VG);
                VertexAccBias* VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid+3*(pKFi->mnId)+3);
                VA->setFixed(false);
                optimizer.addVertex(VA);
            }
        }
    }

    if (bInit && pIncKF!=nullptr)
    {
        VertexGyroBias* VG = new VertexGyroBias(pIncKF);
        VG->setId(4*maxKFid+2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pIncKF);
        VA->setId(4*maxKFid+3);
        VA->setFixed(false);
        optimizer.addVertex(VA);
    }

    // IMU links
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi == nullptr || pKFi->isBad() || pKFi->mnId > maxKFid)
            continue;

        if(pKFi->mPrevKF == nullptr || pKFi->mPrevKF->isBad() || pKFi->mPrevKF->mnId > maxKFid)
            continue;

        if(!pKFi->bImu || !pKFi->mPrevKF->bImu)
        {
            cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
            continue;
        }

        pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
        g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
        g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);

        g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
        g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);

        g2o::HyperGraph::Vertex* VG1(nullptr);
        g2o::HyperGraph::Vertex* VA1(nullptr);
        g2o::HyperGraph::Vertex* VG2(nullptr);
        g2o::HyperGraph::Vertex* VA2(nullptr);
        if (!bInit)
        {
            VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
            VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
            VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
            VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);
            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }
            EdgeGyroRW* egr= new EdgeGyroRW();
            egr->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            egr->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG2));
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
            egr->setInformation(InfoG);
            egr->computeError();
            optimizer.addEdge(egr);

            EdgeAccRW* ear = new EdgeAccRW();
            ear->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            ear->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA2));
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
            ear->setInformation(InfoA);
            ear->computeError();
            optimizer.addEdge(ear);
        }
        else
        {
            VG1 = optimizer.vertex(4*maxKFid+2);
            VA1 = optimizer.vertex(4*maxKFid+3);
            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<endl;
                continue;
            }
        }

        EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
        ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
        ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
        ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
        ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
        ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
        ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

        g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
        ei->setRobustKernel(rki);
        rki->setDelta(sqrt(16.92));
        optimizer.addEdge(ei);
    }

    if (bInit)
    {
        // Add prior to comon biases
        Eigen::Vector3f bprior;
        bprior.setZero();

        EdgePriorGyro* epg = new EdgePriorGyro(bprior);
        epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(4*maxKFid+2)));
        epg->setInformation(priorG*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);

        EdgePriorAcc* epa = new EdgePriorAcc(bprior);
        epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(4*maxKFid+3)));
        epa->setInformation(priorA*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);

    }

    const float thHuberMono = sqrt(5.991);
    const unsigned long iniMPid = maxKFid*5;
    for(MapPoint* pMP : vpMPs)
    {
        if(pMP == nullptr || pMP->isBad())
            continue;
        g2o::VertexPointXYZ* vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(false);
        optimizer.addVertex(vPoint);

        //Set edges
        const map<KeyFrame*, int> observations = pMP->GetObservations();
        bool bAllFixed = true;
        for(auto mit : observations)
        {
            KeyFrame* pKFi = mit.first;
            const int index = mit.second;
            if(pKFi == nullptr || pKFi->isBad() || pKFi->mnId>maxKFid || index < 0)
                continue;

            Eigen::Matrix<double,2,1> obs;
            obs << pKFi->mvKeysUn[index].mPos[0], pKFi->mvKeysUn[index].mPos[1];
            EdgeMono* e = new EdgeMono();
            g2o::OptimizableGraph::Vertex* VP = optimizer.vertex(pKFi->mnId);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
            e->setMeasurement(obs);
            e->setInformation(Eigen::Matrix2d::Identity());
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(thHuberMono);
            optimizer.addEdge(e);
            if(bAllFixed)
            if(!VP->fixed())
                bAllFixed=false;
        }
        if(bAllFixed)
            optimizer.removeVertex(vPoint);
    }

    // coline residual
    for(MapPoint* pMP : vpMPs)
    {
        if(optimizer.vertex(pMP->mnId+iniMPid+1) == NULL)
            continue;
        std::vector<MapColine*> vMCs = pMP->getColinearity();
        for(auto pMC : vMCs)
        {
            if(pMC->isBad() || !pMC->mbValid)
                continue;
            MapPoint* pMPs = pMC->mpMPs;
            MapPoint* pMPe = pMC->mpMPe;
            if( optimizer.vertex(pMPs->mnId+iniMPid+1) == NULL ||
                optimizer.vertex(pMPe->mnId+iniMPid+1) == NULL)
                continue;
            EdgeColine* e = new EdgeColine();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPs->mnId+iniMPid+1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMP->mnId+iniMPid+1)));
            e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pMPe->mnId+iniMPid+1)));
            e->setInformation(Eigen::Matrix3d::Identity() * pMC->aveWeight());
            optimizer.addEdge(e); 
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;
    optimizer.initializeOptimization();
    optimizer.optimize(its);
    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        if(nLoopKF==0)
        {
            SE3f Tcw(VP->estimate().Rcw.cast<float>(), VP->estimate().tcw.cast<float>());
            pKFi->SetPose(Tcw);
        }
        else
        {
            pKFi->mTcwGBA = SE3f(VP->estimate().Rcw.cast<float>(),VP->estimate().tcw.cast<float>());
            pKFi->mnBAGlobalForKF = nLoopKF;

        }
        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            if(nLoopKF==0)
                pKFi->SetVelocity(VV->estimate().cast<float>());
            else
                pKFi->mVwbGBA = VV->estimate().cast<float>();

            VertexGyroBias* VG;
            VertexAccBias* VA;
            if (!bInit)
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            }
            else
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(4*maxKFid+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(4*maxKFid+3));
            }

            Vector6d vb;
            vb << VG->estimate(), VA->estimate();
            IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
            if(nLoopKF==0)
                pKFi->SetNewBias(b);
            else
                pKFi->mBiasGBA = b;
        }
    }

    //Points
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];
        g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
        if(vPoint == nullptr)
            continue;
        if(nLoopKF==0)
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
    // remove bad colinearity edges
    vector<MapEdge*> vpMEs = pMap->GetAllMapEdges();
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
        std::vector<MapColine*> vCLs = pMP->removeColineOutliers();
        for(MapColine* pMC : vCLs)
            pMap->EraseMapColine(pMC);
    }
    pMap->InfoMapChange();
}