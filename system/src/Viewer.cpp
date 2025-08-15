#include <pangolin/pangolin.h>
#include <mutex>

#include "Viewer.h"


void MSViewing::Launch(Map *pMap)
{
    mbStep = false;
    mbStepByStep = false;
    mbFinishRequested = false;
    mpMap = pMap;
    mState=NO_IMAGES_YET;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    mnCurFrameID = 0;
    mptViewer = new thread(&MSViewing::Run, this);
}

void MSViewing::Run()
{
    pangolin::CreateWindowAndBind("Map Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", false, true);
    pangolin::Var<bool> menuCamView("menu.Camera View", false, false);
    pangolin::Var<bool> menuTopView("menu.Top View", false, false);
    pangolin::Var<bool> menuUnFaded("menu.unfaded", false, true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
    pangolin::Var<bool> menuShowColine("menu.Show colines", true, true);
    pangolin::Var<bool> menuShowEdges("menu.Show Edges", true, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph", false, true);
    pangolin::Var<bool> menuShowInertialGraph("menu.Show Inertial", true, true);
    pangolin::Var<bool> menuStepByStep("menu.Step By Step", false, true); // false, true
    pangolin::Var<bool> menuStep("menu.Step", false, false);
    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -1, -3, 0, 0, 0, 0.0, -1.0, 0.0));
    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));
    pangolin::OpenGlMatrix Twc, Twr;
    Twc.SetIdentity();
    pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
    Ow.SetIdentity();
    cv::namedWindow("Current Frame");
    bool bFollow = true;
    bool bCameraView = true;
    cout << "Starting the Viewer" << endl;
    while (1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        GetCurrentOpenGLCameraMatrix(Twc, Ow);
        if (menuFollowCamera && bFollow)
        {
            if (bCameraView)
                s_cam.Follow(Twc);
            else
                s_cam.Follow(Ow);
        }
        else if (menuFollowCamera && !bFollow)
        {
            if (bCameraView)
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, -1, -3, 0, 0, 0, 0.0, -1.0, 0.0));
                s_cam.Follow(Twc);
            }
            else
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024, 768, 3000, 3000, 512, 389, 0.1, 1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, 0.01, 10, 0, 0, 0, 0.0, 0.0, 1.0));
                s_cam.Follow(Ow);
            }
            bFollow = true;
        }
        else if (!menuFollowCamera && bFollow)
            bFollow = false;

        if (menuCamView)
        {
            menuCamView = false;
            bCameraView = true;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, -1, -3, 0, 0, 0, 0.0, -1.0, 0.0));
            s_cam.Follow(Twc);
        }

        if (menuTopView && mpMap->isImuInitialized())
        {
            menuTopView = false;
            bCameraView = false;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024, 768, 3000, 3000, 512, 389, 0.1, 10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, 0.01, 50, 0, 0, 0, 0.0, 0.0, 1.0));
            s_cam.Follow(Ow);
        }

        mbStepByStep = menuStepByStep;
        if(menuStep)
        {
            mbStep = true;
            menuStep = false;
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        DrawCurrentCamera(Twc);
        if (menuShowKeyFrames || menuShowGraph || menuShowInertialGraph)
            DrawKeyFrames(menuShowKeyFrames, menuShowGraph, menuShowInertialGraph);

        mbShowPoint = menuShowPoints;
        mbShowColine = menuShowColine;
        mbShowEdge = menuShowEdges;
        mbunFaded = menuUnFaded;
        if (mbShowPoint)
            DrawMapPoints();
        if (mbShowColine)
            DrawMapColines();
        if (mbShowEdge)
            DrawMapEdges();

        pangolin::FinishFrame();

        cv::Mat toShow = DrawFrame();
        cv::imshow("Current Frame", toShow);
        cv::waitKey(30);

        if (mbFinishRequested)
        {
            mbFinishRequested = false;
            break;
        }
    }
}

void MSViewing::RequestFinish()
{
    mbFinishRequested = true;
    while(mbFinishRequested == true)
        usleep(3000);
    std::cout << "Viewing finished."<<std::endl;
}

cv::Mat MSViewing::DrawFrame()
{
    cv::Mat im;
    vector<KeyPointEx> vIniKeys;
    vector<int> vIniMatches;

    vector<KeyPointEx> vCurrentKeys;
    vector<KeyEdge> vCurrentEdges;

    vector<bool> vbOutliers;
    vector<MapPoint*> vpMapPoints;
    vector<MapEdge*> vpMapEdges;

    SE3<float> Tcw;

    int state; // Tracking state

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state = mState;
        mIm.copyTo(im);

        if(mState==NOT_INITIALIZED)
        {
            vIniKeys = mvIniKeys;
            vIniMatches = mvIniMatches;
        }

        vCurrentKeys = mvCurrentKeys;
        vCurrentEdges = mvCurrentEdges;

        vbOutliers = mvbOutliers;
        vpMapPoints = mvpMapPoints;
        vpMapEdges = mvpMapEdges;

        Tcw = mTcw;
    }

    if(im.channels()<3)
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    cv::resize(im, im, cv::Size(im.cols, im.rows), cv::INTER_LINEAR);

    mnTracked=0;
    // draw colines
    if(mbShowColine)
    {
        for(KeyPointEx kp : vCurrentKeys)
        {
            for(std::pair<unsigned int, unsigned int> cpt : kp.mvColine)
            {
                cv::Point2f pt1,pt2;
                pt1 = cv::Point2f(vCurrentKeys[cpt.first].mPos[0],vCurrentKeys[cpt.first].mPos[1]);
                pt2 = cv::Point2f(vCurrentKeys[cpt.second].mPos[0],vCurrentKeys[cpt.second].mPos[1]);
                cv::line(im,pt1, pt2, cv::Scalar(20,20,255),2);
            }
        } 
    }
    // draw edges
    if(mbShowEdge)
    {
        for(KeyEdge ke : vCurrentEdges)
        {
            const KeyPointEx &kp1 = vCurrentKeys[ke.startIdx];
            const KeyPointEx &kp2 = vCurrentKeys[ke.endIdx];
            cv::line(im, cv::Point2f(kp1.mPos[0],kp1.mPos[1]), cv::Point2f(kp2.mPos[0],kp2.mPos[1]), cv::Scalar(0,255,0),1);
            cv::circle(im, cv::Point2f(kp1.mPos[0],kp1.mPos[1]), 3, cv::Scalar(0,255,0), -1);
            cv::circle(im, cv::Point2f(kp2.mPos[0],kp2.mPos[1]), 3, cv::Scalar(0,255,0), -1);
        }
    }
    // draw keys

    for(unsigned int i=0; i<vCurrentKeys.size(); i++)
    {
        MapPoint * pMP = vpMapPoints[i];
        if(pMP == nullptr || pMP->isBad())
            continue;
        if(!vbOutliers[i])
        {        
            if(mbShowPoint)
                cv::circle(im, cv::Point2f(vCurrentKeys[i].mPos[0],vCurrentKeys[i].mPos[1]), 3, cv::Scalar(0,255,255),1);
            mnTracked++;
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);
    return imWithInfo;
}


void MSViewing::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==OK)
    {
        int nKFs = mpMap->KeyFramesInMap();
        int nMPs = mpMap->MapPointsInMap();
        s << " KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
    }
    else if(nState==LOST)
        s << " TRACK LOST. TRYING TO RELOCALIZE ";

        
    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);
    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

void MSViewing::UpdateFrame(Frame &F)
{
    unique_lock<mutex> lock(mMutex);
    F.srcMat.copyTo(mIm);
    mvCurrentKeys=F.mvKeys;
    mvCurrentEdges = F.mvKeyEdges;

    if(MSTracking::get().mLastProcessedState==NOT_INITIALIZED)
    {
        mvIniKeys= F.mvKeys;
        mvIniMatches=MSTracking::get().mvIniMatches;
    }

    mvbOutliers = F.mvbOutlier;
    mvpMapPoints = F.mvpMapPoints;
    mvpMapEdges = F.mvpMapEdges;

    mTcw = F.GetPose();

    mState = static_cast<int>(MSTracking::get().mLastProcessedState);
    mnCurFrameID = F.mnId;
}

void MSViewing::DrawMapPoints()
{
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    if(vpMPs.empty())
        return;
    glPointSize(2);
    glBegin(GL_POINTS);
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad())
            continue;
        Eigen::Matrix<float,3,1> pos = vpMPs[i]->GetWorldPos();

        std::chrono::steady_clock::time_point tnow = chrono::steady_clock::now();
        double ts = std::chrono::duration_cast<std::chrono::duration<double>>(tnow - vpMPs[i]->startTime).count();
        if(mbunFaded)
            ts = 0;
        if(ts >= 18)
            glColor4f(0.0,0.0,0.0,0.1);
        else
            glColor4f(0.0,0.0,0.0, (20 - ts) / 20.0);

        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();

}

void MSViewing::DrawMapColines()
{
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    if(vpMPs.empty())
        return;
    // draw colinear points
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad())
            continue;
        std::vector<MapColine*> vMPCLs = vpMPs[i]->getColinearity();

        for(auto pMC : vMPCLs)
        {
            if(pMC->isBad() || !pMC->mbValid)
                continue;
            Eigen::Vector3f pos = pMC->mpMPm->GetWorldPos();
            Eigen::Vector3f pos1 = pMC->mpMPs->GetWorldPos();
            Eigen::Vector3f pos2 = pMC->mpMPe->GetWorldPos();
            glLineWidth(2);
            glBegin(GL_LINES);
            glColor3f(1,0,0);
            glVertex3f(pos1(0),pos1(1),pos1(2));
            glVertex3f(pos2(0),pos2(1),pos2(2));
            glEnd();

            glPointSize(3);
            glBegin(GL_POINTS);
            glColor3f(0.0,0.0,0.0);
            glVertex3f(pos(0),pos(1),pos(2));
            glVertex3f(pos1(0),pos1(1),pos1(2));
            glVertex3f(pos2(0),pos2(1),pos2(2));
            glEnd();

        }
    }
}

void MSViewing::DrawMapEdges()
{
    Map* pActiveMap = mpMap;
    if(!pActiveMap)
        return;

    const vector<MapEdge*> &vpMEs = pActiveMap->GetAllMapEdges();

    if(vpMEs.empty())
        return;

    for(size_t i=0, iend=vpMEs.size(); i<iend;i++)
    {
        if(vpMEs[i]->isBad() || !vpMEs[i]->mbValid)
            continue;
        MapPoint* pMPs = vpMEs[i]->mpMPs;
        MapPoint* pMPe = vpMEs[i]->mpMPe;

        Eigen::Matrix<float,3,1> pos1 = pMPs->GetWorldPos();
        Eigen::Matrix<float,3,1> pos2 = pMPe->GetWorldPos();

        // visualization
        if(vpMEs[i]->trackedFrameId == mnCurFrameID)
        {
            glLineWidth(2);
            glBegin(GL_LINES);
            glColor3f(0,1,0);
            glVertex3f(pos1(0),pos1(1),pos1(2));
            glVertex3f(pos2(0),pos2(1),pos2(2));
            glEnd();
        }
        else
        {
            glLineWidth(1);
            glBegin(GL_LINES);
            std::chrono::steady_clock::time_point tnow = chrono::steady_clock::now();
            double ts = std::chrono::duration_cast<std::chrono::duration<double>>(tnow - vpMEs[i]->startTime).count();
            if(mbunFaded)
                ts = 0;
            if(ts >= 18)
                glColor4f(0.2,0.2,0.6, 0.1);
            else
                glColor4f(0.2,0.2,0.6, (20 - ts) / 20.0);
            glVertex3f(pos1(0),pos1(1),pos1(2));
            glVertex3f(pos2(0),pos2(1),pos2(2));
            glEnd();
    
            glPointSize(3);
            glBegin(GL_POINTS);
            if(ts >= 18)
                glColor4f(0.0,0.0,0.0,0.1);
            else
                glColor4f(0.0,0.0,0.0,(20 - ts) / 20.0);
            glVertex3f(pos1(0),pos1(1),pos1(2));
            glVertex3f(pos2(0),pos2(1),pos2(2));
            glEnd();
        }
    }
}

void MSViewing::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph)
{
    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();

            glPushMatrix();
            glMultMatrixf((GLfloat*)Twc.data());
        
            glLineWidth(1);
            glColor3f(0.0f,0.0f,1.0f); // Basic color
            glBegin(GL_LINES);

            glColor3f(1.0f,0.0f,0.0f);
            glVertex3f(0,   0,   0);
            glVertex3f(0.1, 0,   0);
            glColor3f(0.0f,1.0f,0.0f);
            glVertex3f(0,   0,   0);
            glVertex3f(0,   0.1, 0);
            glColor3f(0.0f,0.0f,1.0f);
            glVertex3f(0,   0,   0);
            glVertex3f(0,   0,   0.1);
            glEnd();
            glPopMatrix();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(0.5);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        // int covisibilityCount = 0;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(10);
            Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow(0),Ow(1),Ow(2));
                    glVertex3f(Ow2(0),Ow2(1),Ow2(2));
                    // covisibilityCount++;
                }
            }
            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owl(0),Owl(1),Owl(2));
            }
        }

        glEnd();
        // Debug output - print occasionally
        // static int frameCounter = 0;
        // frameCounter++;
        // if(frameCounter % 100 == 0) {  // Print every 100 frames
        //     cout << "Covisibility connections drawn: " << covisibilityCount << ", KeyFrames: " << vpKFs.size() << endl;
        // }
    }

    if(bDrawInertialGraph && mpMap->isImuInitialized())
    {
        glLineWidth(0.6);
        glColor4f(1.0f, 0.0f,1.0f,0.6f);
        glBegin(GL_LINES);

        //Draw inertial links
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            Eigen::Vector3f Ow = pKFi->GetCameraCenter();
            KeyFrame* pNext = pKFi->mNextKF;
            if(pNext)
            {
                Eigen::Vector3f Owp = pNext->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }
        }
        glEnd();
    }
}

void MSViewing::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    glPushMatrix();
    glMultMatrixd(Twc.m);
    glLineWidth(2);
    glColor3f(0.0f,0.0f,1.0f);
    glBegin(GL_LINES);

    float w(0.2), h(0.15), z(0.2);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);

    glEnd();
    glPopMatrix();
}


void MSViewing::SetCurrentCameraPose(const SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutex);

    mCameraPoses.push_back(Tcw.inverse());
    if(mCameraPoses.size() > 1)
        mCameraPoses.pop_front();

    Eigen::Vector3f aveP(0,0,0);
    Eigen::Vector3f aveR(0,0,0);
    for(auto pos : mCameraPoses)
    {
        aveP += pos.translation();
        aveR += pos.so3().log();
    }
    aveP /= mCameraPoses.size();
    aveR /= mCameraPoses.size();
    
    if(mCameraPoses.front().so3().log().dot(mCameraPoses.back().so3().log()) < 0)
    aveR = mCameraPoses.back().so3().log();

    mCameraPose.translation() = aveP;
    
    // Safety check for aveR before calling SO3::exp
    if (!aveR.allFinite()) {
        std::cout << "Warning: Invalid rotation vector in Viewer" << std::endl;
        mCameraPose.so3() = SO3f();  // Identity rotation
    } else {
        try {
            mCameraPose.so3() = SO3f::exp(aveR);
        } catch (const std::exception& e) {
            std::cout << "Warning: SO3::exp failed in Viewer: " << e.what() << std::endl;
            mCameraPose.so3() = SO3f();  // Identity rotation
        }
    }
    // mCameraPose = Tcw.inverse();
}

void MSViewing::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
{
    Eigen::Matrix4f Twc;
    {
        unique_lock<mutex> lock(mMutex);
        Twc = mCameraPose.matrix();
    }
    for (int i = 0; i<4; i++) {
        M.m[4*i] = Twc(0,i);
        M.m[4*i+1] = Twc(1,i);
        M.m[4*i+2] = Twc(2,i);
        M.m[4*i+3] = Twc(3,i);
    }
    MOw.SetIdentity();
    MOw.m[12] = Twc(0,3);
    MOw.m[13] = Twc(1,3);
    MOw.m[14] = Twc(2,3);
}

void MSViewing::SaveTrajectory(const string &filename)
{
    cout << endl << "Saving trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),[](KeyFrame* pKF1, KeyFrame* pKF2){ return pKF1->mnId < pKF2->mnId; });

    SE3f Twb;
    Twb = vpKFs[0]->GetImuPose();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    list<KeyFrame*>::iterator lRit = MSTracking::get().mlpReferences.begin();
    list<double>::iterator lT = MSTracking::get().mlFrameTimes.begin();

    for(auto lit=MSTracking::get().mlRelativeFramePoses.begin(), lend=MSTracking::get().mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        KeyFrame* pKF = *lRit;
        SE3f Trw;
        if (!pKF)
            continue;
        while(pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->mPrevKF;
        }
        Trw = Trw * pKF->GetPose()*Twb;

        SE3f Twb = (pKF->mpImuCalib->mTbc * (*lit) * Trw).inverse();
        Eigen::Quaternionf q = Twb.unit_quaternion();
        Eigen::Vector3f twb = Twb.translation();
        f << setprecision(6) << (*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << endl;
    }
    f.close();
    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}

void MSViewing::SaveKeyFrameTrajectory(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),[](KeyFrame* pKF1, KeyFrame* pKF2){ return pKF1->mnId < pKF2->mnId; });

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

        if(!pKF || pKF->isBad())
            continue;
        SE3f Twb = pKF->GetImuPose();
        Eigen::Quaternionf q = Twb.unit_quaternion();
        Eigen::Vector3f twb = Twb.translation();
        f << setprecision(6) << pKF->mTimeStamp  << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << endl;
    }
    f.close();
}