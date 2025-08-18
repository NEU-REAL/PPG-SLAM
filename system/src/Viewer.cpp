#include "Viewer.h"

// Standard libraries
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <unistd.h>

// OpenCV
#include <opencv2/opencv.hpp>

// Pangolin
#include <pangolin/pangolin.h>

// PPG-SLAM headers
#include "Tracking.h"
#include "Map.h"
#include "KeyFrame.h"

using namespace std;


/**
 * @brief Initialize and launch the viewer system
 * @param pMap Pointer to the map to be visualized
 * 
 * Algorithm:
 * - Initializes viewer state and control flags
 * - Sets up default frame buffer and tracking state
 * - Creates separate thread for rendering loop to avoid blocking main SLAM pipeline
 */
void MSViewing::Launch(Map *pMap)
{
    // Initialize control flags
    mbStep = false;
    mbStepByStep = false;
    mbFinishRequested = false;
    
    // Set map reference
    mpMap = pMap;
    
    // Initialize tracking state
    mState = NO_IMAGES_YET;
    
    // Create default frame buffer (VGA resolution)
    mIm = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    mnCurFrameID = 0;
    
    // Launch viewer thread to run independently
    mptViewer = new thread(&MSViewing::Run, this);
}

/**
 * @brief Main viewer rendering loop
 * 
 * Algorithm:
 * - Creates Pangolin window with OpenGL context
 * - Sets up interactive GUI controls for visualization options
 * - Implements main rendering loop with camera following and view switching
 * - Handles different view modes: camera view, top view with IMU orientation
 * - Provides real-time control over visualization elements
 */
void MSViewing::Run()
{
    // Create OpenGL window and context
    pangolin::CreateWindowAndBind("PPG-SLAM Map Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);          // Enable depth testing for 3D rendering
    glEnable(GL_BLEND);               // Enable alpha blending for transparency
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Create GUI panel with interactive controls
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    
    // View control variables
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", false, true);
    pangolin::Var<bool> menuCamView("menu.Camera View", false, false);
    pangolin::Var<bool> menuTopView("menu.Top View", false, false);
    
    // Visualization control variables
    pangolin::Var<bool> menuUnFaded("menu.Unfaded", false, true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
    pangolin::Var<bool> menuShowColine("menu.Show Colines", true, true);
    pangolin::Var<bool> menuShowEdges("menu.Show Edges", true, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph", false, true);
    pangolin::Var<bool> menuShowInertialGraph("menu.Show Inertial", true, true);
    
    // Debug control variables
    pangolin::Var<bool> menuStepByStep("menu.Step By Step", false, true);
    pangolin::Var<bool> menuStep("menu.Step", false, false);

    // Define camera render object for 3D scene navigation
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -1, -3, 0, 0, 0, 0.0, -1.0, 0.0));

    // Create 3D viewport with mouse handler
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    // Initialize transformation matrices
    pangolin::OpenGlMatrix Twc, Twr;
    Twc.SetIdentity();
    pangolin::OpenGlMatrix Ow; // Oriented with gravity in the z axis
    Ow.SetIdentity();

    // Create OpenCV window for 2D frame display
    cv::namedWindow("Current Frame");

    // View state tracking
    bool bFollow = true;
    bool bCameraView = true;

    cout << "PPG-SLAM Viewer: Starting visualization..." << endl;

    // Main rendering loop
    while (true)
    {
        // Clear frame buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Update camera matrices from current pose
        GetCurrentOpenGLCameraMatrix(Twc, Ow);

        // Handle camera following logic
        if (menuFollowCamera && bFollow)
        {
            if (bCameraView)
                s_cam.Follow(Twc);          // Follow camera pose
            else
                s_cam.Follow(Ow);           // Follow world origin with IMU orientation
        }
        else if (menuFollowCamera && !bFollow)
        {
            // Reset to following mode with appropriate view
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
        {
            bFollow = false;                // Disable following for manual navigation
        }

        // Handle camera view switch
        if (menuCamView)
        {
            menuCamView = false;
            bCameraView = true;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, -1, -3, 0, 0, 0, 0.0, -1.0, 0.0));
            s_cam.Follow(Twc);
        }

        // Handle top view switch (requires IMU initialization)
        if (menuTopView && mpMap->isImuInitialized())
        {
            menuTopView = false;
            bCameraView = false;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024, 768, 3000, 3000, 512, 389, 0.1, 10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, 0.01, 50, 0, 0, 0, 0.0, 0.0, 1.0));
            s_cam.Follow(Ow);
        }

        // Update debug control flags
        mbStepByStep = menuStepByStep;
        if (menuStep)
        {
            mbStep = true;
            menuStep = false;
        }

        // Activate 3D view and set background
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);  // White background

        // Render current camera pose
        DrawCurrentCamera(Twc);

        // Render keyframes and connectivity graph
        if (menuShowKeyFrames || menuShowGraph || menuShowInertialGraph)
            DrawKeyFrames(menuShowKeyFrames, menuShowGraph, menuShowInertialGraph);

        // Update visualization flags and render map elements

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

        // Finish 3D rendering
        pangolin::FinishFrame();

        // Generate and display 2D frame visualization
        cv::Mat toShow = DrawFrame();
        cv::imshow("Current Frame", toShow);
        cv::waitKey(30);                    // 30ms delay for ~33 FPS

        // Check for termination request
        if (mbFinishRequested)
        {
            mbFinishRequested = false;
            break;
        }
    }
}

/**
 * @brief Request viewer termination and wait for completion
 * 
 * Algorithm:
 * - Sets termination flag to signal the rendering loop
 * - Blocks until viewer thread completes cleanup
 * - Ensures graceful shutdown of OpenGL context
 */
void MSViewing::RequestFinish()
{
    mbFinishRequested = true;
    while (mbFinishRequested == true)
        usleep(3000);                   // Wait 3ms between checks
    cout << "PPG-SLAM Viewer: Visualization terminated." << endl;
}

/**
 * @brief Generate 2D visualization of current frame with feature overlay
 * @return OpenCV Mat containing annotated frame image
 * 
 * Algorithm:
 * - Thread-safely copies current frame data and tracking results
 * - Overlays different feature types with color coding:
 *   - Yellow circles: successfully tracked map points
 *   - Red lines: colinear point relationships
 *   - Green lines/circles: detected edges
 * - Handles both initialization and tracking phases
 * - Adds informational text overlay with tracking statistics
 */
cv::Mat MSViewing::DrawFrame()
{
    // Local copies for thread-safe access
    cv::Mat im;
    vector<KeyPointEx> vIniKeys;
    vector<int> vIniMatches;
    vector<KeyPointEx> vCurrentKeys;
    vector<KeyEdge> vCurrentEdges;
    vector<bool> vbOutliers;
    vector<MapPoint*> vpMapPoints;
    vector<MapEdge*> vpMapEdges;
    SE3<float> Tcw;
    int state;

    // Copy variables within scoped mutex for thread safety
    {
        unique_lock<mutex> lock(mMutex);
        state = mState;
        mIm.copyTo(im);

        // Copy initialization data if in initialization phase
        if (mState == NOT_INITIALIZED)
        {
            vIniKeys = mvIniKeys;
            vIniMatches = mvIniMatches;
        }

        // Copy current frame tracking data
        vCurrentKeys = mvCurrentKeys;
        vCurrentEdges = mvCurrentEdges;
        vbOutliers = mvbOutliers;
        vpMapPoints = mvpMapPoints;
        vpMapEdges = mvpMapEdges;

        Tcw = mTcw;
    }

    // Ensure image is in BGR format for color overlay
    if (im.channels() < 3)
        cvtColor(im, im, cv::COLOR_GRAY2BGR);

    // Resize image to original dimensions (no-op if already correct size)
    cv::resize(im, im, cv::Size(im.cols, im.rows), cv::INTER_LINEAR);

    mnTracked = 0;  // Reset tracked feature counter

    // Draw colinear relationships if enabled
    if (mbShowColine)
    {
        for (const KeyPointEx& kp : vCurrentKeys)
        {
            // Draw lines connecting colinear points (red color)
            for (const std::pair<unsigned int, unsigned int>& cpt : kp.mvColine)
            {
                cv::Point2f pt1(vCurrentKeys[cpt.first].mPos[0], vCurrentKeys[cpt.first].mPos[1]);
                cv::Point2f pt2(vCurrentKeys[cpt.second].mPos[0], vCurrentKeys[cpt.second].mPos[1]);
                cv::line(im, pt1, pt2, cv::Scalar(20, 20, 255), 2);  // Red line
            }
        } 
    }

    // Draw edge features if enabled
    if (mbShowEdge)
    {
        for (const KeyEdge& ke : vCurrentEdges)
        {
            const KeyPointEx &kp1 = vCurrentKeys[ke.startIdx];
            const KeyPointEx &kp2 = vCurrentKeys[ke.endIdx];
            
            // Draw edge as green line with endpoint circles
            cv::line(im, cv::Point2f(kp1.mPos[0], kp1.mPos[1]), 
                         cv::Point2f(kp2.mPos[0], kp2.mPos[1]), 
                         cv::Scalar(0, 255, 0), 1);              // Green line
            cv::circle(im, cv::Point2f(kp1.mPos[0], kp1.mPos[1]), 3, cv::Scalar(0, 255, 0), -1);
            cv::circle(im, cv::Point2f(kp2.mPos[0], kp2.mPos[1]), 3, cv::Scalar(0, 255, 0), -1);
        }
    }
    // Draw tracked map points as yellow circles
    for (unsigned int i = 0; i < vCurrentKeys.size(); i++)
    {
        MapPoint* pMP = vpMapPoints[i];
        if (pMP == nullptr || pMP->isBad())
            continue;
            
        // Only draw inliers (successfully tracked points)
        if (!vbOutliers[i])
        {        
            if (mbShowPoint)
                cv::circle(im, cv::Point2f(vCurrentKeys[i].mPos[0], vCurrentKeys[i].mPos[1]), 
                          3, cv::Scalar(0, 255, 255), 1);  // Yellow circle
            mnTracked++;
        }
    }

    // Add text overlay with tracking information
    cv::Mat imWithInfo;
    DrawTextInfo(im, state, imWithInfo);
    return imWithInfo;
}


/**
 * @brief Add text overlay with tracking information
 * @param im Input image
 * @param nState Current tracking state
 * @param imText Output image with text overlay
 * 
 * Algorithm:
 * - Displays different messages based on tracking state
 * - Shows keyframe count, map point count, and tracking matches
 * - Creates text overlay area at bottom of image
 */
void MSViewing::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    
    // Generate state-specific information text
    if (nState == NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if (nState == OK)
    {
        int nKFs = mpMap->KeyFramesInMap();
        int nMPs = mpMap->MapPointsInMap();
        s << " KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
    }
    else if (nState == LOST)
        s << " TRACK LOST. TRYING TO RELOCALIZE";

    // Calculate text dimensions and create overlay area
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);
    
    // Create output image with text area at bottom
    imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
    im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
    imText.rowRange(im.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
    
    // Draw text with white color
    cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5), 
                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
}

/**
 * @brief Update current frame data for visualization
 * @param F Current frame containing tracking results
 * 
 * Algorithm:
 * - Thread-safely copies frame data including image, keypoints, and edges
 * - Updates tracking state and pose information
 * - Handles initialization phase data separately
 * - Copies map point associations and outlier flags
 */
void MSViewing::UpdateFrame(Frame &F)
{
    unique_lock<mutex> lock(mMutex);
    
    // Copy frame image
    F.srcMat.copyTo(mIm);
    
    // Copy current frame features
    mvCurrentKeys = F.mvKeys;
    mvCurrentEdges = F.mvKeyEdges;

    // Handle initialization phase data
    if (MSTracking::get().mLastProcessedState == NOT_INITIALIZED)
    {
        mvIniKeys = F.mvKeys;
        mvIniMatches = MSTracking::get().mvIniMatches;
    }

    // Copy tracking results
    mvbOutliers = F.mvbOutlier;
    mvpMapPoints = F.mvpMapPoints;
    mvpMapEdges = F.mvpMapEdges;

    // Update pose and state
    mTcw = F.GetPose();
    mState = static_cast<int>(MSTracking::get().mLastProcessedState);
    mnCurFrameID = F.mnId;
}

/**
 * @brief Render all map points in 3D view
 * 
 * Algorithm:
 * - Retrieves all map points from the current map
 * - Implements time-based alpha blending for fading effect
 * - Filters out bad (outlier) map points
 * - Uses temporal visibility decay to highlight recent observations
 * - Renders points as small black dots with variable transparency
 */
void MSViewing::DrawMapPoints()
{
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    if (vpMPs.empty())
        return;

    glPointSize(2);
    glBegin(GL_POINTS);
    
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        if (vpMPs[i]->isBad())
            continue;
            
        Eigen::Matrix<float, 3, 1> pos = vpMPs[i]->GetWorldPos();

        // Calculate time-based fading (18 second fade duration)
        std::chrono::steady_clock::time_point tnow = chrono::steady_clock::now();
        double ts = std::chrono::duration_cast<std::chrono::duration<double>>(
            tnow - vpMPs[i]->startTime).count();
            
        if (mbunFaded)
            ts = 0;  // Disable fading if unfaded mode is enabled
            
        // Apply temporal alpha blending
        if (ts >= 18)
            glColor4f(0.0, 0.0, 0.0, 0.1);                    // Very faded for old points
        else
            glColor4f(0.0, 0.0, 0.0, (20 - ts) / 20.0);       // Linear fade

        glVertex3f(pos(0), pos(1), pos(2));
    }
    glEnd();
}

/**
 * @brief Render colinear point relationships
 * 
 * Algorithm:
 * - Iterates through all map points to find colinear relationships
 * - For each valid colinear triplet (start, middle, end points):
 *   - Draws red line connecting start and end points
 *   - Highlights all three points with black dots
 * - Filters out bad or invalid colinear relationships
 * - Uses distinct red color scheme for easy identification
 */
void MSViewing::DrawMapColines()
{
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    if (vpMPs.empty())
        return;

    // Iterate through all map points to find colinear relationships
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        if (vpMPs[i]->isBad())
            continue;
            
        std::vector<MapColine*> vMPCLs = vpMPs[i]->getColinearity();

        for (auto pMC : vMPCLs)
        {
            if (pMC->isBad() || !pMC->mbValid)
                continue;
                
            // Get positions of colinear triplet: middle, start, end
            Eigen::Vector3f pos = pMC->mpMPm->GetWorldPos();   // Middle point
            Eigen::Vector3f pos1 = pMC->mpMPs->GetWorldPos();  // Start point
            Eigen::Vector3f pos2 = pMC->mpMPe->GetWorldPos();  // End point
            
            // Draw line connecting start and end points
            glLineWidth(2);
            glBegin(GL_LINES);
            glColor3f(1, 0, 0);  // Red color for colines
            glVertex3f(pos1(0), pos1(1), pos1(2));
            glVertex3f(pos2(0), pos2(1), pos2(2));
            glEnd();

            // Highlight all three points of the colinear relationship
            glPointSize(3);
            glBegin(GL_POINTS);
            glColor3f(0.0, 0.0, 0.0);  // Black points
            glVertex3f(pos(0), pos(1), pos(2));    // Middle point
            glVertex3f(pos1(0), pos1(1), pos1(2)); // Start point
            glVertex3f(pos2(0), pos2(1), pos2(2)); // End point
            glEnd();
        }
    }
}

/**
 * @brief Render map edges (line features) in 3D view
 * 
 * Algorithm:
 * - Retrieves all map edges from the current map
 * - Different rendering for current vs. historical edges:
 *   - Current frame edges: bright green lines (width 2)
 *   - Historical edges: blue lines with time-based fading (width 1)
 * - Implements time-based alpha blending for temporal visualization
 * - Highlights endpoints of historical edges with black dots
 */
void MSViewing::DrawMapEdges()
{
    Map* pActiveMap = mpMap;
    if (!pActiveMap)
        return;

    const vector<MapEdge*> &vpMEs = pActiveMap->GetAllMapEdges();
    if (vpMEs.empty())
        return;

    for (size_t i = 0, iend = vpMEs.size(); i < iend; i++)
    {
        if (vpMEs[i]->isBad() || !vpMEs[i]->mbValid)
            continue;
            
        MapPoint* pMPs = vpMEs[i]->mpMPs;  // Start point
        MapPoint* pMPe = vpMEs[i]->mpMPe;  // End point

        Eigen::Matrix<float, 3, 1> pos1 = pMPs->GetWorldPos();
        Eigen::Matrix<float, 3, 1> pos2 = pMPe->GetWorldPos();

        // Render current frame edges with bright green
        if (vpMEs[i]->trackedFrameId == mnCurFrameID)
        {
            glLineWidth(2);
            glBegin(GL_LINES);
            glColor3f(0, 1, 0);  // Bright green for current edges
            glVertex3f(pos1(0), pos1(1), pos1(2));
            glVertex3f(pos2(0), pos2(1), pos2(2));
            glEnd();
        }
        else
        {
            // Render historical edges with time-based fading
            glLineWidth(1);
            glBegin(GL_LINES);
            
            // Calculate time-based fading
            std::chrono::steady_clock::time_point tnow = chrono::steady_clock::now();
            double ts = std::chrono::duration_cast<std::chrono::duration<double>>(
                tnow - vpMEs[i]->startTime).count();
                
            if (mbunFaded)
                ts = 0;  // Disable fading if unfaded mode is enabled
                
            // Apply temporal alpha blending (blue color scheme)
            if (ts >= 18)
                glColor4f(0.2, 0.2, 0.6, 0.1);                    // Very faded blue
            else
                glColor4f(0.2, 0.2, 0.6, (20 - ts) / 20.0);       // Linear fade blue
                
            glVertex3f(pos1(0), pos1(1), pos1(2));
            glVertex3f(pos2(0), pos2(1), pos2(2));
            glEnd();

            // Highlight endpoints of historical edges
            glPointSize(3);
            glBegin(GL_POINTS);
            if (ts >= 18)
                glColor4f(0.0, 0.0, 0.0, 0.1);                    // Very faded black points
            else
                glColor4f(0.0, 0.0, 0.0, (20 - ts) / 20.0);       // Linear fade black points
                
            glVertex3f(pos1(0), pos1(1), pos1(2));
            glVertex3f(pos2(0), pos2(1), pos2(2));
            glEnd();
        }
    }
}

/**
 * @brief Render keyframe poses and connectivity graph
 * @param bDrawKF Whether to draw keyframe coordinate frames
 * @param bDrawGraph Whether to draw covisibility graph
 * @param bDrawInertialGraph Whether to draw IMU connectivity
 * 
 * Algorithm:
 * - Keyframes: Draws RGB coordinate axes for each keyframe pose
 * - Covisibility graph: Green lines connecting keyframes with shared observations
 * - Loop closure edges: Additional connections for detected loops
 * - Inertial graph: Magenta lines showing IMU sequential connections
 * - Uses OpenGL matrix transformations for efficient rendering
 */
void MSViewing::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph)
{
    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    
    // Draw keyframe coordinate axes
    if (bDrawKF)
    {
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();

            // Apply keyframe transformation
            glPushMatrix();
            glMultMatrixf((GLfloat*)Twc.data());
            
            // Draw RGB coordinate axes (10cm length)
            glLineWidth(1);
            glBegin(GL_LINES);
            
            // X-axis (red)
            glColor3f(1.0f, 0.0f, 0.0f);
            glVertex3f(0, 0, 0);
            glVertex3f(0.1, 0, 0);
            
            // Y-axis (green)
            glColor3f(0.0f, 1.0f, 0.0f);
            glVertex3f(0, 0, 0);
            glVertex3f(0, 0.1, 0);
            
            // Z-axis (blue)
            glColor3f(0.0f, 0.0f, 1.0f);
            glVertex3f(0, 0, 0);
            glVertex3f(0, 0, 0.1);
            
            glEnd();
            glPopMatrix();
        }
    }

    // Draw covisibility graph
    if (bDrawGraph)
    {
        glLineWidth(0.5);
        glColor4f(0.0f, 1.0f, 0.0f, 0.6f);  // Semi-transparent green
        glBegin(GL_LINES);

        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            // Draw covisibility connections (minimum 10 shared observations)
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(10);
            Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();
            
            if (!vCovKFs.empty())
            {
                for (vector<KeyFrame*>::const_iterator vit = vCovKFs.begin(), vend = vCovKFs.end(); 
                     vit != vend; vit++)
                {
                    Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow(0), Ow(1), Ow(2));
                    glVertex3f(Ow2(0), Ow2(1), Ow2(2));
                }
            }
            
            // Draw loop closure edges
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for (set<KeyFrame*>::iterator sit = sLoopKFs.begin(), send = sLoopKFs.end(); 
                 sit != send; sit++)
            {
                // Avoid duplicate drawing (only draw if current KF has higher ID)
                if ((*sit)->mnId < vpKFs[i]->mnId)
                    continue;
                    
                Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow(0), Ow(1), Ow(2));
                glVertex3f(Owl(0), Owl(1), Owl(2));
            }
        }
        glEnd();
    }

    // Draw inertial graph (IMU sequential connections)
    if (bDrawInertialGraph && mpMap->isImuInitialized())
    {
        glLineWidth(0.6);
        glColor4f(1.0f, 0.0f, 1.0f, 0.6f);  // Semi-transparent magenta
        glBegin(GL_LINES);

        // Draw sequential connections between keyframes
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            Eigen::Vector3f Ow = pKFi->GetCameraCenter();
            KeyFrame* pNext = pKFi->mNextKF;
            
            if (pNext)
            {
                Eigen::Vector3f Owp = pNext->GetCameraCenter();
                glVertex3f(Ow(0), Ow(1), Ow(2));
                glVertex3f(Owp(0), Owp(1), Owp(2));
            }
        }
        glEnd();
    }
}

/**
 * @brief Render current camera pose as wireframe pyramid
 * @param Twc OpenGL transformation matrix for camera pose
 * 
 * Algorithm:
 * - Draws camera frustum as blue wireframe pyramid
 * - Shows camera orientation and position in 3D space
 * - Uses standard camera frustum geometry (width=0.4m, height=0.3m, depth=0.2m)
 * - Applies OpenGL matrix transformation for correct positioning
 */
void MSViewing::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    // Apply camera transformation
    glPushMatrix();
    glMultMatrixd(Twc.m);
    
    // Draw camera frustum with blue color
    glLineWidth(2);
    glColor3f(0.0f, 0.0f, 1.0f);  // Blue color
    glBegin(GL_LINES);

    // Camera frustum dimensions
    const float w = 0.2f;   // Half width
    const float h = 0.15f;  // Half height  
    const float z = 0.2f;   // Depth

    // Draw lines from camera center to frustum corners
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    // Draw frustum rectangle (far plane)
    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);

    glEnd();
    glPopMatrix();
}


/**
 * @brief Update current camera pose for visualization
 * @param Tcw Camera-to-world transformation (SE3)
 * 
 * Algorithm:
 * - Thread-safely updates camera pose with inverse transformation
 * - Converts from camera-to-world to world-to-camera for rendering
 * - Used by tracking system to update viewer with latest pose estimate
 */
void MSViewing::SetCurrentCameraPose(const SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutex);
    mCameraPose = Tcw.inverse();  // Convert to world-to-camera for rendering
}

/**
 * @brief Convert current pose to OpenGL matrices
 * @param M Output camera transformation matrix
 * @param MOw Output world origin transformation
 * 
 * Algorithm:
 * - Thread-safely retrieves current camera pose
 * - Converts SE3 transformation to column-major OpenGL format
 * - M: Full 4x4 transformation matrix for camera rendering
 * - MOw: Translation-only matrix for world origin visualization
 */
void MSViewing::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
{
    Eigen::Matrix4f Twc;
    
    // Thread-safe pose retrieval
    {
        unique_lock<mutex> lock(mMutex);
        Twc = mCameraPose.matrix();
    }
    
    // Convert to OpenGL column-major format
    for (int i = 0; i < 4; i++) 
    {
        M.m[4*i]   = Twc(0, i);     // Column 0
        M.m[4*i+1] = Twc(1, i);     // Column 1
        M.m[4*i+2] = Twc(2, i);     // Column 2
        M.m[4*i+3] = Twc(3, i);     // Column 3
    }
    
    // World origin transformation (translation only)
    MOw.SetIdentity();
    MOw.m[12] = Twc(0, 3);  // X translation
    MOw.m[13] = Twc(1, 3);  // Y translation
    MOw.m[14] = Twc(2, 3);  // Z translation
}

/**
 * @brief Save complete trajectory to file in TUM RGB-D format
 * @param filename Output file path
 * 
 * Algorithm:
 * - Exports all frame poses with timestamps in TUM format
 * - Handles IMU body frame transformations when available
 * - Sorts keyframes by ID for temporal consistency
 * - Transforms relative poses to absolute world coordinates
 * - Output format: timestamp tx ty tz qw qx qy qz
 */
void MSViewing::SaveTrajectory(const std::string &filename)
{
    cout << endl << "Saving trajectory to " << filename << " ..." << endl;

    // Get and sort keyframes by ID
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), [](KeyFrame* pKF1, KeyFrame* pKF2) { 
        return pKF1->mnId < pKF2->mnId; 
    });

    // Get initial IMU pose if available
    SE3f Twb;
    Twb = vpKFs[0]->GetImuPose();

    // Open output file
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Get tracking data iterators
    list<KeyFrame*>::iterator lRit = MSTracking::get().mlpReferences.begin();
    list<double>::iterator lT = MSTracking::get().mlFrameTimes.begin();

    // Process each frame pose
    for (auto lit = MSTracking::get().mlRelativeFramePoses.begin(), 
         lend = MSTracking::get().mlRelativeFramePoses.end(); 
         lit != lend; lit++, lRit++, lT++)
    {
        KeyFrame* pKF = *lRit;
        SE3f Trw;
        
        if (!pKF)
            continue;
            
        // Handle bad keyframes by accumulating transformations
        while (pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->mPrevKF;
        }
        
        Trw = Trw * pKF->GetPose() * Twb;

        // Transform to IMU body frame
        SE3f Twb_final = (pKF->mpImuCalib->mTbc * (*lit) * Trw).inverse();
        Eigen::Quaternionf q = Twb_final.unit_quaternion();
        Eigen::Vector3f twb = Twb_final.translation();
        
        // Write pose in TUM format
        f << setprecision(6) << (*lT) << " " 
          << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " 
          << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << endl;
    }
    
    f.close();
    cout << "Trajectory saved successfully." << endl;
}

/**
 * @brief Save keyframe trajectory to file in TUM RGB-D format
 * @param filename Output file path
 * 
 * Algorithm:
 * - Exports only keyframe poses (subset of all frames)
 * - Uses keyframe timestamps for temporal accuracy
 * - Sorts keyframes by ID for consistent ordering
 * - Transforms poses to IMU body frame when available
 * - Output format: timestamp tx ty tz qw qx qy qz
 */
void MSViewing::SaveKeyFrameTrajectory(const std::string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    // Get and sort keyframes by ID
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), [](KeyFrame* pKF1, KeyFrame* pKF2) { 
        return pKF1->mnId < pKF2->mnId; 
    });

    // Open output file
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Process each keyframe
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

        // Skip bad keyframes
        if (!pKF || pKF->isBad())
            continue;
            
        // Get IMU body frame pose
        SE3f Twb = pKF->GetImuPose();
        Eigen::Quaternionf q = Twb.unit_quaternion();
        Eigen::Vector3f twb = Twb.translation();
        
        // Write keyframe pose in TUM format
        f << setprecision(6) << pKF->mTimeStamp << " " 
          << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " 
          << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << endl;
    }
    
    f.close();
    cout << "Keyframe trajectory saved successfully." << endl;
}