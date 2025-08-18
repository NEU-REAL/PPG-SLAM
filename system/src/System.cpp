/**
 * @file System.cpp
 * @brief Implementation of the main SLAM system class
 * @author PPG-SLAM Team
 */

#include "System.h"
#include <thread>
#include <pangolin/pangolin.h>

using namespace std;

System::System(const string &strVocFile, const string &strSettingsFile, const string &strNet, const bool bUseViewer)
    : mbShutDown(false)
{
    // Load vocabulary for place recognition
    std::cout << std::endl << "Loading Vocabulary. This could take a while..." << std::endl;
    mpVocabulary = new DBoW3::Vocabulary(strVocFile);
    if(mpVocabulary->empty())
    {
        std::cerr << "Wrong path to vocabulary." << std::endl;
        std::cerr << "Failed to open at: " << strVocFile << std::endl;
        exit(-1);
    }
    std::cout << "Vocabulary loaded!" << std::endl << std::endl;

    // Load and validate settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "Failed to open settings file at: " << strSettingsFile << std::endl;
        exit(-1);
    }

    // Read camera parameters
    int width = fsSettings["Camera.width"];
    int height = fsSettings["Camera.height"];
    float fps = fsSettings["Camera.fps"];
    
    // Initialize camera model based on type
    GeometricCamera* pCam = nullptr;
    std::string cameraModel = fsSettings["Camera.type"];
    std::vector<float> vCalibration(8, 0);
    
    if (cameraModel == "PinHole")
    {
        // Pinhole camera model parameters
        vCalibration[0] = fsSettings["Camera.fx"];  // focal length x
        vCalibration[1] = fsSettings["Camera.fy"];  // focal length y
        vCalibration[2] = fsSettings["Camera.cx"];  // principal point x
        vCalibration[3] = fsSettings["Camera.cy"];  // principal point y
        vCalibration[4] = fsSettings["Camera.k1"];  // radial distortion k1
        vCalibration[5] = fsSettings["Camera.k2"];  // radial distortion k2
        vCalibration[6] = fsSettings["Camera.p1"];  // tangential distortion p1
        vCalibration[7] = fsSettings["Camera.p2"];  // tangential distortion p2
        pCam = new Pinhole(vCalibration, width, height, fps);
    }
    else if (cameraModel == "KannalaBrandt8")
    {
        // Fisheye camera model parameters
        vCalibration[0] = fsSettings["Camera.fx"];  // focal length x
        vCalibration[1] = fsSettings["Camera.fy"];  // focal length y
        vCalibration[2] = fsSettings["Camera.cx"];  // principal point x
        vCalibration[3] = fsSettings["Camera.cy"];  // principal point y
        vCalibration[4] = fsSettings["Camera.k0"];  // fisheye parameter k0
        vCalibration[5] = fsSettings["Camera.k1"];  // fisheye parameter k1
        vCalibration[6] = fsSettings["Camera.k2"];  // fisheye parameter k2
        vCalibration[7] = fsSettings["Camera.k3"];  // fisheye parameter k3
        pCam = new KannalaBrandt8(vCalibration, width, height, fps);
    }
    else
    {
        std::cerr << "Unknown camera model: " << cameraModel << std::endl;
        exit(-1);
    }

    // Read IMU parameters
    float ng = fsSettings["IMU.NoiseGyro"];     // gyroscope noise
    float na = fsSettings["IMU.NoiseAcc"];      // accelerometer noise
    float wg = fsSettings["IMU.GyroWalk"];      // gyroscope random walk
    float wa = fsSettings["IMU.AccWalk"];       // accelerometer random walk
    float imuFreq = fsSettings["IMU.Frequency"]; // IMU frequency
    
    // Read IMU-camera transformation matrix
    cv::Mat cvTbc = fsSettings["IMU.T_b_c1"].mat();
    Eigen::Matrix<double,3,3> eigenR;
    eigenR << cvTbc.at<float>(0,0), cvTbc.at<float>(0,1), cvTbc.at<float>(0,2),
              cvTbc.at<float>(1,0), cvTbc.at<float>(1,1), cvTbc.at<float>(1,2),
              cvTbc.at<float>(2,0), cvTbc.at<float>(2,1), cvTbc.at<float>(2,2);
    
    Eigen::Quaternionf q(eigenR.cast<float>());
    Eigen::Matrix<float,3,1> t;
    t << cvTbc.at<float>(0, 3), cvTbc.at<float>(1, 3), cvTbc.at<float>(2, 3);
    SE3f Tbc = SE3<float>(q, t);
    
    // Scale noise parameters by sqrt of frequency
    const float sf = sqrt(imuFreq);
    IMU::Calib *pImu = new IMU::Calib(Tbc, ng * sf, na * sf, wg / sf, wa / sf, imuFreq);
    
    // Create map with all calibrated sensors
    MapEdge::viewCosTh = fsSettings["ViewCosTh"];
    Map::imuIniTm = fsSettings["IMU.IniTime"];
    mpMap = new Map(pCam, pImu, mpVocabulary);

    // Launch tracking thread with neural network
    MSTracking::get().Launch(mpMap, strNet);

    // Initialize and launch local mapping thread
    MSLocalMapping::get().Launch(mpMap);

    // Initialize and launch loop closing thread
    MSLoopClosing::get().Launch(mpMap, true);
    
    // Launch viewer thread if requested
    if(bUseViewer)
        MSViewing::get().Launch(mpMap);

    // Initialize random seed for reproducible results
    srand(0); 
}

SE3f System::TrackMonocular(const cv::Mat &im, const double &timestamp, 
                           const vector<IMU::Point>& vImuMeas, string filename)
{
    // Check if system is shutting down
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbShutDown)
            return SE3f();  // Return empty pose if shutdown
    }
    
    // Clone input image to avoid modification of original
    cv::Mat imToFeed = im.clone();

    // Feed IMU measurements to tracking thread
    for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
        MSTracking::get().GrabImuData(vImuMeas[i_imu]);

    // Process monocular image and get camera pose
    SE3f Tcw = MSTracking::get().GrabImageMonocular(imToFeed, timestamp, filename);

    return Tcw;
}

void System::Shutdown()
{
    // Set shutdown flag to prevent new tracking requests
    {
        unique_lock<mutex> lock(mMutexReset);
        mbShutDown = true;
    }
    
    std::cout << "Shutting down SLAM system..." << std::endl;

    // Request all threads to finish gracefully
    MSViewing::get().RequestFinish();
    MSLocalMapping::get().RequestFinish();
    MSLoopClosing::get().RequestFinish();

    // Save trajectory data before complete shutdown
    MSViewing::get().SaveTrajectory("CameraTrajectory.txt");
    MSViewing::get().SaveKeyFrameTrajectory("KeyFrameTrajectory.txt");
    
    std::cout << "SLAM system shutdown complete." << std::endl;
}

