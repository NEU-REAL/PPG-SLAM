#include "System.h"
#include <thread>
#include <pangolin/pangolin.h>

System::System(const string &strVocFile, const string &strSettingsFile,const string &strNet, const bool bUseViewer): mbShutDown(false)
{
    //Load Vocabulary
    cout << endl << "Loading Vocabulary. This could take a while..." << endl;
    mpVocabulary = new DBoW3::Vocabulary(strVocFile);
    if(mpVocabulary->empty())
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }
    int width = fsSettings["Camera.width"].operator int();
    int height = fsSettings["Camera.height"].operator int();
    float fps = fsSettings["Camera.fps"].real();
    GeometricCamera* pCam = nullptr;
    std::string cameraModel = fsSettings["Camera.type"].string();
    std::vector<float> vCalibration(8,0);
    if (cameraModel == "PinHole")
    {
        vCalibration[0] = fsSettings["Camera.fx"].real();
        vCalibration[1] = fsSettings["Camera.fy"].real();
        vCalibration[2] = fsSettings["Camera.cx"].real();
        vCalibration[3] = fsSettings["Camera.cy"].real();
        vCalibration[4] = fsSettings["Camera.k1"].real();
        vCalibration[5] = fsSettings["Camera.k2"].real();
        vCalibration[6] = fsSettings["Camera.p1"].real();
        vCalibration[7] = fsSettings["Camera.p2"].real();
        pCam = new Pinhole(vCalibration, width, height, fps);
    }
    else if (cameraModel == "KannalaBrandt8")
    {
        vCalibration[0] = fsSettings["Camera.fx"].real();
        vCalibration[1] = fsSettings["Camera.fy"].real();
        vCalibration[2] = fsSettings["Camera.cx"].real();
        vCalibration[3] = fsSettings["Camera.cy"].real();
        vCalibration[4] = fsSettings["Camera.k0"].real();
        vCalibration[5] = fsSettings["Camera.k1"].real();
        vCalibration[6] = fsSettings["Camera.k2"].real();
        vCalibration[7] = fsSettings["Camera.k3"].real();
        pCam = new KannalaBrandt8(vCalibration, width, height, fps);
    }
    float ng = fsSettings["IMU.NoiseGyro"].real();
    float na = fsSettings["IMU.NoiseAcc"].real();
    float wg = fsSettings["IMU.GyroWalk"].real();
    float wa = fsSettings["IMU.AccWalk"].real();
    float imuFreq = fsSettings["IMU.Frequency"].real();
    cv::Mat cvTbc = fsSettings["IMU.T_b_c1"].mat();
    Eigen::Matrix<double,3,3> eigenR;
    eigenR <<   cvTbc.at<float>(0,0), cvTbc.at<float>(0,1), cvTbc.at<float>(0,2),
                cvTbc.at<float>(1,0), cvTbc.at<float>(1,1), cvTbc.at<float>(1,2),
                cvTbc.at<float>(2,0), cvTbc.at<float>(2,1), cvTbc.at<float>(2,2);
    Eigen::Quaternionf q(eigenR.cast<float>());
    Eigen::Matrix<float,3,1> t;
    t <<  cvTbc.at<float>(0, 3), cvTbc.at<float>(1, 3), cvTbc.at<float>(2, 3);
    SE3f Tbc = SE3<float>(q,t);
    const float sf = sqrt(imuFreq);
    IMU::Calib *pImu = new IMU::Calib(Tbc, ng * sf, na * sf, wg / sf, wa / sf, imuFreq);
    
    //Create map
    MapEdge::viewCosTh = fsSettings["ViewCosTh"].real();
    Map::imuIniTm = fsSettings["IMU.IniTime"].real();
    mpMap = new Map(pCam, pImu, mpVocabulary);

    MSTracking::get().Launch(mpMap,strNet);

    //Initialize the Local Mapping thread and launch
    MSLocalMapping::get().Launch(mpMap);

    //Initialize the Loop Closing thread and launch
    MSLoopClosing::get().Launch(mpMap, true);
    
    if(bUseViewer)
        MSViewing::get().Launch(mpMap);

	srand(0); 
}

SE3f System::TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{

    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbShutDown)
            return SE3f();
    }
    cv::Mat imToFeed = im.clone();

    for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
        MSTracking::get().GrabImuData(vImuMeas[i_imu]);

    SE3f Tcw = MSTracking::get().GrabImageMonocular(imToFeed,timestamp,filename);

    return Tcw;
}

void System::Shutdown()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbShutDown = true;
    }
    cout << "Shutdown" << endl;

    MSViewing::get().RequestFinish();
    MSLocalMapping::get().RequestFinish();
    MSLoopClosing::get().RequestFinish();

    // Save camera trajectory
    MSViewing::get().SaveTrajectory("CameraTrajectory.txt");
    MSViewing::get().SaveKeyFrameTrajectory("KeyFrameTrajectory.txt");
}

