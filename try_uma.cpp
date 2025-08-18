#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>

#include <opencv2/core/core.hpp>

#include "system/include/System.h"
#include "IMU.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cerr << " Argument mismatched  " << std::endl;
        return -1;
    }
    cout << "EXE: " << string(argv[0]) << endl;
    cout << "Vocabulary: " << string(argv[1]) << endl;
    cout << "Config: " << string(argv[2]) << endl;
    cout << "Net folder: " << string(argv[3]) << endl;
    cout << "Dataset: " << string(argv[4]) << endl;

    vector<string> vImagePath;
    vector<double> vImageTimeStamp;
    vector<cv::Point3f> vImuAcc, vImuGyr;
    vector<double> vImuTimestamp;

    string dataFolder = string(argv[4]);
    // load images
    vImagePath.reserve(6000);
    vImageTimeStamp.reserve(6000);
    ifstream fimage;
    fimage.open(dataFolder+ "/cam0/data.csv");
    while(!fimage.eof())
    {
        string s;
        getline(fimage, s);
        if (!s.empty())
        {
            if (s[0] == '#')
                continue;
            int pos = s.find(',');
            string item = s.substr(0, pos);
            vImagePath.push_back(dataFolder + "/cam0/data/" + item + ".png");
            vImageTimeStamp.push_back(stod(item) * 1e-9);
        }
    }

    // load imus
    vImuAcc.reserve(60000);
    vImuGyr.reserve(60000);
    vImuTimestamp.reserve(60000);
    ifstream fImu;
    fImu.open(dataFolder + "/imu0/data.csv");
    
    while(!fImu.eof())
    {
        string s;
        getline(fImu, s);
        if (s[0] == '#')
            continue;
        if (!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != string::npos)
            {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);
            vImuTimestamp.push_back(data[0] * 1e-9);
            vImuAcc.push_back(cv::Point3f(data[4], data[5], data[6]));
            vImuGyr.push_back(cv::Point3f(data[1], data[2], data[3]));
        }
    }

    if(vImageTimeStamp.empty() || vImuTimestamp.empty() )
    {
        cerr << "ERROR: Failed to load images or IMU for sequence" << endl;
        return 1;
    }

    // Find first imu to be considered, supposing imu measurements start first
    int first_imu(0);
    while(vImuTimestamp[first_imu]<=vImageTimeStamp[0])
        first_imu++;
    first_imu--;

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    System SLAM(argv[1],argv[2], argv[3], true);

    int proccIm=0;
    // Main loop
    cv::Mat im;
    vector<IMU::Point> vImuMeas;
    // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));

    for(unsigned int ni=0; ni<vImagePath.size(); ni++, proccIm++)
    {
        // Read image from file
        im = cv::imread(vImagePath[ni], 0); // CV_LOAD_IMAGE_UNCHANGED);
        if (im.empty())
        {
            cerr << endl
                 << "Failed to load image at: " << vImagePath[ni] << endl;
            return 1;
        }
        double tframe = vImageTimeStamp[ni];
        vImuMeas.clear();
        if (ni > 0)
        {
            while (vImuTimestamp[first_imu] <= vImageTimeStamp[ni])
            {
                vImuMeas.push_back(IMU::Point(
                    vImuAcc[first_imu].x, vImuAcc[first_imu].y, vImuAcc[first_imu].z,
                    vImuGyr[first_imu].x, vImuGyr[first_imu].y, vImuGyr[first_imu].z,
                    vImuTimestamp[first_imu]));
                first_imu++;
            }
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        SLAM.TrackMonocular(im, tframe, vImuMeas); // TODO change to monocular_inertial
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        // Wait to load the next frame
        if (ni + 2 < vImageTimeStamp.size())
        {
            double T = 0.05; // 20fps TODO: modify it for acceleration.
            if (ttrack < T)
                usleep((T - ttrack) * 1e6); // 1e6
        }
    }
    SLAM.Shutdown();
    return 0;
}

void LoadImages(const string &strImagePath, const string &strTimePath, vector<string> &vstrImages, vector<double> &vTimeStamps)
{
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    while (!fImu.eof())
    {
        string s;
        getline(fImu, s);
        if (s[0] == '#')
            continue;

        if (!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != string::npos)
            {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);
            vTimeStamps.push_back(data[0] / 1e9);
            vAcc.push_back(cv::Point3f(data[4], data[5], data[6]));
            vGyro.push_back(cv::Point3f(data[1], data[2], data[3]));
        }
    }
}
