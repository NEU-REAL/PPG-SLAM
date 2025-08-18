#include "System.h"
#include <cv_bridge/cv_bridge.h>

SlamNode::SlamNode() : Node("slam_node")
{
    cacheImu.clear();
    cacheImage.clear();
    cacheTimeStamp.clear();
    latestImuTs = 0;
    latestImageTs = 0;

    Load();
    imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("/ppg_slam/imu_raw", 1000, std::bind(&SlamNode::IMU_CB, this, std::placeholders::_1));
    image_sub = this->create_subscription<sensor_msgs::msg::Image>("/ppg_slam/image_raw", 100, std::bind(&SlamNode::Image_CB, this, std::placeholders::_1));
    m_timer_exe = this->create_wall_timer(10ms, std::bind(&SlamNode::timer, this));
    m_timer_map = this->create_wall_timer(500ms, std::bind(&SlamNode::timer_Map, this));
    
    trajectory_pub = this->create_publisher<nav_msgs::msg::Path>("trajectory", 100);
    image_pub = this->create_publisher<sensor_msgs::msg::Image>("image_result", 100);
    map_point_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("map_point", 1);
    map_edge_pub = this->create_publisher<visualization_msgs::msg::Marker>("map_edge", 1);
    map_coline_pub = this->create_publisher<visualization_msgs::msg::Marker>("map_coline", 1);
}

SlamNode::~SlamNode()
{
    cout << "Shutdown" << endl;
    MSLocalMapping::get().RequestFinish();
    MSLoopClosing::get().RequestFinish();

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    if(vpKFs.empty())
    {
        cout <<" map is empty "<<endl;
        return;
    }    
    
    // Save camera trajectory
    cout << endl << "Saving trajectory to CameraTrajectory.txt" << " ..." << endl;

    sort(vpKFs.begin(),vpKFs.end(), [](KeyFrame* pKF1, KeyFrame* pKF2){return pKF1->mnId < pKF2->mnId;});

    SE3f Twb;
    Twb = vpKFs[0]->GetImuPose();

    ofstream f;
    f.open("CameraTrajectory.txt");
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
    cout << endl << "End of saving trajectory.. "<< endl;
}

void SlamNode::Load()
{
    this->declare_parameter<std::string>("vocabulary", "install/ppg_slam/share/ppg_slam/Vocabulary/voc_euroc_9x3.gz");
    this->declare_parameter<std::string>("config", "install/ppg_slam/share/ppg_slam/config/EuRoC.yaml");
    this->declare_parameter<std::string>("net", "install/ppg_slam/share/ppg_slam/net");
    string strVocFile, strSettingsFile, netFile;
    if(!this->get_parameter<std::string>("vocabulary", strVocFile))
        RCLCPP_ERROR(this->get_logger(), "FAIL TO LOAD vocabulary!");
    if(!this->get_parameter<std::string>("config", strSettingsFile))
        RCLCPP_ERROR(this->get_logger(), "FAIL TO LOAD config!");
    if(!this->get_parameter<std::string>("net", netFile))
        RCLCPP_ERROR(this->get_logger(), "FAIL TO LOAD net!");

    //Load Vocabulary
    cout << endl << "Loading Vocabulary... ";
    mpVocabulary = new DBoW3::Vocabulary(strVocFile);
    if(mpVocabulary->empty())
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Check settings file
    cout << "Load settings ... ";
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }
    int width = fsSettings["Camera.width"].operator int();
    int height = fsSettings["Camera.height"].operator int();
    float fps = fsSettings["Camera.fps"].real();
    GeometricCamera* pCam;
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
    
    cout << "Settings Loaded!"<<endl;

    cout << "Create map ...";
    mpMap = new Map(pCam, pImu, mpVocabulary);
    MapEdge::viewCosTh = fsSettings["ViewCosTh"].real();
    Map::imuIniTm = fsSettings["IMU.IniTime"].real();
    cout << "done!"<<endl;

    cout << "Launch tracking module ...";
    MSTracking::get().Launch(mpMap, netFile);
    cout << "done!"<<endl;

    cout << "Launch map optimizing module ...";
    MSLocalMapping::get().Launch(mpMap);
    cout << "done!"<<endl;

    //Initialize the Loop Closing thread and launch
    cout << "Launch loop closing module ...";
    MSLoopClosing::get().Launch(mpMap, true);
    cout << "done!"<<endl;

	srand(0); 
}

void SlamNode::timer()
{
    cv::Mat srcMat;
    double ts;
    std::vector<IMU::Point> vmeas;
    {
        std::unique_lock<std::mutex> lock(mMutexCache);
        if(cacheImage.empty())
            return;
        srcMat = cacheImage.front();
        ts = cacheTimeStamp.front();
        if(ts > cacheImu.back().t)
            return;
        while (!cacheImu.empty() && cacheImu.front().t <= ts)
        {
            vmeas.push_back(cacheImu.front());
            cacheImu.pop_front();
        }
    }

    for(IMU::Point pt : vmeas)
        MSTracking::get().GrabImuData(pt);
    
    cv::Mat showMat = MSTracking::get().GrabImageMonocular(srcMat,ts);
    
    {
        std::unique_lock<std::mutex> lock(mMutexCache);
        cacheImage.pop_front();
        cacheTimeStamp.pop_front();
    }
    
    if (image_pub->get_subscription_count())
    {
        std_msgs::msg::Header hd;
        hd.frame_id = "image";
        hd.stamp = sec2stamp(ts);
        auto image_msg = cv_bridge::CvImage(hd,"rgb8",showMat).toImageMsg();
        image_pub->publish(*image_msg);
    }

    // publish 
    if (trajectory_pub->get_subscription_count())
    {
        vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
        if(vpKFs.empty())
            return;
        sort(vpKFs.begin(),vpKFs.end(), [](KeyFrame* pKF1, KeyFrame* pKF2){return pKF1->mnId < pKF2->mnId;});
        nav_msgs::msg::Path path;
        list<KeyFrame*>::iterator lRit = MSTracking::get().mlpReferences.begin();
        for(auto lit=MSTracking::get().mlRelativeFramePoses.begin(), lend=MSTracking::get().mlRelativeFramePoses.end();lit!=lend;lit++, lRit++)
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
            Trw = Trw * pKF->GetPose();

            SE3f Twb = (pKF->mpImuCalib->mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            geometry_msgs::msg::PoseStamped pose;
            pose.header.frame_id = "world";
            pose.header.stamp = sec2stamp(ts);
            pose.pose.position.x = twb(0);
            pose.pose.position.y = twb(1);
            pose.pose.position.z = twb(2);
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            pose.pose.orientation.w = q.w();
            path.poses.push_back(pose);
        } 
        path.header.frame_id = "world";
        path.header.stamp = sec2stamp(ts);
        if(!path.poses.empty())
            trajectory_pub->publish(path);
    }
    std::unique_lock<std::mutex> lock(mMutexPub);
    mvpAllMPs = mpMap->GetAllMapPoints();
    mvpAllMEs = mpMap->GetAllMapEdges();
    mvpAllCLs = mpMap->GetAllMapColines();
}

void SlamNode::IMU_CB(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    latestImuTs = stamp2sec(msg->header.stamp);
    IMU::Point imuPt(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
                     msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z,
                     latestImuTs);
    std::unique_lock<std::mutex> lock(mMutexCache);
    cacheImu.push_back(imuPt);
}

void SlamNode::Image_CB(const sensor_msgs::msg::Image::SharedPtr msg)
{       
    latestImageTs = stamp2sec(msg->header.stamp);
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg,"mono8");
    cv::Mat srcMat = cv_ptr->image;
    std::unique_lock<std::mutex> lock(mMutexCache);
    cacheImage.push_back(srcMat);
    cacheTimeStamp.push_back(latestImageTs);
}

void SlamNode::timer_Map()
{
    std::vector<MapPoint*> vpAllMPs;
    std::vector<MapEdge*> vpAllMEs;
    std::vector<MapColine*>vpAllCLs;
    {
        std::unique_lock<std::mutex> lock(mMutexPub);
        vpAllMPs = mvpAllMPs;
        vpAllMEs = mvpAllMEs;
        vpAllCLs = mvpAllCLs;
    }

    // pub mappoints
    if(!vpAllMPs.empty())
    {
        sensor_msgs::msg::PointCloud2 _mappoints;
        _mappoints.header.stamp = sec2stamp(latestImageTs); 
        _mappoints.header.frame_id = "world"; 
        _mappoints.height = 1;
        _mappoints.width = vpAllMPs.size();
        _mappoints.is_dense = false;
        _mappoints.is_bigendian = false;
        _mappoints.fields.resize(3);
        _mappoints.fields[0].name = "x";
        _mappoints.fields[0].offset = 0;
        _mappoints.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        _mappoints.fields[0].count = 1;
        _mappoints.fields[1].name = "y";
        _mappoints.fields[1].offset = 4;
        _mappoints.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        _mappoints.fields[1].count = 1;
        _mappoints.fields[2].name = "z";
        _mappoints.fields[2].offset = 8;
        _mappoints.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        _mappoints.fields[2].count = 1;
        _mappoints.point_step = 12;
        _mappoints.row_step = _mappoints.point_step * vpAllMPs.size();
        _mappoints.data.resize(_mappoints.row_step * _mappoints.height);
        size_t valid_idx = 0;
        for(unsigned int i=0; i<vpAllMPs.size(); i++)
        {
            MapPoint* pMP = vpAllMPs[i];
            if(pMP == nullptr || pMP->isBad())
                continue;
            Eigen::Vector3f pos = pMP->GetWorldPos();
            float* data_ptr = reinterpret_cast<float*>(&_mappoints.data[valid_idx * _mappoints.point_step]);
            data_ptr[0] = pos[0];
            data_ptr[1] = pos[1];
            data_ptr[2] = pos[2];
            valid_idx++;
        }
        _mappoints.width = valid_idx;
        _mappoints.row_step = _mappoints.point_step * _mappoints.width;
        _mappoints.data.resize(_mappoints.row_step * _mappoints.height);
        map_point_pub->publish(_mappoints);
    }
    
    if(!vpAllMEs.empty())
    {
        // pub edges
        visualization_msgs::msg::Marker edges;
        edges.header.stamp = sec2stamp(latestImageTs); 
        edges.header.frame_id = "world"; 
        edges.ns = "lines";
        edges.type = visualization_msgs::msg::Marker::LINE_LIST;
        edges.action = visualization_msgs::msg::Marker::ADD;
        // edges.pose.orientation.w = 1.0;
        edges.lifetime = rclcpp::Duration(0, 0);
        edges.id = 0; //key_poses_id++;
        edges.scale.x = 0.01;
        edges.scale.y = 0.01;
        edges.scale.z = 0.01;
        edges.color.r = 0.8;
        edges.color.g = 0.8;
        edges.color.b = 0.8;
        edges.color.a = 0.6;
        for(MapEdge* pME : vpAllMEs)
        {
            if(pME == nullptr || pME->isBad() || !pME->mbValid)
                continue;
            Eigen::Vector3f ps = pME->mpMPs->GetWorldPos();
            Eigen::Vector3f pe = pME->mpMPe->GetWorldPos();
            geometry_msgs::msg::Point p;
            p.x = ps[0];
            p.y = ps[1];
            p.z = ps[2];
            edges.points.push_back(p);
            p.x = pe[0];
            p.y = pe[1];
            p.z = pe[2];
            edges.points.push_back(p);
        }
        map_edge_pub->publish(edges);
    }

    if(!vpAllCLs.empty())
    {
        // pub co-line
        visualization_msgs::msg::Marker colines;
        colines.header.stamp = sec2stamp(latestImageTs); 
        colines.header.frame_id = "world"; 
        colines.ns = "lines";
        colines.type = visualization_msgs::msg::Marker::LINE_LIST;
        colines.action = visualization_msgs::msg::Marker::ADD;
        // colines.pose.orientation.w = 1.0;
        colines.lifetime = rclcpp::Duration(0, 0);
        colines.id = 0; //key_poses_id++;
        colines.scale.x = 0.02;
        colines.scale.y = 0.02;
        colines.scale.z = 0.02;
        colines.color.g = 0.5;
        colines.color.b = 0.5;
        colines.color.r = 1.0;
        colines.color.a = 0.5;
        for(MapColine* pCL : vpAllCLs)
        {
            if(pCL == nullptr || pCL->isBad() || !pCL->mbValid)
                continue;
            Eigen::Vector3f ps = pCL->mpMPs->GetWorldPos();
            Eigen::Vector3f pe = pCL->mpMPe->GetWorldPos();
            geometry_msgs::msg::Point p; 
            p.x = ps[0];
            p.y = ps[1];
            p.z = ps[2];
            colines.points.push_back(p);
            p.x = pe[0];
            p.y = pe[1];
            p.z = pe[2];
            colines.points.push_back(p);
        }
        map_coline_pub->publish(colines);
    }
}

builtin_interfaces::msg::Time SlamNode::sec2stamp(const double &sec)
{
    builtin_interfaces::msg::Time tm;
    tm.sec = static_cast<int32_t>(sec);
    tm.nanosec = static_cast<uint32_t>((sec - tm.sec) * 1e9);
    return tm;
}

double SlamNode::stamp2sec(const builtin_interfaces::msg::Time &in_time)
{
    return static_cast<double>(in_time.sec) + static_cast<double>(in_time.nanosec) * 1e-9;
}
