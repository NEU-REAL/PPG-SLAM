#pragma once
#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "Tracking.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "IMU.h"
#include "KannalaBrandt8.h"
#include "Pinhole.h"
#include "GeometricCamera.h"
 
class Tracking;
class LocalMapping;
class LoopClosing;

class SlamNode : public rclcpp::Node
{
public:

public:
    SlamNode();
    ~SlamNode();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void Load();

    void IMU_CB(const sensor_msgs::msg::Imu::SharedPtr msg);

    void Image_CB(const sensor_msgs::msg::Image::SharedPtr msg);

    void timer();

    void timer_Map();


    builtin_interfaces::msg::Time sec2stamp(const double &sec);
    double stamp2sec(const builtin_interfaces::msg::Time &in_time);


public:
    DBoW3::Vocabulary* mpVocabulary;
    Map* mpMap;
    std::mutex mMutexReset;
    std::list<IMU::Point> cacheImu;
    std::list<cv::Mat> cacheImage;
    std::list<double> cacheTimeStamp;
    double latestImuTs, latestImageTs;
    std::mutex mMutexCache;

public:
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectory_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_point_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr map_edge_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr map_coline_pub;
    rclcpp::TimerBase::SharedPtr m_timer_exe, m_timer_map;

private:
    std::vector<MapPoint*> mvpAllMPs;
    std::vector<MapEdge*> mvpAllMEs;
    std::vector<MapColine*> mvpAllCLs;
    std::mutex mMutexPub;
};