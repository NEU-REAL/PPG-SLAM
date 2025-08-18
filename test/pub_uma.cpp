#include <mutex>
#include <memory>
#include <iostream>
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include <std_msgs/msg/header.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <fstream>

using namespace std;
using namespace std::chrono_literals;

vector<string> vImagePath;
vector<double> vImageTimeStamp;
vector<cv::Point3f> vImuAcc;
vector<cv::Point3f> vImuGyr;
vector<double> vImuTimestamp;


class Talker : public rclcpp::Node
{
public:
    Talker() : Node("uma_publisher")
    {
        // 创建发布者，发布 std_msgs::msg::String 类型的消息到 'topic' 话题，队列大小为 10
        imu_pub = this->create_publisher<sensor_msgs::msg::Imu>("/ppg_slam/imu_raw", 1000);
        image_pub = this->create_publisher<sensor_msgs::msg::Image>("/ppg_slam/image_raw", 100);
        // 创建定时器，每 500 毫秒触发一次，调用 timer_callback 函数
        timer_imu = this->create_wall_timer(1ms, std::bind(&Talker::timer_callback, this));
    }
private:
    void timer_callback()
    {
        if(count_imu == 0 && count_image == 0)
        {
            size_imu = vImuTimestamp.size();
            size_img = vImageTimeStamp.size();
            Tini = std::chrono::high_resolution_clock::now();
            startTime = vImuTimestamp[0] < vImageTimeStamp[0] ? vImuTimestamp[0] : vImageTimeStamp[0];

        }
        double dt = std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::high_resolution_clock::now() - Tini).count();
        while(vImuTimestamp[count_imu] - startTime < dt && count_imu <size_imu)
        {
            sensor_msgs::msg::Imu imu_msg;
            imu_msg.linear_acceleration.x = vImuAcc[count_imu].x;
            imu_msg.linear_acceleration.y = vImuAcc[count_imu].y;
            imu_msg.linear_acceleration.z = vImuAcc[count_imu].z;
            imu_msg.angular_velocity.x = vImuGyr[count_imu].x;
            imu_msg.angular_velocity.y = vImuGyr[count_imu].y;
            imu_msg.angular_velocity.z = vImuGyr[count_imu].z;
            imu_msg.header.frame_id = "imu";
            imu_msg.header.stamp = sec2stamp(vImuTimestamp[count_imu]);
            count_imu++;
            imu_pub->publish(imu_msg);
        }
        while(vImageTimeStamp[count_image] - startTime < dt && count_image <size_img)
        {
            std_msgs::msg::Header hd;
            hd.frame_id = "image";
            hd.stamp = sec2stamp(vImageTimeStamp[count_image]);
            cv::Mat img = cv::imread(vImagePath[count_image], 0);
            auto image_msg = cv_bridge::CvImage(hd,"mono8",img).toImageMsg();
            image_pub->publish(*image_msg);
            count_image++;
            std::cout<<" image: "<<count_image<<std::endl;
        }
        if(count_image == size_img -1)
            std::cerr<<" done."<<std::endl;
    }
    builtin_interfaces::msg::Time sec2stamp(const double &sec)
    {
        builtin_interfaces::msg::Time tm;
        tm.sec = static_cast<int32_t>(sec);
        tm.nanosec = static_cast<uint32_t>((sec - tm.sec) * 1e9);
        return tm;
    }

    double stamp2sec(const builtin_interfaces::msg::Time &in_time)
    {
        return static_cast<double>(in_time.sec) + static_cast<double>(in_time.nanosec) * 1e-9;
    }

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
    rclcpp::TimerBase::SharedPtr timer_image, timer_imu;
    size_t count_imu = 0;
    size_t count_image = 0;
    size_t size_imu, size_img;
    std::chrono::_V2::system_clock::time_point Tini;
    double startTime = 0;

};

int main(int argc, char **argv)
{
    string dataFolder = string(argv[1]);
    // load images
    vImagePath.reserve(6000);
    vImageTimeStamp.reserve(6000);
    ifstream fimage;
    fimage.open(dataFolder+ "/cam0/data.csv");
    while(!fimage.eof())
    {
        string s;
        getline(fimage,s);
        if(!s.empty())
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
        getline(fImu,s);
        if (s[0] == '#')
            continue;
        if(!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);
            vImuTimestamp.push_back(data[0] * 1e-9);
            vImuAcc.push_back(cv::Point3f(data[4],data[5],data[6]));
            vImuGyr.push_back(cv::Point3f(data[1],data[2],data[3]));
        }
    }
    std::cout<< " data loaded : "<< dataFolder<<std::endl;

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Talker>());
    rclcpp::shutdown();
    return 0;
}