//=============================================================================
// EuRoC Dataset Publisher for PPG-SLAM
//=============================================================================
// This file implements a ROS2 node that publishes EuRoC dataset images and IMU data
// in real-time simulation for PPG-SLAM testing.
//=============================================================================

#include <mutex>
#include <memory>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include <std_msgs/msg/header.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace std::chrono_literals;

// Global data containers for EuRoC dataset
vector<string> vImagePath;
vector<double> vImageTimeStamp;
vector<cv::Point3f> vImuAcc;
vector<cv::Point3f> vImuGyr;
vector<double> vImuTimestamp;


/**
 * @brief EuRoC Dataset Publisher Node
 * 
 * Publishes EuRoC dataset images and IMU data with synchronized timing
 * to simulate real-time sensor data for PPG-SLAM testing.
 */
class EuRoCPublisher : public rclcpp::Node
{
public:
    EuRoCPublisher() : Node("euroc_publisher")
    {
        // Create publishers for IMU and image data
        imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("/ppg_slam/imu_raw", 1000);
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/ppg_slam/image_raw", 100);
        
        // Create timer for synchronized data publishing
        timer_ = this->create_wall_timer(1ms, std::bind(&EuRoCPublisher::timerCallback, this));
        
        RCLCPP_INFO(this->get_logger(), "EuRoC Publisher initialized");
    }

private:
    /**
     * @brief Timer callback for publishing sensor data
     * 
     * Publishes IMU and image data based on their timestamps to maintain
     * temporal synchronization with the original dataset timing.
     */
    void timerCallback()
    {
        // Initialize timing on first call
        if (count_imu_ == 0 && count_image_ == 0) {
            initializeTiming();
        }

        // Calculate elapsed time since start
        double elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - start_time_).count();

        // Publish IMU data based on timing
        publishImuData(elapsed_time);
        
        // Publish image data based on timing
        publishImageData(elapsed_time);

        // Check if publishing is complete
        if (count_image_ >= size_img_ - 1) {
            RCLCPP_INFO(this->get_logger(), "Dataset publishing completed");
        }
    }

    /**
     * @brief Initialize timing variables
     */
    void initializeTiming()
    {
        size_imu_ = vImuTimestamp.size();
        size_img_ = vImageTimeStamp.size();
        start_time_ = std::chrono::high_resolution_clock::now();
        
        // Find the earliest timestamp to use as reference
        dataset_start_time_ = (vImuTimestamp[0] < vImageTimeStamp[0]) ? 
                              vImuTimestamp[0] : vImageTimeStamp[0];
        
        RCLCPP_INFO(this->get_logger(), "Publishing %zu IMU samples and %zu images", 
                    size_imu_, size_img_);
    }

    /**
     * @brief Publish IMU data based on elapsed time
     * @param elapsed_time Current elapsed time since start
     */
    void publishImuData(double elapsed_time)
    {
        while (count_imu_ < size_imu_ && 
               (vImuTimestamp[count_imu_] - dataset_start_time_) < elapsed_time) {
            
            sensor_msgs::msg::Imu imu_msg;
            
            // Set acceleration data
            imu_msg.linear_acceleration.x = vImuAcc[count_imu_].x;
            imu_msg.linear_acceleration.y = vImuAcc[count_imu_].y;
            imu_msg.linear_acceleration.z = vImuAcc[count_imu_].z;
            
            // Set angular velocity data
            imu_msg.angular_velocity.x = vImuGyr[count_imu_].x;
            imu_msg.angular_velocity.y = vImuGyr[count_imu_].y;
            imu_msg.angular_velocity.z = vImuGyr[count_imu_].z;
            
            // Set header information
            imu_msg.header.frame_id = "imu";
            imu_msg.header.stamp = sec2stamp(vImuTimestamp[count_imu_]);
            
            imu_pub_->publish(imu_msg);
            count_imu_++;
        }
    }

    /**
     * @brief Publish image data based on elapsed time
     * @param elapsed_time Current elapsed time since start
     */
    void publishImageData(double elapsed_time)
    {
        while (count_image_ < size_img_ && 
               (vImageTimeStamp[count_image_] - dataset_start_time_) < elapsed_time) {
            
            // Create header
            std_msgs::msg::Header header;
            header.frame_id = "camera";
            header.stamp = sec2stamp(vImageTimeStamp[count_image_]);
            
            // Load and publish image
            cv::Mat img = cv::imread(vImagePath[count_image_], cv::IMREAD_GRAYSCALE);
            if (!img.empty()) {
                auto image_msg = cv_bridge::CvImage(header, "mono8", img).toImageMsg();
                image_pub_->publish(*image_msg);
                
                if (count_image_ % 100 == 0) {
                    RCLCPP_INFO(this->get_logger(), "Published image %zu/%zu", 
                                count_image_, size_img_);
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "Failed to load image: %s", 
                            vImagePath[count_image_].c_str());
            }
            
            count_image_++;
        }
    }

    /**
     * @brief Convert seconds to ROS2 timestamp
     * @param sec Time in seconds
     * @return ROS2 timestamp message
     */
    builtin_interfaces::msg::Time sec2stamp(const double &sec)
    {
        builtin_interfaces::msg::Time timestamp;
        timestamp.sec = static_cast<int32_t>(sec);
        timestamp.nanosec = static_cast<uint32_t>((sec - timestamp.sec) * 1e9);
        return timestamp;
    }

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Counters and sizes
    size_t count_imu_ = 0;
    size_t count_image_ = 0;
    size_t size_imu_ = 0;
    size_t size_img_ = 0;

    // Timing variables
    std::chrono::high_resolution_clock::time_point start_time_;
    double dataset_start_time_ = 0.0;
};

/**
 * @brief Load EuRoC dataset images
 * @param data_folder Path to EuRoC dataset folder
 * @return true if successful, false otherwise
 */
bool loadImages(const string& data_folder)
{
    vImagePath.reserve(6000);
    vImageTimeStamp.reserve(6000);
    
    ifstream image_file(data_folder + "/mav0/cam0/data.csv");
    if (!image_file.is_open()) {
        cerr << "Error: Cannot open image CSV file" << endl;
        return false;
    }

    string line;
    while (getline(image_file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        size_t comma_pos = line.find(',');
        if (comma_pos != string::npos) {
            string timestamp_str = line.substr(0, comma_pos);
            vImagePath.push_back(data_folder + "/mav0/cam0/data/" + timestamp_str + ".png");
            vImageTimeStamp.push_back(stod(timestamp_str) * 1e-9);
        }
    }
    image_file.close();
    
    cout << "Loaded " << vImagePath.size() << " image entries" << endl;
    return true;
}

/**
 * @brief Load EuRoC dataset IMU data
 * @param data_folder Path to EuRoC dataset folder
 * @return true if successful, false otherwise
 */
bool loadImuData(const string& data_folder)
{
    vImuAcc.reserve(60000);
    vImuGyr.reserve(60000);
    vImuTimestamp.reserve(60000);
    
    ifstream imu_file(data_folder + "/mav0/imu0/data.csv");
    if (!imu_file.is_open()) {
        cerr << "Error: Cannot open IMU CSV file" << endl;
        return false;
    }

    string line;
    while (getline(imu_file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Parse CSV line: timestamp,wx,wy,wz,ax,ay,az
        vector<double> data;
        string token;
        size_t pos = 0;
        
        while ((pos = line.find(',')) != string::npos) {
            token = line.substr(0, pos);
            data.push_back(stod(token));
            line.erase(0, pos + 1);
        }
        data.push_back(stod(line)); // Last token
        
        if (data.size() == 7) {
            vImuTimestamp.push_back(data[0] * 1e-9);  // Convert nanoseconds to seconds
            vImuGyr.push_back(cv::Point3f(data[1], data[2], data[3]));  // Angular velocity
            vImuAcc.push_back(cv::Point3f(data[4], data[5], data[6]));  // Linear acceleration
        }
    }
    imu_file.close();
    
    cout << "Loaded " << vImuTimestamp.size() << " IMU entries" << endl;
    return true;
}

/**
 * @brief Main function
 */
int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <path_to_euroc_dataset>" << endl;
        return -1;
    }

    string data_folder = string(argv[1]);
    cout << "Loading EuRoC dataset from: " << data_folder << endl;

    // Load dataset
    if (!loadImages(data_folder) || !loadImuData(data_folder)) {
        cerr << "Failed to load dataset" << endl;
        return -1;
    }

    cout << "Dataset loaded successfully" << endl;
    cout << "Starting ROS2 publisher..." << endl;

    // Initialize ROS2 and start publishing
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EuRoCPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}