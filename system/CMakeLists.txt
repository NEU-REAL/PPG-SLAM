add_library(System SHARED
    src/LocalMapping.cpp
    src/LoopClosing.cpp
    src/Tracking.cpp
    src/System.cpp
)

target_link_libraries(System
    Sensors Feature Matching Map Optimizing
    ${G2O_LIBRARIES}
    ${DBoW3_LIBS}
)

ament_target_dependencies(System
    rclcpp
    std_msgs
    tf2
    tf2_ros
    nav_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    cv_bridge
)
