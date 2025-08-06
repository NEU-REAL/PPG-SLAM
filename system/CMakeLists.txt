add_library(System
    src/LocalMapping.cpp
    src/LoopClosing.cpp
    src/Tracking.cpp
    src/System.cpp
)

target_link_libraries(System
Sensors Feature Matching Map Optimizing
${Pangolin_LIBRARIES}
)

ament_target_dependencies(System
    tf2 tf2_ros nav_msgs sensor_msgs geometry_msgs visualization_msgs cv_bridge
)
