cmake_minimum_required(VERSION 3.8)
project(ppg_slam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3 -g -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g -std=c++17")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(cv_bridge REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Torch REQUIRED PATHS /home/z/module/libtorch)
find_package(DBoW3 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${DBoW3_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o
    ${PROJECT_SOURCE_DIR}/sensors/include
    ${PROJECT_SOURCE_DIR}/feature/include
    ${PROJECT_SOURCE_DIR}/map/include
    ${PROJECT_SOURCE_DIR}/optimizing/include
    ${PROJECT_SOURCE_DIR}/matching/include
    ${PROJECT_SOURCE_DIR}/system/include
)

add_subdirectory(${PROJECT_SOURCE_DIR}/sensors)
add_subdirectory(${PROJECT_SOURCE_DIR}/feature)
add_subdirectory(${PROJECT_SOURCE_DIR}/matching)
add_subdirectory(${PROJECT_SOURCE_DIR}/map)
add_subdirectory(${PROJECT_SOURCE_DIR}/optimizing)
add_subdirectory(${PROJECT_SOURCE_DIR}/system)
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/g2o)

#***************ros exe******************# 
add_executable(${PROJECT_NAME}_node main.cpp)

target_link_libraries(${PROJECT_NAME}_node
Sensors Feature Matching Map Optimizing System
${DBoW3_LIBS}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${TORCH_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

add_executable(pub_euroc_node test/pub_euroc.cpp)
target_link_libraries(pub_euroc_node ${OpenCV_LIBS} )
ament_target_dependencies(pub_euroc_node
    tf2 tf2_ros nav_msgs sensor_msgs geometry_msgs visualization_msgs cv_bridge
)
add_executable(pub_tum_node test/pub_tum.cpp)
target_link_libraries(pub_tum_node ${OpenCV_LIBS} )
ament_target_dependencies(pub_tum_node
    tf2 tf2_ros nav_msgs sensor_msgs geometry_msgs visualization_msgs cv_bridge
)
add_executable(pub_uma_node test/pub_uma.cpp)
target_link_libraries(pub_uma_node ${OpenCV_LIBS} )
ament_target_dependencies(pub_uma_node
    tf2 tf2_ros nav_msgs sensor_msgs geometry_msgs visualization_msgs cv_bridge
)

install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})
install(TARGETS pub_euroc_node DESTINATION lib/${PROJECT_NAME})
install(TARGETS pub_tum_node   DESTINATION lib/${PROJECT_NAME})
install(TARGETS pub_uma_node   DESTINATION lib/${PROJECT_NAME})

install(TARGETS Sensors DESTINATION lib)
install(TARGETS Feature DESTINATION lib)
install(TARGETS Matching DESTINATION lib)
install(TARGETS Map DESTINATION lib)
install(TARGETS Optimizing DESTINATION lib)
install(TARGETS System DESTINATION lib)
install(TARGETS g2o DESTINATION lib)


install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})
install(DIRECTORY net DESTINATION share/${PROJECT_NAME})
install(DIRECTORY Vocabulary DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
