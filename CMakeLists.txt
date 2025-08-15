cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM3)

SET(CMAKE_BUILD_TYPE Release) 
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE}) 

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O1 -g -std=c++17 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O1 -g -std=c++17 -fPIC")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(g2o REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Torch REQUIRED PATHS /home/z/module/libtorch)
find_package(DBoW3 REQUIRED)

# Manually set the paths for g2o
set(g2o_INCLUDE_DIRS "/usr/local/include")
set(g2o_LIBRARIES 
    g2o_core g2o_types_slam3d g2o_solver_dense g2o_types_sba g2o_types_sim3 g2o_solver_eigen fmt
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${DBoW3_INCLUDE_DIRS}
    ${g2o_INCLUDE_DIRS}
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

link_directories(${PROJECT_SOURCE_DIR}/lib)

#Monocular inertial examples
add_executable(mono_inertial_euroc try_euroc.cpp)
target_link_libraries(mono_inertial_euroc 
Sensors Feature Matching Map Optimizing System
${g2o_LIBRARIES}
${DBoW3_LIBS}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${TORCH_LIBRARIES}
)

add_executable(mono_inertial_tum_vi try_tumvi.cpp)
target_link_libraries(mono_inertial_tum_vi 
Sensors Feature Matching Map Optimizing System
${g2o_LIBRARIES}
${DBoW3_LIBS}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${TORCH_LIBRARIES}
)

add_executable(mono_inertial_uma try_uma.cpp)
target_link_libraries(mono_inertial_uma  
Sensors Feature Matching Map Optimizing System
${g2o_LIBRARIES}
${DBoW3_LIBS}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${TORCH_LIBRARIES}
)

add_executable(train_vocabulary trainBow.cpp)
target_link_libraries(train_vocabulary  
Sensors Feature Matching Map Optimizing System
${g2o_LIBRARIES}
${DBoW3_LIBS}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${TORCH_LIBRARIES}
)