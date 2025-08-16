cmake_minimum_required(VERSION 2.8)
project(PPG_SLAM)

SET(CMAKE_BUILD_TYPE Release) 
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE}) 

# Add Eigen alignment flags to prevent AVX alignment issues
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3 -g -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g -std=c++17")
# Disable aggressive vectorization in Release mode to prevent alignment issues
# set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG -mno-avx -mno-avx2")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -mno-avx -mno-avx2")

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Torch REQUIRED PATHS /home/z/module/libtorch)
find_package(DBoW3 REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${DBoW3_INCLUDE_DIRS}
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
    ${DBoW3_LIBS}
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${G2O_LIBRARIES}
    Sensors Feature Matching Map Optimizing System
)

add_executable(mono_inertial_tum_vi try_tumvi.cpp)
target_link_libraries(mono_inertial_tum_vi 
    ${DBoW3_LIBS}
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${G2O_LIBRARIES}
    Sensors Feature Matching Map Optimizing System
)

add_executable(mono_inertial_uma try_uma.cpp)
target_link_libraries(mono_inertial_uma  
    ${DBoW3_LIBS}
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${G2O_LIBRARIES}
    Sensors Feature Matching Map Optimizing System
)

add_executable(train_vocabulary trainBow.cpp)
target_link_libraries(train_vocabulary  
    ${DBoW3_LIBS}
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${G2O_LIBRARIES}
    Sensors Feature Matching Map Optimizing System
)