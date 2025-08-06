include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_library(Sensors
    include/GeometricCamera.h
    src/KannalaBrandt8.cpp
    src/Pinhole.cpp
    src/TwoViewReconstruction.cpp
    src/IMU.cpp 
)