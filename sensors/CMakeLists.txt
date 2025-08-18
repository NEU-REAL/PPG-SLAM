add_library(Sensors SHARED
    src/GeometricCamera.cpp
    src/KannalaBrandt8.cpp
    src/Pinhole.cpp
    src/TwoViewReconstruction.cpp
    src/IMU.cpp 
)

target_link_libraries(Sensors
    ${OpenCV_LIBS}
)