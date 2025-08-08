find_package(Pangolin REQUIRED)
include_directories(
    ${Pangolin_INCLUDE_DIRS}
) 

add_library(System
    src/LocalMapping.cpp
    src/LoopClosing.cpp
    src/Tracking.cpp
    src/Viewer.cpp
    src/System.cpp
)

target_link_libraries(System
Sensors Feature Matching Map Optimizing
${Pangolin_LIBRARIES}
)
