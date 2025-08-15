add_library(System SHARED
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
