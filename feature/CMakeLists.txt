add_library(Feature SHARED
    src/KeyFrame.cpp
    src/MapPoint.cpp
    src/PPGExtractor.cpp
    src/PPGGraph.cpp
) 

target_link_libraries(Feature
    Sensors
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
