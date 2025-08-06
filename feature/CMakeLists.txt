add_library(Feature
    src/KeyFrame.cpp
    src/MapPoint.cpp
    src/PPGExtractor.cpp
    src/PPGGraph.cpp
) 

target_link_libraries(Sensors)
