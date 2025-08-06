include_directories(
    ../sensors/include
    ../feature/include
    ../map/include
    include
    ${OpenCV_INCLUDE_DIRS}
) 

add_library(Optimizing
    include/G2oVertex.h
    src/G2oEdge.cpp
    src/G2oVertex.cpp
    src/Optimizer.cpp
)

target_link_libraries(Optimizing
Sensors Feature Matching Map
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)
