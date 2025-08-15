add_library(Optimizing SHARED
    src/G2oEdge.cpp
    src/G2oVertex.cpp
    src/OptimizerLBA.cpp
    src/OptimizerGBA.cpp
    src/OptimizerPose.cpp
    src/OptimizerInertial.cpp
    src/OptimizerGraph.cpp
)

target_link_libraries(Optimizing
Sensors Feature Map
${g2o_LIBRARIES}
)
