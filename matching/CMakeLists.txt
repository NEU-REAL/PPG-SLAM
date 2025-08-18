add_library(Matching SHARED
    src/Matcher.cpp
    src/MLPnPsolver.cpp
    src/Sim3Solver.cpp
)

target_link_libraries(Matching
    Sensors
    ${OpenCV_LIBS}
)