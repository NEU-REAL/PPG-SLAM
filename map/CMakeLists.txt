add_library(Map SHARED
    src/Map.cpp
    src/Frame.cpp
)

target_link_libraries(Map Sensors)