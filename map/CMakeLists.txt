add_library(Map
    src/Map.cpp
    src/Frame.cpp
)

target_link_libraries(Map Sensors)