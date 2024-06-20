cmake_minimum_required(VERSION 3.22)
project(EDEN)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(SYSTEM "${PROJECT_BINARY_DIR}"
                    "${PROJECT_SOURCE_DIR}/progressbar"
                    "${PROJECT_SOURCE_DIR}/argparse"
                    "${PROJECT_SOURCE_DIR}/eden"
                    "${PROJECT_SOURCE_DIR}/cxxopts"
)

# add gsemm subdirectory to the project
add_subdirectory(eden)
add_subdirectory(matplotplusplus)
set(CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")

## Test capacity
add_executable(capacity experiments/capacity.cpp)
target_link_libraries(capacity PUBLIC eden matplot "${TORCH_LIBRARIES}")
set_property(TARGET capacity PROPERTY CXX_STANDARD 17)

## Test capacity
add_executable(capacity_threaded experiments/capacity_threaded.cpp)
target_link_libraries(capacity_threaded PUBLIC eden matplot "${TORCH_LIBRARIES}")
set_property(TARGET capacity_threaded PROPERTY CXX_STANDARD 17)
