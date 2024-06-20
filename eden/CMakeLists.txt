find_package(Torch REQUIRED)

add_library(eden HopfieldNetwork.cpp
        EDEN.cpp
        EDEN.h)
target_link_libraries(eden "${TORCH_LIBRARIES}")
