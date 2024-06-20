# Introduction

This is a C++ library written for developing experiments for energy-based Hopfield networks. I wrote this 
library primarily because python is slow to simulate experiments for modern versions of Hopfield Networks that have
very high memory capacity (~ a million memories). The library is written using Pytorch C++ for GPU and CPU acceleration. 

# Installation

## Prerequisites

The following libraries are required prior to build EDEN.

- `cxxopts` - for commant line argument parsing
- `progressbar` - for displaying progressbar
- `libtorch` - for hardware acceleration
- `matplotplusplus` - for plotting diagnostics

The library can be configured and build by running the following commands from the build directory.

````
cmake -DCMAKE_PREFIX_PATH=<libtorch_path> ..

cmake --build . --parallel 8
````

The experiment binaries will be written to the `bin` folder in the build directory.
