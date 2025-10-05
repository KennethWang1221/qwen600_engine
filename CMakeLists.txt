# Minimum CMake version required for CUDA support
cmake_minimum_required(VERSION 3.18)

# Define project name and languages
project(qwen600_engine CUDA CXX)

# Set CUDA architecture based on your GPU
# 75 for RTX 20 series, GTX 16 series (Turing)
# 86 for RTX 30 series (Ampere)
# 89 for RTX 40 series (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Set CUDA compiler path
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Enable highest optimization level for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable separate compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add executable target (commented until we create main.cu)
# add_executable(qwen600_engine
#     main.cu
# )

# Link libraries (commented until we add the executable)
# target_link_libraries(qwen600_engine
#     CUDA::cublas
#     CUDA::cuda_driver
#     CUDA::cudart
# )
