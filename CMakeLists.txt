# Minimum CMake version required for CUDA support
cmake_minimum_required(VERSION 3.18)

# Define project name and languages
project(qwen600_engine CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set build type and debug flags
set(CMAKE_BUILD_TYPE Debug)

# Configure debug and release flags separately
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug mode: disable optimizations, enable debug info
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    # Keep debug symbols and disable optimizations that interfere with debugging
    add_compile_definitions(DEBUG_MODE)
else()
    # Release mode: enable optimizations
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Set CUDA architecture based on your GPU
# 75=Turing (20xx)
# 86=Ampere (30xx)
# 89=Ada Lovelace (40xx)
# 90=Hopper
set(CMAKE_CUDA_ARCHITECTURES 75 86)  # Supporting common architectures

# Enable position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Print build information
message(STATUS "========================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "C++ Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add executable targets
add_executable(qwen600_engine
    main.cu
)

# Link libraries
target_link_libraries(qwen600_engine
    PRIVATE
    CUDA::cublas
    CUDA::cudart
)

# Add CUDA specific compile options
target_compile_options(qwen600_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Disable warnings for both C++ and CUDA
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qwen600_engine PRIVATE -w)
endif()
if (CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    target_compile_options(qwen600_engine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>)
endif()