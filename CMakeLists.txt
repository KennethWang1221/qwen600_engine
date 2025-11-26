# CMakeLists.txt for Qwen3-0.6B CUDA Engine
cmake_minimum_required(VERSION 3.18)
project(qwen_cuda_engine LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA (required for both C++ and Python)
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# C++ Executable (main inference engine)
# ============================================================================
option(BUILD_EXECUTABLE "Build C++ executable" ON)

if(BUILD_EXECUTABLE)
    add_executable(qwen600_engine main.cu)

    target_include_directories(qwen600_engine PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(qwen600_engine PRIVATE
        CUDA::cudart
        CUDA::cublas
    )

    set_target_properties(qwen600_engine PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86"
    )
endif()

# ============================================================================
# Python Bindings (optional, only if pybind11 is available)
# ============================================================================
option(BUILD_PYTHON "Build Python bindings" ON)

if(BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development QUIET)
    find_package(pybind11 CONFIG QUIET)
    
    if(Python_FOUND AND pybind11_FOUND)
        message(STATUS "Building Python bindings")
        
        pybind11_add_module(_qwen_core 
            python/bindings.cu
        )
        
        target_include_directories(_qwen_core PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        
        target_link_libraries(_qwen_core PRIVATE
            CUDA::cudart
            CUDA::cublas
        )
        
        set_target_properties(_qwen_core PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "70;75;80;86"
        )
        
        install(TARGETS _qwen_core
            LIBRARY DESTINATION .
        )
    else()
        if(NOT Python_FOUND)
            message(WARNING "Python not found. Skipping Python bindings.")
        endif()
        if(NOT pybind11_FOUND)
            message(WARNING "pybind11 not found. Skipping Python bindings. Install with: pip install pybind11")
        endif()
        message(STATUS "Building C++ executable only")
    endif()
endif()
