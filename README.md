# QWEN600 Engine

A CUDA-based inference engine for QWEN3-0.6B model implementation, focusing on educational purposes and CUDA programming practice.

## Project Overview

This project aims to create a lightweight, efficient inference engine for the QWEN3-0.6B model with the following features:
- Single batch inference engine
- Static-constanted for compile-time optimization
- Pure CUDA C/C++ implementation
- Minimal library dependencies (cuBLAS, CUB)
- Efficient memory pipeline
- Zero-cost pointer-based weight management on GPU

## Development Status

🚧 Currently under development 🚧

## Requirements

- CUDA Toolkit
- CMake (>= 3.18)
- C++17 compatible compiler
- cuBLAS library

## Building from Source

```bash
mkdir build
cd build
cmake ..
make
```

## License

MIT License

## Acknowledgments

This project is inspired by:
- llama.cpp
- qwen3.c
- LLMs-from-scratch
