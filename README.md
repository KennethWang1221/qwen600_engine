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
- CUDA-GDB for debugging

## Building from Source

### Release Build (for performance)
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Debug Build (for development)
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

### Clean Rebuild
If you need to rebuild from scratch:
```bash
cd build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug ..  # or Release
make VERBOSE=1  # Shows detailed compilation commands
```

## Debugging with CUDA-GDB

1. Build in Debug mode first:
```bash
cd build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

2. Start CUDA-GDB with TUI (Text User Interface):
```bash
cuda-gdb-python3.12-tui qwen600_engine
```

3. Common debugging commands:
```bash
# Breakpoints
(cuda-gdb) break main              # Set breakpoint at main
(cuda-gdb) info breakpoints        # List all breakpoints
(cuda-gdb) delete N                # Delete breakpoint N
(cuda-gdb) disable N               # Disable breakpoint N
(cuda-gdb) enable N                # Enable breakpoint N

# Navigation
(cuda-gdb) run models/ -t 0.6      # Run with arguments
(cuda-gdb) n                       # Next line
(cuda-gdb) s                       # Step into function
(cuda-gdb) up                      # Move up one stack frame
(cuda-gdb) down                    # Move down one stack frame
(cuda-gdb) frame N                 # Switch to frame N
(cuda-gdb) list                    # Show source code around current line
(cuda-gdb) where                   # Show current location (like bt)

# Inspection
(cuda-gdb) p variable_name         # Print variable value
(cuda-gdb) bt                      # Show backtrace
(cuda-gdb) info cuda threads       # Show CUDA threads
(cuda-gdb) cuda thread             # Switch between CUDA threads
(cuda-gdb) layout src              # Show source code view
(cuda-gdb) quit                    # Exit debugger
```

4. Keyboard shortcuts in TUI mode:
- Ctrl-X + 1: Show single window
- Ctrl-X + 2: Show two windows
- Ctrl-L: Refresh screen
- Ctrl-P/Ctrl-N: Previous/Next command
- Ctrl-X + A: Toggle TUI mode

## License

MIT License

## Acknowledgments

This project is inspired by:
- llama.cpp
- qwen3.c
- LLMs-from-scratch