# Qwen Inference Engine

A from-scratch GPU-accelerated LLM inference engine for learning CUDA, C++, and transformer internals.

## Project Status

**Phase 0: ✅ Complete** - Initialization verified (16/16 tests passing)  
**Phase 1: 🔨 Ready** - CUDA kernels implementation

## Quick Start

### Build
```bash
mkdir -p build && cd build
cmake ..
make
```

### Verify Phase 0
```bash
cd build
./test_phase0_initialization /path/to/model.safetensors
```

Should see: **"✓ ALL TESTS PASSED! READY FOR PHASE 1!"**

## Project Structure

```
qwen600_engine/
├── main.cu                 # Main application
├── config.h                # Model configuration
├── qwen_model.cuh          # Core model implementation
├── static_loader.h         # Weight loading
├── CMakeLists.txt          # Build system
├── docs/                   # 📚 All documentation
│   ├── LEARNING_GUIDE.md        # Complete 8-week roadmap (ALL phases)
│   └── PHASE0_VERIFICATION.md   # Phase 0 testing & status
├── tests/                  # 🧪 All tests (one per phase)
│   └── test_phase0_initialization.cu
└── build/                  # Build output
```

## Documentation

- **[📖 LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md)** - Complete 8-week roadmap for ALL phases
- **[✅ PHASE0_VERIFICATION.md](docs/PHASE0_VERIFICATION.md)** - Phase 0 testing & completion status
- **[🔨 BUILD_GUIDE.md](docs/BUILD_GUIDE.md)** - Proper build workflow (avoid common mistakes)

## What's Implemented

### ✅ Phase 0: Setup & Weight Loading
- Project infrastructure (CMake, build system)
- **`build_transformer()`** - Complete initialization function
- SafeTensors weight loading (~1.5 GB)
- GPU memory management (~2.4 GB total)
- cuBLAS initialization
- Zero memory leaks

### 🔨 Phase 1: CUDA Kernels (Next)
- RMSNorm
- RoPE (Rotary Position Embedding)
- Attention mechanism
- SwiGLU activation

See [Learning Guide](docs/LEARNING_GUIDE.md) for full roadmap.

## Requirements

- CUDA Toolkit 11.0+
- CMake 3.15+
- C++17 compiler
- GPU with 3GB+ VRAM

## Model

Download Qwen 0.6B model in SafeTensors format:
```bash
huggingface-cli download Qwen/Qwen2.5-0.6B --include "*.safetensors" --local-dir ./model
```

## Debugging with CUDA-GDB

### Setup
```bash
# Build with debug symbols
cd build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)

# Start debugger
cuda-gdb-python3.12-tui qwen600_engine
```

### Essential Commands
```bash
# Navigation
n, next          # Next line
s, step          # Step into function
c, continue      # Continue to breakpoint
bt               # Show call stack
up               # Go up one stack frame
down             # Go back down
frame N          # Jump to frame N

# Display Control
layout src       # Show source view
refresh          # Refresh screen
Ctrl-X-o        # Switch between windows
Ctrl-X-a        # Exit/enter TUI mode
focus cmd        # Focus command window
focus src        # Focus source window

# TUI Navigation
# When focus is on source window:
Up/Down arrows   # Scroll source code
PgUp/PgDn       # Page up/down in source

# When focus is on command window:
Up/Down arrows   # Command history
Left/Right      # Edit command line (use Ctrl-B/Ctrl-F if arrows don't work)

# Debugging Commands (what you CAN run in CUDA-GDB)
break main                 # Set breakpoint at function
break file.cu:123         # Set breakpoint at line
info breakpoints          # List all breakpoints
delete 1                  # Delete breakpoint #1

# Running/Control
run <args>                # Start program with arguments
continue, c               # Continue execution
next, n                  # Execute next line
step, s                  # Step into function
finish                   # Run until function returns

# Examining Variables (works with compiled variables only)
print var                # Print variable value
print *array@10         # Print 10 array elements
print sizeof(var)       # Print size
info locals             # Show local variables
info args               # Show function arguments

# Calling Functions (must exist in your compiled code)
call my_function()      # Call a function from your code
call printf("test\n")   # Call standard functions

# Stack Navigation
bt, backtrace           # Show call stack
frame 0                 # Jump to frame 0
up                      # Go up one frame
down                    # Go down one frame
where                   # Show current location

# What you CANNOT do in CUDA-GDB:
# ❌ Run arbitrary C code: printf("\n" COLOR_CYAN "...");
# ❌ Use preprocessor macros: COLOR_CYAN, #define
# ❌ Declare new variables: int x = 5;
# ❌ Write loops: for(int i=0; i<10; i++)

# Run program with parameters
run ../../Qwen3-0.6B -r 1 -t 0.65 -p 0.9 -k 20
```

### Fix Common Issues
```bash
# If display repeats lines:
set pagination on
set height 0
set logging off

# If variables show "optimized out":
set print static-members off
set print object on
```

## Next Steps

1. ✅ Verify Phase 0 passes all tests
2. 📖 Read [Learning Guide](docs/LEARNING_GUIDE.md) Week 1-2
3. 🔧 Implement RMSNorm kernel
4. 🧪 Test and verify
5. 🚀 Continue with remaining kernels

## License

MIT License---

**Current Status**: Phase 0 complete, ready for kernel implementation! 🚀


