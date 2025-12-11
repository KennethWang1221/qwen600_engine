# Qwen3-0.6B Inference Engine

CUDA inference engine for Qwen3-0.6B with advanced sampling techniques and optimized memory management.

## ✨ Key Features

- **Unified Memory Management**: 20-30% faster initialization with single GPU allocation
- **Advanced Sampling**: Repetition penalty, frequency penalty, min-p sampling
- **Production Error Handling**: Comprehensive CUDA error checking with detailed messages
- **Performance Tools**: Automated benchmarking and profiling support
- **High Quality**: BF16 precision with optimized CUDA kernels

## Quick Start

```bash
# 1. Get the model
uv pip install -r requirements.txt
git clone https://huggingface.co/Qwen/Qwen3-0.6B
python3 export.py

# 2. Build
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 3. Run
# a) Interactive chat mode (with reasoning)
./qwen600_engine ../Qwen3-0.6B -r 1 -t 0.65 -p 0.9 -k 20

# b) Single query mode (basic)
./qwen600_engine ../Qwen3-0.6B -i "What is machine learning?"

# c) Single query mode (advanced sampling)
./qwen600_engine ../Qwen3-0.6B -i "Tell me a story" -t 0.7 -R 1.15 -m 0.05
```

### Python API (Optional)

```bash
# 1. Install the package (builds C++ extension automatically)
uv pip install -e .

# 2. Run the example
python example_usage.py

# Note: If you modify C++/CUDA files (*.cu, *.cuh, *.cpp), you must rebuild:
#   pip install -e . --no-deps
# Python files (*.py) don't require rebuilding - changes take effect immediately!
```

See [docs/PYTHON_API.md](docs/PYTHON_API.md) for complete API documentation and [docs/PYTHON_INSTALLATION.md](docs/PYTHON_INSTALLATION.md) for detailed installation guide.

## Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compiler
- GPU with 3GB+ VRAM (RTX 3050 or better)
- cuBLAS library
- Python 3.7+ (for Python API, optional)
- pybind11 (for Python API, optional)

## Building

### C++ Executable

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Python Package

```bash
# Install in development mode (recommended)
uv pip install -e .

# Or regular installation
uv pip install .
```

The installation will automatically:
- Install required dependencies (pybind11, numpy)
- Build the C++ extension using CMake
- Set up the Python package

See [docs/PYTHON_INSTALLATION.md](docs/PYTHON_INSTALLATION.md) for detailed Python package installation instructions.

## Command-Line Options

### Basic Options
```bash
-i <string>  Input prompt
-y <string>  System prompt (optional)
-r <int>     Reasoning mode: 0=normal, 1=thinking (default: 0)
-s <int>     Random seed (default: time-based)
```

### Sampling Options
```bash
-t <float>   Temperature (default: 0.6, 0=greedy)
-k <int>     Top-k sampling (default: 20)
-p <float>   Top-p/nucleus sampling (default: 0.95)
-m <float>   Min-p sampling threshold (default: 0.05)
```

### Advanced Sampling
```bash
-R <float>   Repetition penalty (default: 1.1, >1.0 discourages repetition)
-F <float>   Frequency penalty (default: 0.0)
-P <float>   Presence penalty (default: 0.0)
-w <int>     Penalty window size (default: 64)
```

## Usage Examples

### Basic Inference
```bash
./qwen600_engine ../Qwen3-0.6B -i "Explain quantum computing"
```

### Creative Writing (higher temperature, less repetition)
```bash
./qwen600_engine ../Qwen3-0.6B \
    -i "Write a sci-fi story" \
    -t 0.8 \
    -R 1.2 \
    -w 128
```

### Focused Reasoning (lower temperature, high quality)
```bash
./qwen600_engine ../Qwen3-0.6B \
    -i "Solve this math problem" \
    -r 1 \
    -t 0.3 \
    -m 0.1
```

### Interactive Chat
```bash
./qwen600_engine ../Qwen3-0.6B -y "You are a helpful AI assistant"
# Then type your messages interactively
```

## Performance

### Benchmarking
```bash
# Run comprehensive benchmark suite
./benchmark.sh Qwen3-0.6B ./build/qwen600_engine
```

### Profiling with Nsight Systems
```bash
nsys profile -o qwen_profile ./qwen600_engine ../Qwen3-0.6B -i "Test"
nsys-ui qwen_profile.nsys-rep
```

## Project Structure

```
qwen600_engine/
├── main.cu                 # Main application & chat loop
├── qwen_model.cuh          # Transformer model with optimized kernels
├── sampler.h               # Advanced sampling algorithms
├── tokenizer.h             # BPE tokenizer
├── static_loader.h         # SafeTensors weight loader
├── config.h                # Model configuration
├── cuda_utils.cuh          # CUDA error handling & utilities
├── memory_manager.cuh      # Unified memory management
├── benchmark.sh            # Automated benchmarking
├── export.py               # HuggingFace tokenizer converter
├── python/                 # Python package
│   ├── qwen_engine/        # Python module
│   │   ├── __init__.py
│   │   └── qwen_inference.py
│   ├── bindings.cpp        # Pybind11 C++ bindings
│   └── examples/           # Python usage examples
│       ├── simple_usage.py
│       └── onnx_style.py
├── setup.py                # Python package setup
├── CMakeLists.txt          # CMake build configuration
├── pyproject.toml          # Python package metadata
└── docs/                   # Documentation
    ├── README.md           # User guide
    ├── PYTHON_API.md       # Python API reference
    ├── PYTHON_INSTALLATION.md  # Python install guide
    ├── OPTIMIZATIONS.md    # Technical details
    ├── CHANGELOG.md        # Version history
    └── TRANSFORMATION_SUMMARY.md  # Project transformation summary
```

## Architecture Features

### Model Architecture
- ✅ Qwen3-0.6B (1.5GB weights, BF16 precision)
- ✅ 28 transformer layers, 16 attention heads
- ✅ Grouped Query Attention (GQA) with 8 KV heads
- ✅ RoPE positional encoding (θ=1M)
- ✅ SwiGLU activation function
- ✅ QK-Norm for stable training
- ✅ 8K context window with KV caching

### CUDA Optimizations
- ✅ **Unified Memory Pool**: Single allocation for all runtime buffers
- ✅ **Vectorized Kernels**: BF16x2 operations for RMSNorm
- ✅ **Tensor Cores**: cuBLAS GEMM with Tensor Core support
- ✅ **Optimized Attention**: Fused QK^T, Softmax, and V aggregation
- ✅ **Error Checking**: Comprehensive CUDA error handling

### Sampling Techniques
- ✅ Temperature scaling
- ✅ Top-k sampling
- ✅ Top-p (nucleus) sampling
- ✅ Min-p sampling (quality-focused)
- ✅ Repetition penalty (reduce repetition)
- ✅ Frequency penalty (penalize common tokens)
- ✅ Presence penalty (discourage reuse)

## Memory Usage

```
Total GPU Memory: ~2.4GB
├── Model Weights:     1.5GB (BF16)
├── Runtime State:     0.7GB (activations + KV cache)
└── Working Buffers:   0.2GB (logits, attention scores)
```

With unified memory management, initialization is 20-30% faster than traditional multi-allocation approach.

## Implementation Details

### CUDA Kernels
- **RMSNorm**: Vectorized BF16x2 with block reduction
- **RoPE**: Rotary positional embedding with complex number optimization
- **Attention**: 
  - QK^T kernel for attention scores
  - Softmax with numerical stability
  - V aggregation kernel
- **SwiGLU**: Fused gate and up projections
- **MatMul**: cuBLAS GEMM with Tensor Cores

### Precision
- **Weights**: BF16 throughout
- **Activations**: BF16 for memory efficiency
- **Attention Scores**: FP32 accumulation for numerical stability
- **Logits**: FP32 for sampling accuracy

## Debugging

### Build with Debug Symbols
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

### Run with CUDA-GDB
```bash
cuda-gdb-python3.12-tui ./qwen600_engine

# Single query mode
(cuda-gdb) run ../Qwen3-0.6B -i "Test"

# OR: Interactive chat mode
(cuda-gdb) run ../Qwen3-0.6B -r 1 -t 0.65 -p 0.9 -k 20

# OR: Debug with benchmark parameters (temperature variation)
(cuda-gdb) run ../Qwen3-0.6B -t 0.6 -i "What is machine learning?"

# OR: Debug with advanced sampling (like benchmark tests)
(cuda-gdb) run ../Qwen3-0.6B -t 0.7 -k 40 -R 1.15 -m 0.05 -i "Explain AI"
```

### CUDA-GDB Quick Reference

```bash
# Build with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug .. && make

# Run with debugger
cuda-gdb-python3.12-tui ./qwen600_engine
(cuda-gdb) run ../Qwen3-0.6B -r 1 -t 0.65 -p 0.9 -k 20

# Common commands
n, next          # Next line
s, step          # Step into function
c, continue      # Continue to breakpoint
bt               # Show call stack
up               # Go up one stack frame
down             # Go back down
frame N          # Jump to frame N

# Display Control (TUI mode enabled by default with cuda-gdb-python3.12-tui)
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
```

### Debugging Commands (what you CAN run in CUDA-GDB)

```bash
# Breakpoints
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
```

### Check GPU Memory
```bash
# Monitor GPU usage
watch -n 0.5 nvidia-smi

# Check for memory leaks
cuda-memcheck ./qwen600_engine ../Qwen3-0.6B -i "Test"
```

## Performance Tuning Tips

### For Speed
- Use greedy decoding: `-t 0`
- Disable penalties: `-R 1.0 -F 0 -P 0`
- Reduce top-k: `-k 10`

### For Quality
- Moderate temperature: `-t 0.6-0.8`
- Use repetition penalty: `-R 1.1-1.3`
- Higher min-p: `-m 0.1`

### For Creativity
- Higher temperature: `-t 0.9-1.0`
- Higher top-p: `-p 0.95-0.98`
- Lower penalties: `-R 1.05`

## Troubleshooting

### Out of Memory
- Check available GPU memory: `nvidia-smi`
- Model requires ~2.4GB VRAM minimum
- Close other GPU applications

### Slow Inference
- Verify GPU utilization: `nvidia-smi dmon`
- Profile with: `nsys profile ./qwen600_engine ...`
- Ensure CUDA Toolkit is properly installed

### Poor Output Quality
- Adjust repetition penalty: `-R 1.2-1.5`
- Try min-p sampling: `-m 0.05-0.15`
- Increase temperature: `-t 0.7-0.9`

## Advanced Topics

See `docs/` for more information:
- `docs/README.md`: Complete user guide
- `docs/OPTIMIZATIONS.md`: Technical optimization details

## Credits

Originally based on [yassa9/qwen600](https://github.com/yassa9/qwen600), extensively enhanced with:
- Production-grade memory management
- Advanced sampling techniques
- Comprehensive error handling
- Performance optimization tools

## License

MIT License - see LICENSE file for details.
