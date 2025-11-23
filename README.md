# Qwen3-0.6B Inference Engine

CUDA-based LLM inference engine for Qwen3-0.6B. Single-batch, static, GPU-only implementation for learning CUDA and transformer architecture.

## Quick Start

```bash
# 1. Get the model
git clone https://huggingface.co/Qwen/Qwen3-0.6B
python3 export.py Qwen3-0.6B

# 2. Build
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 3. Run
./qwen600_engine ../Qwen3-0.6B -r 1 -t 0.65 -p 0.9 -k 20
```

## Requirements

- CUDA Toolkit 11.0+
- CMake 3.15+
- C++17 compiler
- GPU with 3GB+ VRAM (RTX 3050 or better)
- cuBLAS library

## Project Structure

```
qwen600_engine/
├── main.cu                 # Main chat loop
├── config.h                # Model hyperparameters
├── qwen_model.cuh          # Transformer implementation
├── static_loader.h         # SafeTensors weight loader
├── tokenizer.h             # BPE tokenizer
├── sampler.h               # Top-k/top-p sampling
└── export.py               # HuggingFace tokenizer converter
```

## Usage

```bash
./qwen600_engine <model_dir> [options]

Options:
  -r <int>    reasoning mode (0=off, 1=thinking)
  -t <float>  temperature (default: 0.6)
  -p <float>  top-p (default: 0.95)
  -k <int>    top-k (default: 20)
  -s <int>    random seed
  -i <string> input prompt (single query mode)
  -y <string> system prompt
```

## Features

- ✅ SafeTensors weight loading with mmap
- ✅ BF16 precision throughout
- ✅ Grouped Query Attention (GQA)
- ✅ RoPE positional encoding
- ✅ SwiGLU activation
- ✅ QK-Norm (Q/K normalization)
- ✅ KV caching for efficiency
- ✅ Top-k/Top-p sampling
- ✅ Reasoning mode (thinking tokens)

## Implementation Details

- **Kernels**: RMSNorm (vectorized BF16x2), RoPE, Attention (QK^T, Softmax, V), SwiGLU
- **Matrix ops**: cuBLAS with Tensor Cores
- **Memory**: Single GPU allocation (~2.4GB total: 1.5GB weights + 0.9GB state)
- **Precision**: BF16 compute, FP32 accumulation for attention scores

## Debugging

### CUDA-GDB Quick Reference

```bash
# Build with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug .. && make

# Run with debugger
cuda-gdb ./qwen600_engine
(gdb) run ../Qwen3-0.6B -r 1

# TUI mode
layout src               # Show source
Ctrl-X-o                # Switch windows
refresh                 # Fix display

# Common commands
break main              # Breakpoint
next, step             # Navigate
print var              # Inspect
bt                     # Backtrace
```

## License

MIT License
