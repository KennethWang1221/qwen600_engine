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

## Next Steps

1. ✅ Verify Phase 0 passes all tests
2. 📖 Read [Learning Guide](docs/LEARNING_GUIDE.md) Week 1-2
3. 🔧 Implement RMSNorm kernel
4. 🧪 Test and verify
5. 🚀 Continue with remaining kernels

## License

MIT License

---

**Current Status**: Phase 0 complete, ready for kernel implementation! 🚀
