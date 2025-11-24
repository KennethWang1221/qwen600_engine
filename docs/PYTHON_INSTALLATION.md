# Python Package Installation Guide

## Prerequisites

Before installing the Python package, ensure you have:

### System Requirements

1. **CUDA Toolkit** (11.0 or higher)
   ```bash
   nvcc --version  # Check CUDA version
   ```

2. **CMake** (3.18 or higher)
   ```bash
   cmake --version
   ```

3. **Python** (3.7 or higher)
   ```bash
   python --version
   ```

4. **C++17 Compiler**
   - GCC 7+ (Linux)
   - MSVC 2019+ (Windows)
   - Clang 8+ (macOS)

5. **GPU with 3GB+ VRAM**

---

## Installation Methods

### Method 1: Install from Source (Development)

This is recommended for development and customization.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/qwen600_engine
cd qwen600_engine

# 2. Install pybind11 (required for Python bindings)
pip install pybind11

# 3. Install in development mode
pip install -e .
```

**Development mode (`-e`)** allows you to modify the source code and see changes immediately without reinstalling.

### Method 2: Regular Installation

For production use:

```bash
git clone https://github.com/yourusername/qwen600_engine
cd qwen600_engine
pip install .
```

### Method 3: Install from PyPI (Coming Soon)

Once published to PyPI:

```bash
pip install qwen-cuda-engine
```

---

## Verify Installation

Test that the package is correctly installed:

```python
import qwen_engine
print(qwen_engine.__version__)  # Should print: 2.0.0

# Check if C++ extension loaded
from qwen_engine import QwenInferenceSession, SamplingConfig
print("✓ Python wrapper imported successfully")
```

---

## Build from Source (Manual)

If you need more control over the build process:

```bash
# 1. Create build directory
mkdir -p build && cd build

# 2. Configure with CMake
cmake .. \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DCMAKE_BUILD_TYPE=Release

# 3. Build
make -j4

# 4. Install Python package
cd ..
pip install -e .
```

---

## Troubleshooting

### Issue: "pybind11 not found"

```
CMake Error: Could not find pybind11
```

**Solution:**
```bash
pip install pybind11
```

### Issue: "CUDA not found"

```
CMake Error: Could not find CUDA
```

**Solution:**
Ensure CUDA is installed and `nvcc` is in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: "ImportError: cannot import name '_qwen_core'"

```python
ImportError: cannot import name '_qwen_core'
```

**Solution:**
The C++ extension failed to build. Reinstall with verbose output:
```bash
pip install --force-reinstall -e . --verbose
```

Check the output for compilation errors.

### Issue: "CUDA out of memory"

```
RuntimeError: CUDA error: out of memory
```

**Solution:**
- Close other GPU applications
- Use a GPU with more VRAM
- The model requires ~2.4GB VRAM

### Issue: Compilation fails with "unsupported GPU architecture"

**Solution:**
Edit `CMakeLists.txt` and adjust `CUDA_ARCHITECTURES`:
```cmake
# Find your GPU compute capability from:
# https://developer.nvidia.com/cuda-gpus
set_target_properties(_qwen_core PROPERTIES
    CUDA_ARCHITECTURES "75"  # Change to your GPU's compute capability
)
```

Common compute capabilities:
- Tesla T4: 75
- RTX 2080: 75
- RTX 3080: 86
- RTX 4090: 89
- A100: 80

---

## Platform-Specific Notes

### Linux

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    nvidia-cuda-toolkit

# Install package
pip install -e .
```

### Windows

```bash
# Use Visual Studio 2019+ with CUDA support
# Open "x64 Native Tools Command Prompt for VS 2019"

pip install -e .
```

### macOS

**Note:** CUDA is not supported on macOS. This package requires NVIDIA GPU.

---

## Uninstallation

```bash
pip uninstall qwen-cuda-engine
```

---

## Development Setup

For contributing to the project:

```bash
# 1. Clone and install in development mode
git clone https://github.com/yourusername/qwen600_engine
cd qwen600_engine
pip install -e ".[dev]"  # Includes dev dependencies

# 2. Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# 3. Run tests
pytest tests/

# 4. Format code
black python/
```

---

## Next Steps

After installation:

1. **Download the model**: Get Qwen3-0.6B weights
2. **Read the API docs**: See [PYTHON_API.md](PYTHON_API.md)
3. **Run examples**: Check [python/examples/](../python/examples/)
4. **Test your installation**:

```python
from qwen_engine import generate_text, SamplingConfig

output = generate_text(
    "path/to/Qwen3-0.6B",
    "What is machine learning?",
    SamplingConfig(temperature=0.7)
)
print(output)
```

---

## Getting Help

- **Documentation**: [docs/PYTHON_API.md](PYTHON_API.md)
- **Examples**: [python/examples/](../python/examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/qwen600_engine/issues)

---

## License

MIT License - see LICENSE file for details.

