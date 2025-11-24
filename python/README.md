# Qwen Engine - Python Package

High-performance CUDA inference engine for Qwen3-0.6B with an ONNX Runtime-style Python API.

## Features

- **🚀 Fast**: Optimized CUDA kernels with Tensor Cores
- **🐍 Pythonic**: Simple, intuitive API similar to ONNX Runtime
- **🎛️ Flexible**: Advanced sampling (temperature, top-k, top-p, min-p, penalties)
- **💾 Efficient**: Unified memory management, BF16 precision
- **🛡️ Robust**: Comprehensive error handling

## Quick Start

### Installation

```bash
pip install pybind11
pip install -e .
```

### Usage

```python
from qwen_engine import QwenInferenceSession, SamplingConfig

# Create session
session = QwenInferenceSession("path/to/Qwen3-0.6B")

# Generate text
output = session.generate("What is machine learning?")
print(output)

# Customize sampling
config = SamplingConfig(temperature=0.8, top_k=40, max_tokens=200)
output = session.generate("Tell me a story", config)

# Cleanup
session.close()
```

## Examples

See [examples/](examples/) for more:
- `simple_usage.py`: Basic usage patterns
- `onnx_style.py`: ONNX Runtime-style examples

## Documentation

- **Quick Start**: [docs/PYTHON_QUICKSTART.md](../docs/PYTHON_QUICKSTART.md)
- **Full API Reference**: [docs/PYTHON_API.md](../docs/PYTHON_API.md)
- **Installation Guide**: [docs/PYTHON_INSTALLATION.md](../docs/PYTHON_INSTALLATION.md)

## Requirements

- Python 3.7+
- CUDA Toolkit 11.0+
- GPU with 3GB+ VRAM
- pybind11

## Testing

```bash
python tests/test_installation.py
```

## License

MIT License - see [LICENSE](../LICENSE) for details.

