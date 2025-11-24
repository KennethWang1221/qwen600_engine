# Python API Documentation

## Overview

The Qwen CUDA Engine provides a high-level Python API similar to ONNX Runtime, making it easy to integrate high-performance CUDA inference into your Python applications.

---

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/yourusername/qwen600_engine
cd qwen600_engine

# Install in development mode
pip install -e .

# Or regular installation
pip install .
```

### Requirements

- Python 3.7+
- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compiler
- GPU with 3GB+ VRAM

---

## Quick Start

```python
from qwen_engine import QwenInferenceSession, SamplingConfig

# Initialize session (loads model to GPU)
session = QwenInferenceSession("path/to/Qwen3-0.6B")

# Generate text
output = session.generate("What is machine learning?")
print(output)

# With custom settings
config = SamplingConfig(temperature=0.8, top_k=40, max_tokens=200)
output = session.generate("Tell me a story", config)

# Cleanup
session.close()
```

---

## API Reference

### QwenInferenceSession

Main class for running inference.

#### Constructor

```python
QwenInferenceSession(model_path: str, device: int = 0)
```

**Parameters:**
- `model_path`: Path to model directory containing:
  - `model.safetensors`
  - `tokenizer.bin`
  - `template_*.txt` files
- `device`: CUDA device ID (default: 0)

**Raises:**
- `FileNotFoundError`: If model files not found
- `RuntimeError`: If GPU not available

**Example:**
```python
session = QwenInferenceSession("Qwen3-0.6B", device=0)
```

#### Methods

##### `generate()`

Generate text from a prompt.

```python
generate(
    prompt: str,
    config: Optional[SamplingConfig] = None,
    system_prompt: Optional[str] = None,
    stream: bool = False
) -> str
```

**Parameters:**
- `prompt`: Input text prompt
- `config`: Sampling configuration (uses defaults if None)
- `system_prompt`: Optional system prompt for this generation
- `stream`: If True, yield tokens as generated (TODO)

**Returns:**
- Generated text as string

**Example:**
```python
output = session.generate(
    "Explain quantum computing",
    SamplingConfig(temperature=0.7, max_tokens=200)
)
```

##### `set_system_prompt()`

Set a persistent system prompt.

```python
set_system_prompt(system_prompt: str)
```

**Example:**
```python
session.set_system_prompt("You are a helpful assistant")
output = session.generate("Hello!")  # Uses system prompt
```

##### `reset_conversation()`

Reset conversation history and sampling state.

```python
reset_conversation()
```

##### `get_model_info()`

Get model information and configuration.

```python
get_model_info() -> Dict[str, Any]
```

**Returns:**
- Dictionary with model metadata

**Example:**
```python
info = session.get_model_info()
print(f"Context length: {info['context_length']}")
print(f"Vocab size: {info['vocab_size']}")
```

##### `close()`

Explicitly close session and free GPU memory.

```python
close()
```

---

### SamplingConfig

Configuration for text generation sampling.

```python
@dataclass
class SamplingConfig:
    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.05
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    penalty_window: int = 64
    max_tokens: int = 512
    seed: Optional[int] = None
    reasoning_mode: int = 0
```

**Parameters:**
- `temperature`: Sampling temperature (0.0 = greedy, higher = more random)
- `top_k`: Keep only top k tokens
- `top_p`: Cumulative probability threshold (nucleus sampling)
- `min_p`: Minimum probability threshold (quality-focused)
- `repetition_penalty`: Penalty for repeating tokens (>1.0 discourages)
- `frequency_penalty`: Penalty based on token frequency
- `presence_penalty`: Flat penalty for any used token
- `penalty_window`: Number of recent tokens to track
- `max_tokens`: Maximum tokens to generate
- `seed`: Random seed for reproducibility (None = random)
- `reasoning_mode`: Enable thinking mode (0=off, 1=on)

**Example:**
```python
config = SamplingConfig(
    temperature=0.8,
    top_k=40,
    repetition_penalty=1.2,
    max_tokens=300,
    seed=42  # Reproducible results
)
```

---

### Convenience Functions

#### `generate_text()`

One-off text generation (creates session, generates, cleans up).

```python
generate_text(
    model_path: str,
    prompt: str,
    config: Optional[SamplingConfig] = None,
    device: int = 0
) -> str
```

**Example:**
```python
from qwen_engine import generate_text, SamplingConfig

output = generate_text(
    "Qwen3-0.6B",
    "What is AI?",
    SamplingConfig(temperature=0.7)
)
```

---

## Usage Patterns

### Pattern 1: Context Manager (Recommended)

```python
with QwenInferenceSession("Qwen3-0.6B") as session:
    config = SamplingConfig(temperature=0.7)
    
    # Multiple generations (efficient!)
    for prompt in prompts:
        output = session.generate(prompt, config)
        print(output)
# Automatic cleanup!
```

### Pattern 2: Manual Management

```python
session = QwenInferenceSession("Qwen3-0.6B")

try:
    output = session.generate("Hello")
    print(output)
finally:
    session.close()  # Always cleanup
```

### Pattern 3: One-off Generation

```python
# For single generation, use convenience function
output = generate_text("Qwen3-0.6B", "Hello world")
```

---

## Advanced Examples

### Creative Writing

```python
config = SamplingConfig(
    temperature=0.9,       # High creativity
    top_k=50,
    min_p=0.05,
    repetition_penalty=1.2,  # Avoid repetition
    max_tokens=500
)

story = session.generate("Write a sci-fi story", config)
```

### Factual Q&A

```python
config = SamplingConfig(
    temperature=0.3,  # Low temperature = more focused
    top_k=10,
    max_tokens=150
)

answer = session.generate("What is the capital of France?", config)
```

### Reasoning Tasks

```python
config = SamplingConfig(
    temperature=0.6,
    reasoning_mode=1,  # Show thinking process
    max_tokens=300
)

solution = session.generate("Solve: 2x + 5 = 13", config)
```

---

## Error Handling

```python
try:
    session = QwenInferenceSession("path/to/model")
    output = session.generate("Hello")
except FileNotFoundError as e:
    print(f"Model not found: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
finally:
    if 'session' in locals():
        session.close()
```

---

## Performance Tips

### 1. Reuse Sessions

```python
# ✗ BAD: Creates new session for each generation (slow!)
for prompt in prompts:
    session = QwenInferenceSession("model")
    output = session.generate(prompt)
    session.close()

# ✓ GOOD: Reuse session (much faster!)
with QwenInferenceSession("model") as session:
    for prompt in prompts:
        output = session.generate(prompt)
```

### 2. Batch Similar Queries

```python
# Process similar queries together
with QwenInferenceSession("model") as session:
    config = SamplingConfig(temperature=0.7)
    
    results = [
        session.generate(prompt, config)
        for prompt in batch_prompts
    ]
```

### 3. Adjust max_tokens

```python
# Don't generate more than needed
short_config = SamplingConfig(max_tokens=50)   # For short answers
long_config = SamplingConfig(max_tokens=500)   # For stories
```

---

## Comparison with Other Libraries

### ONNX Runtime Style

```python
# ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": input_data})

# Qwen Engine (similar API)
from qwen_engine import QwenInferenceSession
session = QwenInferenceSession("Qwen3-0.6B")
output = session.generate("prompt")
```

### Transformers Library Style

```python
# Hugging Face Transformers
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Qwen Engine (simpler, faster)
from qwen_engine import QwenInferenceSession
session = QwenInferenceSession("Qwen3-0.6B")
# Tokenizer handled internally!
```

---

## FAQ

**Q: Can I use multiple GPU devices?**
A: Yes, specify `device` parameter: `QwenInferenceSession("model", device=1)`

**Q: How do I enable reasoning mode?**
A: Set `reasoning_mode=1` in SamplingConfig

**Q: Can I stream tokens as they're generated?**
A: Not yet implemented. Coming in future version.

**Q: How much GPU memory is needed?**
A: ~2.4 GB for Qwen3-0.6B model

**Q: Is it thread-safe?**
A: Each session should be used by one thread. Create separate sessions for multi-threading.

---

## Troubleshooting

### Import Error

```
ImportError: Failed to import C++ extension
```

**Solution:** Reinstall the package:
```bash
pip install --force-reinstall -e .
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Close other GPU applications or use a GPU with more memory.

### Model Files Not Found

```
FileNotFoundError: model.safetensors not found
```

**Solution:** Ensure model directory contains all required files:
- `model.safetensors`
- `tokenizer.bin`
- `template_*.txt`

---

## License

MIT License - see LICENSE file for details.

