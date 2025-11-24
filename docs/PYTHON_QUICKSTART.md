# Python Quick Start Guide

Get started with Qwen Engine in 5 minutes!

## Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/qwen600_engine
cd qwen600_engine

# 2. Install dependencies
pip install pybind11

# 3. Install package
pip install -e .
```

## Verify Installation

```bash
python python/tests/test_installation.py
```

You should see:
```
✓ Package imported successfully (version 2.0.0)
✓ C++ extension loaded successfully
✓ SamplingConfig works
✓ QwenInferenceSession class available
✓ Installation successful! Ready to use Qwen Engine.
```

## Your First Inference

### Step 1: Get the Model

```bash
# Download Qwen3-0.6B
git clone https://huggingface.co/Qwen/Qwen3-0.6B

# Export to SafeTensors format
python export.py Qwen3-0.6B
```

### Step 2: Run Your First Query

Create a file `my_first_inference.py`:

```python
from qwen_engine import QwenInferenceSession, SamplingConfig

# Initialize session (loads model to GPU)
session = QwenInferenceSession("Qwen3-0.6B")

# Generate text
output = session.generate("What is artificial intelligence?")
print(output)

# Cleanup
session.close()
```

Run it:
```bash
python my_first_inference.py
```

### Step 3: Customize Sampling

```python
from qwen_engine import QwenInferenceSession, SamplingConfig

with QwenInferenceSession("Qwen3-0.6B") as session:
    # More creative output
    creative_config = SamplingConfig(
        temperature=0.9,     # Higher = more creative
        top_k=50,           # Consider more options
        max_tokens=200      # Longer output
    )
    
    story = session.generate("Tell me a story about a robot", creative_config)
    print(story)
    
    # More focused output
    focused_config = SamplingConfig(
        temperature=0.3,    # Lower = more focused
        top_k=10,          # Fewer options
        max_tokens=100     # Shorter output
    )
    
    answer = session.generate("What is 2+2?", focused_config)
    print(answer)
```

## Common Patterns

### Pattern 1: Single Query (Simple)

```python
from qwen_engine import generate_text

# One-liner for single queries
output = generate_text("Qwen3-0.6B", "Hello, how are you?")
print(output)
```

### Pattern 2: Multiple Queries (Efficient)

```python
from qwen_engine import QwenInferenceSession, SamplingConfig

# Reuse session for multiple queries (much faster!)
with QwenInferenceSession("Qwen3-0.6B") as session:
    config = SamplingConfig(temperature=0.7)
    
    questions = [
        "What is Python?",
        "What is machine learning?",
        "What is deep learning?"
    ]
    
    for q in questions:
        answer = session.generate(q, config)
        print(f"Q: {q}")
        print(f"A: {answer}\n")
```

### Pattern 3: With System Prompt

```python
from qwen_engine import QwenInferenceSession

with QwenInferenceSession("Qwen3-0.6B") as session:
    # Set a personality
    session.set_system_prompt("You are a friendly Python tutor")
    
    # All subsequent queries use this personality
    response = session.generate("How do I read a file?")
    print(response)
```

### Pattern 4: Production-Ready

```python
from qwen_engine import QwenInferenceSession, SamplingConfig

try:
    with QwenInferenceSession("Qwen3-0.6B") as session:
        config = SamplingConfig(
            temperature=0.6,
            repetition_penalty=1.15,  # Reduce repetition
            seed=42                   # Reproducible results
        )
        
        output = session.generate("Your prompt", config)
        print(output)
        
except FileNotFoundError as e:
    print(f"Model not found: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
```

## Configuration Cheat Sheet

### For Creative Writing
```python
SamplingConfig(
    temperature=0.9,         # High creativity
    top_k=50,
    repetition_penalty=1.2,  # Avoid repetition
    max_tokens=500          # Long output
)
```

### For Factual Q&A
```python
SamplingConfig(
    temperature=0.3,  # Focused, precise
    top_k=10,
    min_p=0.1,       # Quality threshold
    max_tokens=150   # Concise answers
)
```

### For Code Generation
```python
SamplingConfig(
    temperature=0.4,
    top_k=15,
    repetition_penalty=1.1,
    max_tokens=300
)
```

### For Reasoning Tasks
```python
SamplingConfig(
    temperature=0.6,
    reasoning_mode=1,  # Show thinking process
    max_tokens=400
)
```

## Performance Tips

### ✓ DO: Reuse Sessions
```python
# ✓ GOOD: Create session once, use many times
with QwenInferenceSession("model") as session:
    for prompt in many_prompts:
        output = session.generate(prompt)
```

### ✗ DON'T: Create New Sessions
```python
# ✗ BAD: Creates new session each time (very slow!)
for prompt in many_prompts:
    session = QwenInferenceSession("model")
    output = session.generate(prompt)
    session.close()
```

### ✓ DO: Use Context Manager
```python
# ✓ GOOD: Automatic cleanup
with QwenInferenceSession("model") as session:
    output = session.generate("prompt")
# GPU memory freed automatically
```

### ✗ DON'T: Forget to Close
```python
# ✗ BAD: GPU memory leak
session = QwenInferenceSession("model")
output = session.generate("prompt")
# Forgot to close! Memory still allocated
```

## Troubleshooting

### "ImportError: cannot import name '_qwen_core'"

**Problem:** C++ extension not built correctly

**Solution:**
```bash
pip install --force-reinstall -e . --verbose
# Check output for errors
```

### "FileNotFoundError: model.safetensors not found"

**Problem:** Model files missing

**Solution:**
Ensure your model directory has:
- `model.safetensors`
- `tokenizer.bin`
- `template_*.txt`

### "RuntimeError: CUDA out of memory"

**Problem:** Not enough GPU memory

**Solution:**
- Close other GPU applications
- Use a GPU with 3GB+ VRAM
- Qwen3-0.6B needs ~2.4GB

### Output is nonsensical

**Problem:** Model files corrupted or wrong version

**Solution:**
```bash
# Re-export model
python export.py Qwen3-0.6B

# Verify SHA256 hash (if available)
sha256sum Qwen3-0.6B/model.safetensors
```

## Next Steps

- **Full API Documentation**: [PYTHON_API.md](PYTHON_API.md)
- **More Examples**: [python/examples/](../python/examples/)
- **Installation Guide**: [PYTHON_INSTALLATION.md](PYTHON_INSTALLATION.md)
- **C++ Documentation**: [README.md](../README.md)

## Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Documentation**: Check the `docs/` directory
- **Examples**: See `python/examples/` for code samples

---

**Happy coding! 🚀**

