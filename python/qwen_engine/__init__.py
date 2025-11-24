"""
Qwen3-0.6B CUDA Inference Engine - Python Wrapper

A high-performance CUDA-based inference engine for Qwen3-0.6B model.
"""

from .qwen_inference import QwenInferenceSession, SamplingConfig

__version__ = "2.0.0"
__all__ = ["QwenInferenceSession", "SamplingConfig"]

# Simple usage example:
# >>> from qwen_engine import QwenInferenceSession, SamplingConfig
# >>> session = QwenInferenceSession("path/to/Qwen3-0.6B")
# >>> config = SamplingConfig(temperature=0.7, top_k=40)
# >>> output = session.generate("What is machine learning?", config)
# >>> print(output)

