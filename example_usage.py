#!/usr/bin/env python
"""
Simple example of using Qwen3-0.6B CUDA inference engine from Python.

Before running:
    1. Install the package: pip install -e .
    2. Run: python example_usage.py
"""

import sys
import os

# Add python/ directory to path (for development without pip install)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from qwen_engine import QwenInferenceSession, SamplingConfig

# Create session with model path
# Option 1: Use existing model in test directory
model_path = "/home/ai2/anaconda3/PROJECTS/test/Qwen3-0.6B"

# Option 2: Download model first if you don't have it:
#   git clone https://huggingface.co/Qwen/Qwen3-0.6B
#   python3 export.py Qwen3-0.6B
#   Then set: model_path = "Qwen3-0.6B"

session = QwenInferenceSession(model_path)

# Generate text
output = session.generate("What is machine learning?")
print(output)

# With custom configuration
# config = SamplingConfig(temperature=0.8, top_k=40, max_tokens=200)
# output = session.generate("Tell me a story", config)
# print(output)

# Cleanup
session.close()

