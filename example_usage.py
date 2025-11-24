#!/usr/bin/env python
"""
Simple example of using Qwen3-0.6B CUDA inference engine from Python.

Before running:
    export PYTHONPATH=$(pwd)/python:$PYTHONPATH
    python example_usage.py
"""

import sys
import os

# Add python/ directory to path (so you can run directly)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from qwen_engine import QwenInferenceSession, SamplingConfig

# Create session
session = QwenInferenceSession("Qwen3-0.6B")

# Generate text
output = session.generate("What is machine learning?")
print(output)

# With custom configuration
# config = SamplingConfig(temperature=0.8, top_k=40, max_tokens=200)
# output = session.generate("Tell me a story", config)
# print(output)

# Cleanup
session.close()

