"""
Python wrapper for Qwen3-0.6B CUDA inference engine.

This module provides a high-level Python API similar to ONNX Runtime
for easy integration into Python applications.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import os

# Import the C++ extension (will be built by pybind11)
try:
    from . import _qwen_core
except ImportError as e:
    raise ImportError(
        "Failed to import C++ extension. "
        "Please ensure the package is properly installed: pip install -e ."
    ) from e


@dataclass
class SamplingConfig:
    """Configuration for text generation sampling.
    
    Similar to ONNX Runtime's generation config, this class controls
    how the model generates text.
    
    Args:
        temperature: Sampling temperature (0.0 = greedy, higher = more random)
        top_k: Top-k sampling (keep only top k tokens)
        top_p: Top-p/nucleus sampling (cumulative probability threshold)
        min_p: Min-p sampling (minimum probability threshold)
        repetition_penalty: Penalty for repeating tokens (>1.0 discourages)
        frequency_penalty: Penalty based on token frequency
        presence_penalty: Flat penalty for any used token
        penalty_window: Number of recent tokens to track for penalties
        max_tokens: Maximum number of tokens to generate
        seed: Random seed for reproducibility (None = random)
        reasoning_mode: Enable thinking/reasoning mode (0=off, 1=on)
    
    Example:
        >>> config = SamplingConfig(temperature=0.7, top_k=40, max_tokens=100)
        >>> output = session.generate("Hello", config)
    """
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for C++ backend."""
        return {
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'min_p': self.min_p,
            'repetition_penalty': self.repetition_penalty,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'penalty_window': self.penalty_window,
            'max_tokens': self.max_tokens,
            'seed': self.seed if self.seed is not None else -1,
            'reasoning_mode': self.reasoning_mode,
        }


class QwenInferenceSession:
    """High-performance CUDA inference session for Qwen3-0.6B.
    
    This class provides a simple, ONNX Runtime-like interface for
    running inference with the Qwen3-0.6B model on CUDA.
    
    Example:
        >>> # Initialize session (loads model to GPU)
        >>> session = QwenInferenceSession("path/to/Qwen3-0.6B")
        >>> 
        >>> # Single query
        >>> output = session.generate("What is AI?")
        >>> print(output)
        >>> 
        >>> # With custom sampling
        >>> config = SamplingConfig(temperature=0.8, top_k=40)
        >>> output = session.generate("Tell me a story", config)
        >>> 
        >>> # Chat mode with system prompt
        >>> session.set_system_prompt("You are a helpful assistant")
        >>> response = session.generate("Hello!")
        >>> 
        >>> # Cleanup (automatic on deletion, but can be explicit)
        >>> session.close()
    """
    
    def __init__(self, model_path: str, device: int = 0, reasoning_mode: int = 0):
        """Initialize the inference session.
        
        Args:
            model_path: Path to model directory containing:
                - model.safetensors
                - tokenizer.bin
                - template_*.txt files
            device: CUDA device ID (default: 0)
            reasoning_mode: Enable thinking mode: 0=off, 1=on (default: 0)
        
        Raises:
            FileNotFoundError: If model files not found
            RuntimeError: If GPU not available or initialization fails
        """
        # Set _closed first to avoid AttributeError in __del__ if initialization fails
        self._closed = True  # Mark as closed by default, set to False after successful init
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        required_files = [
            "model.safetensors",
            "tokenizer.bin",
            "template_user.txt",
            "template_system.txt"
        ]
        
        for filename in required_files:
            filepath = os.path.join(model_path, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Required file not found: {filepath}\n"
                    f"Please ensure the model directory contains all required files."
                )
        
        # Initialize C++ backend
        try:
            self._session = _qwen_core.InferenceSession(model_path, device, reasoning_mode)
            self._model_path = model_path
            self._device = device
            self._reasoning_mode = reasoning_mode
            self._closed = False  # Only set to False after successful initialization
        except Exception as e:
            raise RuntimeError(f"Failed to initialize inference session: {e}")
        
        print(f"✓ Loaded Qwen3-0.6B model from {model_path}")
        print(f"✓ Using GPU device {device}")
    
    def generate(
        self, 
        prompt: str,
        config: Optional[SamplingConfig] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            config: Sampling configuration (uses defaults if None)
            system_prompt: Optional system prompt for this generation
            stream: If True, yield tokens as they're generated (TODO)
        
        Returns:
            Generated text as a string
        
        Raises:
            RuntimeError: If session is closed or generation fails
        
        Example:
            >>> output = session.generate(
            ...     "Explain quantum computing",
            ...     SamplingConfig(temperature=0.7, max_tokens=200)
            ... )
        """
        if self._closed:
            raise RuntimeError("Session is closed. Create a new session.")
        
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        
        # Use default config if not provided
        if config is None:
            config = SamplingConfig()
        
        # Convert config to dict for C++ backend
        config_dict = config.to_dict()
        
        # Call C++ backend
        try:
            output = self._session.generate(
                prompt, 
                config_dict,
                system_prompt if system_prompt else ""
            )
            return output
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
    
    def set_system_prompt(self, system_prompt: str):
        """Set a persistent system prompt for all subsequent generations.
        
        Args:
            system_prompt: System prompt text (e.g., "You are a helpful assistant")
        
        Example:
            >>> session.set_system_prompt("You are a creative writer")
            >>> story = session.generate("Write a short story")
        """
        if self._closed:
            raise RuntimeError("Session is closed")
        
        self._session.set_system_prompt(system_prompt)
    
    def reset_conversation(self):
        """Reset the conversation history and sampling state.
        
        Useful when starting a new conversation or clearing context.
        """
        if self._closed:
            raise RuntimeError("Session is closed")
        
        self._session.reset_state()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration.
        
        Returns:
            Dictionary containing model metadata
        """
        if self._closed:
            raise RuntimeError("Session is closed")
        
        return {
            'model_path': self._model_path,
            'device': self._device,
            'reasoning_mode': self._reasoning_mode,
            'model_name': 'Qwen3-0.6B',
            'context_length': 8192,
            'vocab_size': 151936,
            'num_layers': 28,
            'num_heads': 16,
            'hidden_dim': 1024,
        }
    
    def close(self):
        """Explicitly close the session and free GPU memory.
        
        This is called automatically when the object is deleted,
        but you can call it explicitly for immediate cleanup.
        """
        if not self._closed:
            self._session.cleanup()
            self._closed = True
            print("✓ Session closed, GPU memory freed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
    
    def __del__(self):
        """Destructor - ensures GPU memory is freed."""
        if not self._closed:
            self.close()
    
    def __repr__(self) -> str:
        status = "closed" if self._closed else "active"
        return f"QwenInferenceSession(model='{self._model_path}', device={self._device}, status={status})"


# Convenience function for one-off generation
def generate_text(
    model_path: str,
    prompt: str,
    config: Optional[SamplingConfig] = None,
    device: int = 0,
    reasoning_mode: int = 0
) -> str:
    """Convenience function for one-off text generation.
    
    Creates a session, generates text, and cleans up automatically.
    Use QwenInferenceSession directly for multiple generations (more efficient).
    
    Args:
        model_path: Path to model directory
        prompt: Input prompt
        config: Sampling configuration
        device: CUDA device ID
        reasoning_mode: Enable thinking mode: 0=off, 1=on (default: 0)
    
    Returns:
        Generated text
    
    Example:
        >>> from qwen_engine import generate_text, SamplingConfig
        >>> output = generate_text(
        ...     "path/to/Qwen3-0.6B",
        ...     "What is machine learning?",
        ...     SamplingConfig(temperature=0.7)
        ... )
    """
    with QwenInferenceSession(model_path, device, reasoning_mode) as session:
        return session.generate(prompt, config)

