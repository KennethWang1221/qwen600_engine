"""
Simple usage examples for Qwen CUDA Inference Engine Python API.

These examples demonstrate how to use the Python wrapper
similar to ONNX Runtime or other inference libraries.
"""

from qwen_engine import QwenInferenceSession, SamplingConfig, generate_text


def example1_basic_usage():
    """Example 1: Basic text generation."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create inference session
    session = QwenInferenceSession("path/to/Qwen3-0.6B")
    
    # Generate text with default settings
    output = session.generate("What is machine learning?")
    print(f"Q: What is machine learning?")
    print(f"A: {output}\n")
    
    # Cleanup
    session.close()


def example2_custom_sampling():
    """Example 2: Custom sampling configuration."""
    print("=" * 60)
    print("Example 2: Custom Sampling")
    print("=" * 60)
    
    session = QwenInferenceSession("path/to/Qwen3-0.6B")
    
    # Create custom sampling config
    config = SamplingConfig(
        temperature=0.8,      # More creative
        top_k=40,            # Consider top 40 tokens
        repetition_penalty=1.2,  # Discourage repetition
        max_tokens=200       # Generate up to 200 tokens
    )
    
    output = session.generate("Tell me a short story about a robot", config)
    print(f"Story: {output}\n")
    
    session.close()


def example3_context_manager():
    """Example 3: Using context manager (recommended)."""
    print("=" * 60)
    print("Example 3: Context Manager (Recommended)")
    print("=" * 60)
    
    # Automatic cleanup with 'with' statement
    with QwenInferenceSession("path/to/Qwen3-0.6B") as session:
        config = SamplingConfig(temperature=0.7, max_tokens=150)
        
        # Multiple generations in same session (efficient!)
        questions = [
            "What is Python?",
            "Explain quantum computing briefly",
            "What are neural networks?"
        ]
        
        for q in questions:
            output = session.generate(q, config)
            print(f"Q: {q}")
            print(f"A: {output}\n")
    
    # Session automatically closed here!


def example4_system_prompt():
    """Example 4: Using system prompts."""
    print("=" * 60)
    print("Example 4: System Prompts")
    print("=" * 60)
    
    with QwenInferenceSession("path/to/Qwen3-0.6B") as session:
        # Set a system prompt
        session.set_system_prompt("You are a helpful Python programming tutor")
        
        config = SamplingConfig(temperature=0.6, max_tokens=200)
        
        # All generations will use this system prompt
        output1 = session.generate("How do I read a file in Python?", config)
        print(f"Q: How do I read a file in Python?")
        print(f"A: {output1}\n")
        
        output2 = session.generate("What's the difference between list and tuple?", config)
        print(f"Q: What's the difference between list and tuple?")
        print(f"A: {output2}\n")


def example5_one_off_generation():
    """Example 5: One-off generation (convenience function)."""
    print("=" * 60)
    print("Example 5: One-off Generation")
    print("=" * 60)
    
    # For single generations, use the convenience function
    output = generate_text(
        model_path="path/to/Qwen3-0.6B",
        prompt="What is the capital of France?",
        config=SamplingConfig(temperature=0.3, max_tokens=50)
    )
    
    print(f"Q: What is the capital of France?")
    print(f"A: {output}\n")


def example6_reasoning_mode():
    """Example 6: Reasoning mode (thinking tokens)."""
    print("=" * 60)
    print("Example 6: Reasoning Mode")
    print("=" * 60)
    
    with QwenInferenceSession("path/to/Qwen3-0.6B") as session:
        # Enable reasoning mode
        config = SamplingConfig(
            temperature=0.6,
            reasoning_mode=1,  # Show thinking process
            max_tokens=300
        )
        
        output = session.generate(
            "Solve this: If a train travels 120 km in 2 hours, what's its speed?",
            config
        )
        print(f"Problem: If a train travels 120 km in 2 hours, what's its speed?")
        print(f"Solution (with reasoning): {output}\n")


def example7_advanced_sampling():
    """Example 7: Advanced sampling with all parameters."""
    print("=" * 60)
    print("Example 7: Advanced Sampling")
    print("=" * 60)
    
    with QwenInferenceSession("path/to/Qwen3-0.6B") as session:
        # Fine-tuned configuration for creative writing
        config = SamplingConfig(
            temperature=0.9,           # High creativity
            top_k=50,                  # Wide token selection
            top_p=0.95,                # Nucleus sampling
            min_p=0.05,                # Quality threshold
            repetition_penalty=1.15,   # Reduce repetition
            frequency_penalty=0.1,     # Penalize common words
            presence_penalty=0.05,     # Encourage diversity
            penalty_window=128,        # Track more history
            max_tokens=400,            # Longer output
            seed=42                    # Reproducible results
        )
        
        output = session.generate("Write a haiku about programming", config)
        print(f"Creative haiku:\n{output}\n")


def example8_model_info():
    """Example 8: Getting model information."""
    print("=" * 60)
    print("Example 8: Model Information")
    print("=" * 60)
    
    with QwenInferenceSession("path/to/Qwen3-0.6B") as session:
        info = session.get_model_info()
        
        print("Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()


def example9_error_handling():
    """Example 9: Proper error handling."""
    print("=" * 60)
    print("Example 9: Error Handling")
    print("=" * 60)
    
    try:
        # This will fail if model path is wrong
        session = QwenInferenceSession("invalid/path")
    except FileNotFoundError as e:
        print(f"✗ Model not found: {e}")
    except RuntimeError as e:
        print(f"✗ Runtime error: {e}")
    
    try:
        with QwenInferenceSession("path/to/Qwen3-0.6B") as session:
            # Empty prompt will raise ValueError
            output = session.generate("")
    except ValueError as e:
        print(f"✗ Invalid input: {e}")
    
    print("✓ Error handling works correctly\n")


def example10_batch_processing():
    """Example 10: Batch processing multiple prompts."""
    print("=" * 60)
    print("Example 10: Batch Processing")
    print("=" * 60)
    
    prompts = [
        "What is AI?",
        "Explain machine learning",
        "What is deep learning?",
        "Define neural networks",
        "What is NLP?"
    ]
    
    # Process all prompts in single session (efficient!)
    with QwenInferenceSession("path/to/Qwen3-0.6B") as session:
        config = SamplingConfig(temperature=0.6, max_tokens=100)
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            output = session.generate(prompt, config)
            results.append((prompt, output))
            print(f"[{i}/{len(prompts)}] Processed: {prompt[:30]}...")
        
        print("\nResults:")
        for prompt, output in results:
            print(f"\nQ: {prompt}")
            print(f"A: {output[:100]}...")  # Show first 100 chars


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Qwen CUDA Engine - Python API Examples")
    print("=" * 60 + "\n")
    
    print("NOTE: Replace 'path/to/Qwen3-0.6B' with your actual model path!\n")
    
    # Run examples (comment out ones you don't want to run)
    # example1_basic_usage()
    # example2_custom_sampling()
    # example3_context_manager()
    # example4_system_prompt()
    # example5_one_off_generation()
    # example6_reasoning_mode()
    # example7_advanced_sampling()
    # example8_model_info()
    # example9_error_handling()
    # example10_batch_processing()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60 + "\n")

