"""
Example: Using Qwen Engine with ONNX Runtime-style API

This example demonstrates how Qwen Engine's API is similar to ONNX Runtime,
making it easy for users familiar with ONNX to switch to our faster CUDA engine.
"""

from qwen_engine import QwenInferenceSession, SamplingConfig


def onnx_style_usage():
    """
    Demonstrate ONNX Runtime-style usage pattern.
    
    In ONNX Runtime, you typically:
    1. Create InferenceSession with model path
    2. Run inference with inputs
    3. Get outputs
    
    Qwen Engine follows the same pattern!
    """
    
    print("=" * 70)
    print("ONNX Runtime-Style Usage with Qwen Engine")
    print("=" * 70)
    
    # Step 1: Create inference session (like onnxruntime.InferenceSession)
    print("\n[1/3] Creating inference session...")
    session = QwenInferenceSession("path/to/Qwen3-0.6B")
    print("✓ Session created")
    
    # Step 2: Prepare configuration (like ONNX generation config)
    print("\n[2/3] Configuring generation parameters...")
    config = SamplingConfig(
        temperature=0.7,
        top_k=40,
        max_tokens=200
    )
    print("✓ Configuration ready")
    
    # Step 3: Run inference (like session.run())
    print("\n[3/3] Running inference...")
    prompts = [
        "What is artificial intelligence?",
        "Explain neural networks briefly",
        "What is the future of AI?"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Query {i}/{len(prompts)} ---")
        print(f"Prompt: {prompt}")
        
        # This is like: outputs = session.run(None, {"input": prompt})
        output = session.generate(prompt, config)
        
        print(f"Output: {output[:150]}...")  # Show first 150 chars
    
    # Step 4: Cleanup (like ONNX session cleanup)
    print("\n[4/4] Cleaning up...")
    session.close()
    print("✓ Session closed\n")


def comparison_example():
    """
    Side-by-side comparison of ONNX Runtime vs Qwen Engine API.
    """
    
    print("=" * 70)
    print("API Comparison: ONNX Runtime vs Qwen Engine")
    print("=" * 70)
    
    print("\n" + "─" * 70)
    print("ONNX Runtime (Pseudo-code):")
    print("─" * 70)
    print("""
    import onnxruntime as ort
    
    # Create session
    session = ort.InferenceSession("model.onnx",
                                   providers=['CUDAExecutionProvider'])
    
    # Run inference
    inputs = {"input_ids": input_data}
    outputs = session.run(None, inputs)
    
    # Get results
    result = outputs[0]
    """)
    
    print("\n" + "─" * 70)
    print("Qwen Engine (Actual code):")
    print("─" * 70)
    print("""
    from qwen_engine import QwenInferenceSession, SamplingConfig
    
    # Create session (automatic GPU selection)
    session = QwenInferenceSession("Qwen3-0.6B")
    
    # Run inference (tokenization automatic!)
    config = SamplingConfig(temperature=0.7)
    output = session.generate("Hello", config)
    
    # Result is ready
    print(output)
    """)
    
    print("\n" + "─" * 70)
    print("Key Advantages of Qwen Engine:")
    print("─" * 70)
    print("✓ No manual tokenization needed")
    print("✓ Built-in sampling strategies")
    print("✓ Optimized CUDA kernels (faster than generic ONNX)")
    print("✓ Simpler API for text generation")
    print("✓ Direct string input/output")
    print()


def production_example():
    """
    Production-ready example with error handling and resource management.
    """
    
    print("=" * 70)
    print("Production-Ready Example")
    print("=" * 70)
    
    try:
        # Use context manager (automatic cleanup)
        with QwenInferenceSession("path/to/Qwen3-0.6B") as session:
            
            # Get model info
            info = session.get_model_info()
            print(f"\nModel: {info['model_name']}")
            print(f"Context Length: {info['context_length']}")
            print(f"Device: GPU {info['device']}\n")
            
            # Production config
            config = SamplingConfig(
                temperature=0.6,          # Balanced
                top_k=20,                 # Controlled diversity
                repetition_penalty=1.15,  # Reduce repetition
                max_tokens=256,           # Reasonable length
                seed=42                   # Reproducible
            )
            
            # Run inference
            prompt = "Explain quantum computing in simple terms"
            print(f"Query: {prompt}\n")
            
            output = session.generate(prompt, config)
            print(f"Response:\n{output}\n")
            
    except FileNotFoundError as e:
        print(f"✗ Model not found: {e}")
    except RuntimeError as e:
        print(f"✗ Runtime error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print("✓ Completed successfully")


def batch_processing_example():
    """
    Efficient batch processing like ONNX Runtime.
    """
    
    print("=" * 70)
    print("Batch Processing Example")
    print("=" * 70)
    
    # Simulate a batch of queries (like in production)
    batch_queries = [
        "What is machine learning?",
        "Define deep learning",
        "What is NLP?",
        "Explain computer vision",
        "What are transformers?",
    ]
    
    print(f"\nProcessing {len(batch_queries)} queries...\n")
    
    with QwenInferenceSession("path/to/Qwen3-0.6B") as session:
        config = SamplingConfig(
            temperature=0.6,
            max_tokens=100,
            seed=42  # Reproducible results
        )
        
        results = []
        for i, query in enumerate(batch_queries, 1):
            print(f"[{i}/{len(batch_queries)}] Processing: {query[:40]}...")
            output = session.generate(query, config)
            results.append((query, output))
        
        print("\n" + "─" * 70)
        print("Results:")
        print("─" * 70)
        for query, output in results:
            print(f"\nQ: {query}")
            print(f"A: {output[:80]}...")  # First 80 chars
    
    print("\n✓ Batch processing complete")


def advanced_config_example():
    """
    Advanced configuration options similar to ONNX Generation Config.
    """
    
    print("=" * 70)
    print("Advanced Configuration Example")
    print("=" * 70)
    
    configs = {
        "creative_writing": SamplingConfig(
            temperature=0.9,
            top_k=50,
            repetition_penalty=1.2,
            frequency_penalty=0.1,
            presence_penalty=0.05,
        ),
        
        "factual_qa": SamplingConfig(
            temperature=0.3,
            top_k=10,
            min_p=0.1,
        ),
        
        "code_generation": SamplingConfig(
            temperature=0.4,
            top_k=15,
            repetition_penalty=1.1,
        ),
        
        "reasoning": SamplingConfig(
            temperature=0.6,
            reasoning_mode=1,
            max_tokens=300,
        ),
    }
    
    with QwenInferenceSession("path/to/Qwen3-0.6B") as session:
        for mode, config in configs.items():
            print(f"\n{'='*70}")
            print(f"Mode: {mode.upper()}")
            print('='*70)
            print(f"Config: temp={config.temperature}, "
                  f"top_k={config.top_k}, "
                  f"max_tokens={config.max_tokens}")
            
            # Example prompt for each mode
            if mode == "creative_writing":
                prompt = "Write a short poem about AI"
            elif mode == "factual_qa":
                prompt = "What is the capital of France?"
            elif mode == "code_generation":
                prompt = "Write a Python function to reverse a string"
            else:  # reasoning
                prompt = "If 5 cats can catch 5 mice in 5 minutes, how many cats are needed to catch 100 mice in 100 minutes?"
            
            output = session.generate(prompt, config)
            print(f"\nPrompt: {prompt}")
            print(f"Output: {output[:120]}...")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Qwen Engine - ONNX Runtime Style Examples")
    print("=" * 70 + "\n")
    
    print("NOTE: Replace 'path/to/Qwen3-0.6B' with your actual model path!\n")
    
    # Run examples (comment out ones you don't want)
    # onnx_style_usage()
    # comparison_example()
    # production_example()
    # batch_processing_example()
    # advanced_config_example()
    
    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70 + "\n")

