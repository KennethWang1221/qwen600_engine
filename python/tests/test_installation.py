"""
Test script to verify Qwen Engine installation.

Run this after installation to ensure everything works correctly.
"""

import sys

def test_import():
    """Test 1: Can we import the package?"""
    print("Test 1: Importing package...")
    try:
        import qwen_engine
        print(f"✓ Package imported successfully (version {qwen_engine.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import package: {e}")
        print("\nTroubleshooting:")
        print("  1. Reinstall: pip install --force-reinstall -e .")
        print("  2. Check Python version: python --version (need 3.7+)")
        print("  3. Check installation: pip show qwen-cuda-engine")
        return False


def test_cpp_extension():
    """Test 2: Is the C++ extension loaded?"""
    print("\nTest 2: Loading C++ extension...")
    try:
        from qwen_engine import _qwen_core
        print("✓ C++ extension loaded successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to load C++ extension: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure pybind11 is installed: pip install pybind11")
        print("  2. Ensure CUDA is installed: nvcc --version")
        print("  3. Rebuild: pip install --force-reinstall -e .")
        return False


def test_classes():
    """Test 3: Can we create class instances?"""
    print("\nTest 3: Testing class definitions...")
    try:
        from qwen_engine import QwenInferenceSession, SamplingConfig
        
        # Test SamplingConfig
        config = SamplingConfig(temperature=0.7, top_k=40)
        assert config.temperature == 0.7
        assert config.top_k == 40
        print("✓ SamplingConfig works")
        
        # Test QwenInferenceSession exists (don't initialize without model)
        assert QwenInferenceSession is not None
        print("✓ QwenInferenceSession class available")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test classes: {e}")
        return False


def test_cuda():
    """Test 4: Is CUDA available?"""
    print("\nTest 4: Checking CUDA availability...")
    try:
        # Try to create a test session (will fail if no model, but tests CUDA)
        from qwen_engine import QwenInferenceSession
        
        # We can't actually test without a model, but we can check imports
        import os
        if os.system("nvcc --version > /dev/null 2>&1") == 0:
            print("✓ CUDA toolkit detected")
            return True
        else:
            print("⚠ CUDA toolkit not found in PATH")
            print("  This may cause issues when initializing models")
            return True  # Not a fatal error for installation test
    except Exception as e:
        print(f"⚠ Could not verify CUDA: {e}")
        return True  # Not a fatal error


def print_system_info():
    """Print system information."""
    print("\n" + "="*60)
    print("System Information")
    print("="*60)
    
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    
    try:
        import qwen_engine
        print(f"Qwen Engine version: {qwen_engine.__version__}")
        print(f"Qwen Engine path: {qwen_engine.__file__}")
    except:
        pass
    
    # Check CUDA
    import os
    if os.system("nvcc --version > /dev/null 2>&1") == 0:
        os.system("nvcc --version | grep 'release' | head -1")
    
    print("="*60)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Qwen Engine Installation Test")
    print("="*60)
    
    tests = [
        test_import,
        test_cpp_extension,
        test_classes,
        test_cuda,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ Installation successful! Ready to use Qwen Engine.")
        print("\nNext steps:")
        print("  1. Download Qwen3-0.6B model")
        print("  2. Try: python python/examples/simple_usage.py")
        print("  3. Read: docs/PYTHON_API.md for full documentation")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("  See docs/PYTHON_INSTALLATION.md for troubleshooting")
        sys.exit(1)
    
    print_system_info()


if __name__ == "__main__":
    main()

