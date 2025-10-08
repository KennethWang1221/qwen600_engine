// test_initialization.cu
//
// Verification test for Phase 0: Initialization
// This tests that build_transformer() works correctly before implementing kernels.
//
// What this tests:
// 1. Weight loading from SafeTensors file
// 2. GPU memory allocation (RunState)
// 3. cuBLAS initialization
// 4. Memory cleanup
//
// If this passes, you can proceed to implement kernels!

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "config.h"
#include "qwen_model.cuh"

// ANSI color codes for output
#define COLOR_GREEN   "\033[32m"
#define COLOR_RED     "\033[31m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_RESET   "\033[0m"

// Test result tracking
int tests_passed = 0;
int tests_failed = 0;

#define TEST_ASSERT(condition, test_name) \
    do { \
        if (condition) { \
            printf(COLOR_GREEN "  ✓ PASS: %s\n" COLOR_RESET, test_name); \
            tests_passed++; \
        } else { \
            printf(COLOR_RED "  ✗ FAIL: %s\n" COLOR_RESET, test_name); \
            tests_failed++; \
        } \
    } while(0)

// Helper: Check CUDA memory usage
void print_gpu_memory_usage() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    
    size_t used_bytes = total_bytes - free_bytes;
    double used_mb = used_bytes / (1024.0 * 1024.0);
    double total_mb = total_bytes / (1024.0 * 1024.0);
    
    printf("    GPU Memory: %.0f MB / %.0f MB used\n", used_mb, total_mb);
}

// Test 1: Basic CUDA functionality
bool test_cuda_available() {
    printf("\n" COLOR_CYAN "Test 1: CUDA Device Available" COLOR_RESET "\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        printf(COLOR_RED "  No CUDA devices found!\n" COLOR_RESET);
        return false;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("  Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Memory: %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    
    TEST_ASSERT(device_count > 0, "CUDA device available");
    return true;
}

// Test 2: Weight loading
bool test_weight_loading(const char* model_path) {
    printf("\n" COLOR_CYAN "Test 2: Weight Loading" COLOR_RESET "\n");
    
    // Check if file exists
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        printf(COLOR_RED "  Model file not found: %s\n" COLOR_RESET, model_path);
        TEST_ASSERT(false, "Model file exists");
        return false;
    }
    
    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);
    
    printf("  Model file size: %.2f MB\n", file_size / (1024.0 * 1024.0));
    TEST_ASSERT(file_size > 0, "Model file is readable");
    
    // Try to load weights
    try {
        size_t free_before, total;
        cudaMemGetInfo(&free_before, &total);
        
        qwen_loader::QwenWeights weights;
        qwen_loader::load_qwen_weights(model_path, weights);
        
        size_t free_after, _;
        cudaMemGetInfo(&free_after, &_);
        
        size_t allocated = free_before - free_after;
        printf("  GPU memory allocated: %.2f MB\n", allocated / (1024.0 * 1024.0));
        
        // Check that weights are not null
        bool weights_valid = (weights.token_embedding_table != nullptr) &&
                            (weights.output_head_weight != nullptr) &&
                            (weights.final_norm_weight != nullptr);
        
        TEST_ASSERT(weights_valid, "Weight pointers are valid");
        TEST_ASSERT(allocated > 1000 * 1024 * 1024, "Allocated ~1.5GB for weights");
        
        return weights_valid;
        
    } catch (const std::exception& e) {
        printf(COLOR_RED "  Exception: %s\n" COLOR_RESET, e.what());
        TEST_ASSERT(false, "Weight loading succeeded");
        return false;
    }
}

// Test 3: RunState allocation
bool test_runstate_allocation() {
    printf("\n" COLOR_CYAN "Test 3: RunState Memory Allocation" COLOR_RESET "\n");
    
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);
    
    RunState state;
    malloc_run_state(&state);
    
    size_t free_after, _;
    cudaMemGetInfo(&free_after, &_);
    
    size_t allocated = free_before - free_after;
    printf("  GPU memory allocated: %.2f MB\n", allocated / (1024.0 * 1024.0));
    
    // Check expected allocation size (~900-950 MB)
    // Exact size depends on GPU memory alignment
    bool size_correct = (allocated > 850 * 1024 * 1024) && 
                       (allocated < 1000 * 1024 * 1024);
    
    TEST_ASSERT(state.x != nullptr, "Activation buffer allocated");
    TEST_ASSERT(state.key_cache != nullptr, "Key cache allocated");
    TEST_ASSERT(state.value_cache != nullptr, "Value cache allocated");
    TEST_ASSERT(size_correct, "Allocated ~940MB for RunState");
    
    // Clean up
    cudaFree(state.x);
    cudaFree(state.xb);
    cudaFree(state.xb2);
    cudaFree(state.hb);
    cudaFree(state.hb2);
    cudaFree(state.q);
    cudaFree(state.att);
    cudaFree(state.logits);
    cudaFree(state.key_cache);
    cudaFree(state.value_cache);
    cudaFree(state.d_logits_fp32);
    
    return size_correct;
}

// Test 4: Complete transformer initialization
bool test_build_transformer(const char* model_path) {
    printf("\n" COLOR_CYAN "Test 4: Complete Transformer Initialization" COLOR_RESET "\n");
    
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);
    print_gpu_memory_usage();
    
    Transformer transformer;
    
    printf("\n  Calling build_transformer()...\n\n");
    build_transformer(&transformer, model_path);
    
    printf("\n  After initialization:\n");
    size_t free_after, _;
    cudaMemGetInfo(&free_after, &_);
    print_gpu_memory_usage();
    
    size_t allocated = free_before - free_after;
    printf("  Total allocated: %.2f MB\n", allocated / (1024.0 * 1024.0));
    
    // Verify all components
    bool weights_valid = (transformer.weights.token_embedding_table != nullptr);
    bool state_valid = (transformer.state.x != nullptr);
    bool cublas_valid = (transformer.cublas_handle != nullptr);
    bool host_valid = (transformer.h_logits != nullptr);
    
    TEST_ASSERT(weights_valid, "Weights loaded");
    TEST_ASSERT(state_valid, "RunState allocated");
    TEST_ASSERT(cublas_valid, "cuBLAS initialized");
    TEST_ASSERT(host_valid, "Host memory allocated");
    
    // Check total allocation (~2.4 GB)
    bool total_size_correct = (allocated > 2000 * 1024 * 1024) && 
                             (allocated < 3000 * 1024 * 1024);
    TEST_ASSERT(total_size_correct, "Total memory ~2.4GB");
    
    // Clean up
    printf("\n  Calling free_transformer()...\n");
    free_transformer(&transformer);
    
    printf("  After cleanup:\n");
    print_gpu_memory_usage();
    
    return weights_valid && state_valid && cublas_valid && host_valid;
}

// Test 5: Memory leak check
bool test_no_memory_leaks(const char* model_path) {
    printf("\n" COLOR_CYAN "Test 5: Memory Leak Check" COLOR_RESET "\n");
    
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);
    printf("  Before: ");
    print_gpu_memory_usage();
    
    // Allocate and free multiple times
    for (int i = 0; i < 3; i++) {
        Transformer transformer;
        build_transformer(&transformer, model_path);
        free_transformer(&transformer);
    }
    
    size_t free_after, _;
    cudaMemGetInfo(&free_after, &_);
    printf("  After 3x alloc/free: ");
    print_gpu_memory_usage();
    
    // Allow small difference due to CUDA driver overhead
    long long diff = (long long)free_after - (long long)free_before;
    bool no_leak = (abs(diff) < 10 * 1024 * 1024); // Allow 10MB difference
    
    printf("  Memory difference: %.2f MB\n", diff / (1024.0 * 1024.0));
    
    TEST_ASSERT(no_leak, "No significant memory leaks");
    
    return no_leak;
}

// Test 6: Basic pointer arithmetic (sanity check)
bool test_weight_pointers(const char* model_path) {
    printf("\n" COLOR_CYAN "Test 6: Weight Pointer Sanity Check" COLOR_RESET "\n");
    
    Transformer transformer;
    build_transformer(&transformer, model_path);
    
    // Check that layer pointers are in ascending order and non-null
    bool pointers_valid = true;
    for (int i = 0; i < N_LAYERS; i++) {
        auto& layer = transformer.weights.layers[i];
        
        if (layer.input_layernorm_weight == nullptr ||
            layer.attention.q_proj_weight == nullptr ||
            layer.ffn.gate_proj_weight == nullptr) {
            pointers_valid = false;
            printf("  Layer %d has null pointers!\n", i);
            break;
        }
    }
    
    TEST_ASSERT(pointers_valid, "All layer pointers are valid");
    
    // Check that embedding table is accessible
    bf16* emb = transformer.weights.token_embedding_table;
    TEST_ASSERT(emb != nullptr, "Embedding table accessible");
    
    free_transformer(&transformer);
    
    return pointers_valid;
}

// Main test runner
int main(int argc, char** argv) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║    QWEN INFERENCE ENGINE - INITIALIZATION VERIFICATION     ║\n");
    printf("║                     Phase 0 Checkpoint                     ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    
    // Get model path from arguments
    const char* model_path;
    if (argc < 2) {
        printf(COLOR_YELLOW "\nUsage: %s <path_to_model.safetensors>\n" COLOR_RESET, argv[0]);
        printf("Example: %s /path/to/model.safetensors\n", argv[0]);
        return 1;
    }
    model_path = argv[1];
    
    printf("\nModel: %s\n", model_path);
    
    // Run all tests
    test_cuda_available();
    test_weight_loading(model_path);
    test_runstate_allocation();
    test_build_transformer(model_path);
    test_no_memory_leaks(model_path);
    test_weight_pointers(model_path);
    
    // Print summary
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                       TEST SUMMARY                         ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    int total = tests_passed + tests_failed;
    printf("  Total Tests:  %d\n", total);
    printf("  " COLOR_GREEN "Passed:       %d" COLOR_RESET "\n", tests_passed);
    if (tests_failed > 0) {
        printf("  " COLOR_RED "Failed:       %d" COLOR_RESET "\n", tests_failed);
    }
    printf("\n");
    
    if (tests_failed == 0) {
        printf(COLOR_GREEN "╔════════════════════════════════════════════════════════════╗\n");
        printf("║          ✓ ALL TESTS PASSED! READY FOR PHASE 1!           ║\n");
        printf("║                                                            ║\n");
        printf("║  Your initialization is working correctly.                ║\n");
        printf("║  You can now proceed to implement CUDA kernels.           ║\n");
        printf("║                                                            ║\n");
        printf("║  Next step: Implement RMSNorm kernel                      ║\n");
        printf("║  See LEARNING_GUIDE.md for details                        ║\n");
        printf("╚════════════════════════════════════════════════════════════╝\n" COLOR_RESET);
        printf("\n");
        return 0;
    } else {
        printf(COLOR_RED "╔════════════════════════════════════════════════════════════╗\n");
        printf("║                    SOME TESTS FAILED                       ║\n");
        printf("║                                                            ║\n");
        printf("║  Please fix the issues before proceeding.                 ║\n");
        printf("║  Check the error messages above for details.              ║\n");
        printf("╚════════════════════════════════════════════════════════════╝\n" COLOR_RESET);
        printf("\n");
        return 1;
    }
}

