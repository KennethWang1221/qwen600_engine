// 1. weight loading from safetensors file 
// 2. GPU memory allocation 
// 3. cuBLAS initialization 
// 4. Memory cleanup 


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "config.h"
#include "qwen_model_practice.cuh"

// ANSI color codes for output
#define COLOR_GREEN   "\033[32m"
#define COLOR_RED     "\033[31m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_RESET   "\033[0m"

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

// Test2: weight loading
bool test_weight_loading(const char* model_path){
    printf("\n" COLOR_CYAN "Test 2: Weight Loading" COLOR_RESET "\n");

    // check if file exists
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        printf(COLOR_RED " Model file not found: %s\n" COLOR_RESET, model_path);
        TEST_ASSERT(false, "Model file is exists");
        return false;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);

    printf(" Model file size: %.2f MB\n", file_size / (1024.0 * 1024.0));
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
        printf("GPU memory allocated: %.2f MB\n", allocated / (1024.0 * 1024.0));

        // check that weights are not null 
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


int main(int argc, char** argv){
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║    QWEN INFERENCE ENGINE - INITIALIZATION VERIFICATION     ║\n");
    printf("║                     Phase 0 Checkpoint                     ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    
    // get model path from arguments
    const char* model_path;
    if (argc < 2) {
        printf(COLOR_YELLOW "\nUsage: %s <path_to_model.safetensors>\n" COLOR_RESET, argv[0]);
        printf("Example: %s /path/to/model.safetensors\n", argv[0]);
        return 1;
    }
    model_path = argv[1];
    printf("\n Model: %s\n", model_path);
    // Run all tests
    test_cuda_available();
    test_weight_loading(model_path);
}