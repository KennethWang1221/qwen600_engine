// cuda_utils.cuh
// =============================================================================
// LEARNING NOTE: Comprehensive CUDA utilities and error handling
// This file is a TOOLBOX of helpful functions for GPU programming
// Think of it as your "safety net" - it catches errors before they become bugs!
// =============================================================================

#pragma once  // Prevent multiple inclusion

#include <cuda_runtime.h>   // Core CUDA API (cudaMalloc, cudaMemcpy, etc.)
#include <cublas_v2.h>      // cuBLAS library for matrix operations
#include <stdio.h>          // For printf (error messages)
#include <stdlib.h>         // For exit()

// ================================================================
// Enhanced CUDA Error Checking
// ================================================================
// LEARNING NOTE: GPU errors are SILENT by default! Your code might fail
// and you won't know why. These macros make errors LOUD and clear.

// WHAT IS A MACRO?
// - Code that gets copy-pasted by the preprocessor before compilation
// - #define NAME(args) replacement_code
// - Useful for adding error checking without writing it every time

// WHY THE 'do { ... } while(0)' PATTERN?
// This is a MACRO TRICK that makes it safe to use in if-statements:
//   if (condition) CUDA_CHECK(cudaMalloc(...));  // Works correctly!
// Without do-while, the macro expansion could break

#define CUDA_CHECK(call) do { \
    /* STEP 1: Call the CUDA function and capture the return code */ \
    cudaError_t err = call; \
    \
    /* STEP 2: Check if it succeeded */ \
    if (err != cudaSuccess) { \
        /* STEP 3: Print detailed error information */ \
        fprintf(stderr, "\n" COLOR_BOLD_RED "CUDA Error:" COLOR_RESET " %s\n", \
                cudaGetErrorString(err));  /* Human-readable error message */ \
        fprintf(stderr, "  File: %s\n", __FILE__);        /* Which file? */ \
        fprintf(stderr, "  Line: %d\n", __LINE__);        /* Which line? */ \
        fprintf(stderr, "  Function: %s\n", #call);       /* What failed? */ \
        \
        /* STEP 4: Stop the program - can't continue with GPU errors! */ \
        exit(EXIT_FAILURE); \
    } \
} while (0)  /* Semicolon goes OUTSIDE when using the macro */

// LEARNING: The '#' operator in '#call' converts the argument to a string
// If you write: CUDA_CHECK(cudaMalloc(&ptr, 100))
// Then #call becomes: "cudaMalloc(&ptr, 100)" - very helpful for debugging!

// GOOD PRACTICE: Use CUDA_CHECK() for ALL CUDA runtime API calls:
//   CUDA_CHECK(cudaMalloc(&ptr, size));
//   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
//   CUDA_CHECK(cudaDeviceSynchronize());

// ---------------------- Kernel Error Checking ----------------------

#define CUDA_CHECK_KERNEL() do { \
    /* STEP 1: Check if kernel LAUNCH failed (wrong grid/block size, etc.) */ \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "\n" COLOR_BOLD_RED "CUDA Kernel Error:" COLOR_RESET " %s\n", \
                cudaGetErrorString(err)); \
        fprintf(stderr, "  File: %s\n", __FILE__); \
        fprintf(stderr, "  Line: %d\n", __LINE__); \
        exit(EXIT_FAILURE); \
    } \
    \
    /* STEP 2: Wait for kernel to finish and check if EXECUTION failed */ \
    err = cudaDeviceSynchronize();  /* Blocks until kernel completes */ \
    if (err != cudaSuccess) { \
        fprintf(stderr, "\n" COLOR_BOLD_RED "CUDA Kernel Execution Error:" COLOR_RESET " %s\n", \
                cudaGetErrorString(err)); \
        fprintf(stderr, "  File: %s\n", __FILE__); \
        fprintf(stderr, "  Line: %d\n", __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// LEARNING: Kernel launches are ASYNCHRONOUS - they return immediately!
// You must call cudaDeviceSynchronize() to wait and check for errors.
// This macro does both: checks launch error + waits + checks execution error

// USAGE:
//   my_kernel<<<blocks, threads>>>(args);
//   CUDA_CHECK_KERNEL();  // Put this right after kernel launch!

// ---------------------- cuBLAS Error Checking ----------------------

#define CUBLAS_CHECK(call) do { \
    /* cuBLAS returns cublasStatus_t instead of cudaError_t */ \
    cublasStatus_t status = call; \
    \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        /* NOTE: cuBLAS doesn't have string conversion like CUDA */ \
        fprintf(stderr, "\n" COLOR_BOLD_RED "cuBLAS Error:" COLOR_RESET " %d\n", status); \
        fprintf(stderr, "  File: %s\n", __FILE__); \
        fprintf(stderr, "  Line: %d\n", __LINE__); \
        fprintf(stderr, "  Function: %s\n", #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// SUGGESTION: Could add a switch statement to convert status codes to strings:
// switch(status) {
//     case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
//     case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
//     ...
// }

// ================================================================
// GPU Memory Information
// ================================================================
// LEARNING NOTE: Always check if you have enough GPU memory before allocating!

inline void print_gpu_memory_info() {
    // WHAT IS 'inline'?
    // - Suggests to compiler: "copy this function body at call site"
    // - Avoids function call overhead for small functions
    // - Header-only functions MUST be inline to avoid multiple definition errors
    
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // ^ This queries the GPU for memory statistics
    
    // Convert bytes to gigabytes for human readability
    // LEARNING: 1 GB = 1024 MB = 1024*1024 KB = 1024*1024*1024 bytes
    double free_gb = (double)free_bytes / (1024.0 * 1024.0 * 1024.0);
    double total_gb = (double)total_bytes / (1024.0 * 1024.0 * 1024.0);
    double used_gb = total_gb - free_gb;
    
    // Print with color formatting and 2 decimal places
    printf(COLOR_CYAN "GPU Memory:" COLOR_RESET " %.2f GB used / %.2f GB total (%.2f GB free)\n",
           used_gb, total_gb, free_gb);
}

inline bool check_gpu_memory_available(size_t required_bytes) {
    // GOOD PRACTICE: Validate before allocating!
    // Returns true if GPU has enough free memory
    
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    
    return free_bytes >= required_bytes;  // Simple comparison
}

// SUGGESTION: Add a function that prints a warning if memory is low:
// void warn_if_memory_low(size_t threshold_gb = 1) { ... }

// ================================================================
// GPU Device Information
// ================================================================
// LEARNING NOTE: Different GPUs have different capabilities!
// Always check what your GPU can do before running kernels.

inline void print_gpu_device_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));  // Which GPU are we using? (0, 1, 2...)
    
    cudaDeviceProp prop;  // Structure containing device properties
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // Print human-readable device information
    printf(COLOR_CYAN "GPU Device:" COLOR_RESET " %s\n", prop.name);
    // Example: "NVIDIA GeForce RTX 3090"
    
    printf(COLOR_CYAN "Compute Capability:" COLOR_RESET " %d.%d\n", 
           prop.major, prop.minor);
    // LEARNING: Compute capability determines GPU features
    // 7.5 = Turing, 8.0/8.6 = Ampere, 8.9 = Ada Lovelace, 9.0 = Hopper
    // Higher = newer = more features
    
    printf(COLOR_CYAN "Total Global Memory:" COLOR_RESET " %.2f GB\n", 
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    printf(COLOR_CYAN "Max Threads Per Block:" COLOR_RESET " %d\n", 
           prop.maxThreadsPerBlock);
    // IMPORTANT: Don't launch kernels with more threads than this!
    // Usually 1024 for modern GPUs
    
    printf(COLOR_CYAN "Warp Size:" COLOR_RESET " %d\n", prop.warpSize);
    // LEARNING: A "warp" is 32 threads that execute together (SIMT)
    // Always 32 on NVIDIA GPUs - design your algorithms around this!
}

// WHY PRINT THIS INFO?
// - Helps debug performance issues
// - Validates GPU is detected correctly
// - Useful for users to know what hardware they're using

// ================================================================
// Performance Timing Utilities
// ================================================================
// LEARNING NOTE: This section only compiles if ENABLE_PROFILING is defined
// Conditional compilation using #ifdef

#ifdef ENABLE_PROFILING
// This code only exists if you compile with -DENABLE_PROFILING flag

#include <chrono>   // C++11 timing library (std::chrono)
#include <map>      // Dictionary/hashtable for storing timings
#include <string>   // For section names

// WHAT IS A CLASS?
// - Bundles data (variables) and functions (methods) together
// - 'private:' means only this class can access these members
// - 'public:' means anyone can call these methods

class PerformanceTimer {
private:
    // LEARNING: std::map is a key-value store (like Python dict)
    // std::map<key_type, value_type> name;
    std::map<std::string, double> timings;   // section name -> total time
    std::map<std::string, int> counts;       // section name -> call count
    
    // C++ chrono time point - represents a specific moment in time
    std::chrono::high_resolution_clock::time_point start_time;
    
    std::string current_section;  // What are we currently timing?
    
public:
    void start(const std::string& section) {
        // LEARNING: 'const &' means: pass by reference (no copy), read-only
        // This is EFFICIENT - no string copying!
        
        current_section = section;
        start_time = std::chrono::high_resolution_clock::now();
        // ^ Captures current time with nanosecond precision
    }
    
    void end() {
        // Capture end time immediately - minimize measurement overhead
        auto end_time = std::chrono::high_resolution_clock::now();
        // LEARNING: 'auto' keyword: compiler figures out the type automatically
        
        // Calculate elapsed time in milliseconds
        double elapsed = std::chrono::duration<double, std::milli>(
            end_time - start_time
        ).count();
        
        // GOOD PRACTICE: Accumulate timings for multiple calls
        timings[current_section] += elapsed;  // Total time
        counts[current_section]++;             // Number of calls
        
        // LEARNING: std::map automatically creates entries if they don't exist!
        // First time: timings["forward"] = 0 + elapsed
    }
    
    void print_summary() {
        printf("\n" COLOR_CYAN "=== Performance Summary ===" COLOR_RESET "\n");
        
        // LEARNING: C++17 structured bindings - unpack key-value pairs
        // for (const auto& [key, value] : map)
        for (const auto& [section, total_time] : timings) {
            int count = counts[section];
            double avg_time = total_time / count;  // Average per call
            
            printf("  %s: %.2f ms total (%.2f ms avg, %d calls)\n",
                   section.c_str(),  // Convert std::string to C string
                   total_time, avg_time, count);
        }
    }
    
    void reset() {
        // Clear all accumulated timings - useful between runs
        timings.clear();
        counts.clear();
    }
};

// PATTERN: Meyer's Singleton - thread-safe lazy initialization
inline PerformanceTimer& get_global_timer() {
    // LEARNING: 'static' inside function = created once, persists forever
    // This creates a SINGLE global timer that everyone shares
    static PerformanceTimer timer;
    return timer;  // Return by reference (no copy)
}

// Convenience macros for easy profiling
#define PROFILE_START(name) get_global_timer().start(name)
#define PROFILE_END() get_global_timer().end()
#define PROFILE_SUMMARY() get_global_timer().print_summary()

// USAGE:
//   PROFILE_START("forward_pass");
//   ... your code ...
//   PROFILE_END();
//   PROFILE_SUMMARY();  // At end of program

#else
// If ENABLE_PROFILING is NOT defined, these macros do nothing
// BENEFIT: Zero overhead when profiling is disabled!
#define PROFILE_START(name)
#define PROFILE_END()
#define PROFILE_SUMMARY()
#endif

// ================================================================
// Warp-Level Primitives
// ================================================================
// LEARNING NOTE: These are DEVICE functions - they run ON THE GPU
// __device__ = callable only from GPU code
// __inline__ = strongly suggest inlining for performance

__inline__ __device__ float warpReduceSum(float val) {
    // WHAT IS A WARP?
    // - 32 threads that execute together (SIMT = Single Instruction Multiple Thread)
    // - All threads in a warp execute the same instruction at the same time
    // - This function sums a value across all 32 threads in the warp
    
    // LEARNING: Butterfly reduction pattern
    // Iteration 1: thread 0 gets value from thread 16 (offset=16)
    // Iteration 2: thread 0 gets value from thread 8  (offset=8)
    // Iteration 3: thread 0 gets value from thread 4  (offset=4)
    // ... continues until offset=1
    // Final result: thread 0 has sum of all 32 threads!
    
    #pragma unroll  // OPTIMIZATION: Tell compiler to unroll this loop
                    // Converts loop to straight-line code for performance
    for (int offset = 16; offset > 0; offset >>= 1) {  // >>= 1 means divide by 2
        // __shfl_down_sync: Get value from thread (threadIdx + offset)
        // 0xffffffff: All 32 threads participate (bitmask)
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // Thread 0 has the sum, others have partial sums
}

// PERFORMANCE BENEFIT: Warp operations are MUCH faster than shared memory
// or atomic operations. Use them when possible!

__inline__ __device__ float warpReduceMax(float val) {
    // Same pattern as warpReduceSum, but finds maximum instead
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        // fmaxf: float max function
    }
    return val;  // Thread 0 has the maximum
}

// SUGGESTION: Add warpReduceMin() for completeness

// ================================================================
// Memory Alignment Utilities
// ================================================================
// LEARNING NOTE: GPUs perform best with ALIGNED memory access
// Aligned = address is a multiple of some power of 2 (usually 256 bytes)

inline size_t align_to(size_t size, size_t alignment = 256) {
    // WHAT THIS DOES: Rounds 'size' UP to the next multiple of 'alignment'
    // Examples with alignment=256:
    //   100 -> 256
    //   256 -> 256
    //   300 -> 512
    
    // THE MATH:
    // 1. size + alignment - 1: Add (alignment - 1) to round up
    // 2. / alignment: Divide by alignment (rounds down)
    // 3. * alignment: Multiply back (gives next multiple)
    
    return ((size + alignment - 1) / alignment) * alignment;
    
    // EXAMPLE: align_to(100, 256)
    // = ((100 + 255) / 256) * 256
    // = (355 / 256) * 256
    // = 1 * 256 = 256
}

// WHY ALIGN MEMORY?
// - Faster GPU memory access (coalesced reads/writes)
// - Meets hardware requirements for some operations
// - Better cache utilization

template<typename T>
inline T* align_ptr(void* ptr, size_t alignment = 256) {
    // WHAT THIS DOES: Takes a pointer and returns an aligned version
    // LEARNING: reinterpret_cast converts between pointer types
    
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    // uintptr_t = integer large enough to hold a pointer
    
    // Apply same alignment math as align_to()
    uintptr_t aligned_addr = ((addr + alignment - 1) / alignment) * alignment;
    
    // Convert back to pointer of desired type
    return reinterpret_cast<T*>(aligned_addr);
}

// USAGE:
//   void* raw_ptr = malloc(1000);
//   float* aligned = align_ptr<float>(raw_ptr);
//   // Now aligned points to first 256-byte boundary after raw_ptr

// =============================================================================
// CODE REVIEW SUMMARY FOR JUNIOR DEVELOPERS:
// 
// ✅ GOOD PRACTICES USED HERE:
// 1. Comprehensive error checking (CUDA_CHECK macros)
// 2. Inline functions in headers (correct for header-only utilities)
// 3. Conditional compilation (#ifdef) for optional features
// 4. Clear function names (print_gpu_memory_info, warpReduceSum)
// 5. const references for strings (avoid copying)
// 6. Modern C++ features (auto, structured bindings, std::chrono)
// 
// 📚 KEY CONCEPTS YOU LEARNED:
// 1. Macros and do-while(0) pattern
// 2. Warp-level primitives for GPU optimization
// 3. Memory alignment for performance
// 4. Performance profiling with std::chrono
// 5. Error handling patterns for CUDA and cuBLAS
// 
// 💡 SUGGESTIONS FOR IMPROVEMENT:
// 1. Add string conversion for cuBLAS error codes
// 2. Consider adding cudaEventRecord for more precise GPU timing
// 3. Could add memory leak detection utilities
// 4. Consider adding wrappers for common patterns (RAII for cudaMalloc)
// =============================================================================
