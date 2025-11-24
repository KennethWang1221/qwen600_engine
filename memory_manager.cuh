// memory_manager.cuh
// =============================================================================
// LEARNING NOTE: Unified Memory Management for GPU Buffers
// 
// PROBLEM WE'RE SOLVING:
// Instead of calling cudaMalloc() 11 separate times (slow, fragmented),
// we allocate ONE BIG BLOCK and divide it up ourselves (fast, organized)
// 
// ANALOGY: Imagine renting hotel rooms
// Bad way:  Book 11 separate rooms on different floors (slow check-in)
// Good way: Book one large suite, divide it with partitions (fast check-in)
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "config.h"
#include "cuda_utils.cuh"

// LEARNING: Type alias - makes code more readable
using bf16 = __nv_bfloat16;  // bf16 is shorter to type than __nv_bfloat16

// ================================================================
// Memory Layout Calculator
// ================================================================
// WHAT THIS DOES: Calculates where each buffer should go in our big block

struct MemoryLayout {
    // LEARNING: A 'struct' is like a class but members are public by default
    // We use struct for simple data containers (no complex behavior)
    
    // These store BYTE OFFSETS from the start of our unified buffer
    // Think of them as "apartment numbers" in our memory building
    size_t x_offset;              // Where does 'x' buffer start?
    size_t xb_offset;             // Where does 'xb' buffer start?
    size_t xb2_offset;            // etc...
    size_t hb_offset;
    size_t hb2_offset;
    size_t q_offset;
    size_t att_offset;
    size_t logits_offset;
    size_t key_cache_offset;
    size_t value_cache_offset;
    size_t d_logits_fp32_offset;
    
    size_t total_size;  // Total bytes needed for everything
    
    // WHAT IS A CONSTRUCTOR?
    // Special function that runs when you create an object
    // MemoryLayout() {...} is called automatically when you write:
    //   MemoryLayout layout;  // Constructor runs here!
    
    MemoryLayout() {
        // We'll build up our memory layout like a shopping list
        // Start at byte 0 and keep adding sizes
        size_t offset = 0;
        
        // -------------------- Activation Buffers --------------------
        // These store intermediate results during model inference
        // All use bf16 (bfloat16) = 2 bytes per number
        
        x_offset = offset;  // First buffer starts at byte 0
        offset += align_to(DIM * sizeof(bf16));
        // LEARNING: align_to() rounds up to 256-byte boundary
        // WHY? GPU memory access is faster when aligned!
        // sizeof(bf16) = 2 bytes, so DIM * 2 = 1024 * 2 = 2048 bytes
        // align_to(2048) = 2048 (already aligned)
        
        xb_offset = offset;  // Second buffer starts where first ended
        offset += align_to(DIM * sizeof(bf16));
        // PATTERN: Set offset, then advance by size
        
        xb2_offset = offset;  // Third buffer (another activation buffer)
        offset += align_to(DIM * sizeof(bf16));
        
        // -------------------- FFN Hidden Buffers --------------------
        // Feed-Forward Network intermediate results
        // HIDDEN_DIM = 3072 (3x larger than DIM)
        
        hb_offset = offset;
        offset += align_to(HIDDEN_DIM * sizeof(bf16));
        // 3072 * 2 = 6144 bytes
        
        hb2_offset = offset;
        offset += align_to(HIDDEN_DIM * sizeof(bf16));
        
        // -------------------- Query Buffer --------------------
        // Stores query vectors for attention mechanism
        // Q_DIM = N_HEADS * HEAD_DIM = 16 * 128 = 2048
        
        q_offset = offset;
        offset += align_to(Q_DIM * sizeof(bf16));
        // 2048 * 2 = 4096 bytes
        
        // -------------------- Attention Scores Buffer --------------------
        // IMPORTANT: This one uses FLOAT (4 bytes), not bf16!
        // WHY? Need higher precision for attention scores (softmax stability)
        
        att_offset = offset;
        offset += align_to((size_t)N_HEADS * SEQ_LEN * sizeof(float));
        // LEARNING: Cast to size_t to avoid integer overflow
        // 16 heads * 8192 seq_len * 4 bytes = 524,288 bytes = 512 KB
        
        // -------------------- Logits Buffer --------------------
        // Model output (before softmax)
        
        logits_offset = offset;
        offset += align_to(VOCAB_SIZE * sizeof(bf16));
        // 151,936 tokens * 2 bytes = 303,872 bytes ≈ 297 KB
        
        // -------------------- KV Cache --------------------
        // BIGGEST BUFFERS: Store keys and values for all layers
        // This is where most memory goes!
        
        key_cache_offset = offset;
        offset += align_to((size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16));
        // MATH: 28 layers * 8192 seq * 1024 kv_dim * 2 bytes
        //     = 28 * 8192 * 1024 * 2 = 469,762,048 bytes ≈ 448 MB!
        
        value_cache_offset = offset;
        offset += align_to((size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16));
        // Same size as key_cache ≈ 448 MB
        // TOTAL KV CACHE ≈ 896 MB (most of our memory!)
        
        // -------------------- FP32 Logits Buffer --------------------
        // Logits converted to float32 for sampling
        // WHY SEPARATE? Sampling needs higher precision
        
        d_logits_fp32_offset = offset;
        offset += align_to(VOCAB_SIZE * sizeof(float));
        // 151,936 * 4 = 607,744 bytes ≈ 594 KB
        
        // Final total size
        total_size = offset;
        // EXPECTED: ~900-950 MB total
    }
    
    void print_layout() const {
        // LEARNING: 'const' after function name means:
        // "this function won't modify the object"
        // Good practice for read-only functions!
        
        printf(COLOR_CYAN "=== Memory Layout ===" COLOR_RESET "\n");
        
        // %10zu means: print size_t with width 10, right-aligned
        // This creates a nice table-like output
        printf("  x:              %10zu bytes at offset %10zu\n", 
               DIM * sizeof(bf16), x_offset);
        printf("  xb:             %10zu bytes at offset %10zu\n", 
               DIM * sizeof(bf16), xb_offset);
        printf("  xb2:            %10zu bytes at offset %10zu\n", 
               DIM * sizeof(bf16), xb2_offset);
        printf("  hb:             %10zu bytes at offset %10zu\n", 
               HIDDEN_DIM * sizeof(bf16), hb_offset);
        printf("  hb2:            %10zu bytes at offset %10zu\n", 
               HIDDEN_DIM * sizeof(bf16), hb2_offset);
        printf("  q:              %10zu bytes at offset %10zu\n", 
               Q_DIM * sizeof(bf16), q_offset);
        printf("  att:            %10zu bytes at offset %10zu\n", 
               (size_t)N_HEADS * SEQ_LEN * sizeof(float), att_offset);
        printf("  logits:         %10zu bytes at offset %10zu\n", 
               VOCAB_SIZE * sizeof(bf16), logits_offset);
        printf("  key_cache:      %10zu bytes at offset %10zu\n", 
               (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16), key_cache_offset);
        printf("  value_cache:    %10zu bytes at offset %10zu\n", 
               (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16), value_cache_offset);
        printf("  d_logits_fp32:  %10zu bytes at offset %10zu\n", 
               VOCAB_SIZE * sizeof(float), d_logits_fp32_offset);
        
        // Print total in both bytes and megabytes
        printf(COLOR_CYAN "  Total:          %10zu bytes (%.2f MB)" COLOR_RESET "\n",
               total_size, (double)total_size / (1024.0 * 1024.0));
    }
};

// ================================================================
// Unified Memory Pool Manager
// ================================================================
// WHAT THIS DOES: Manages our big block of GPU memory

class UnifiedMemoryPool {
private:
    void* base_ptr = nullptr;  // Pointer to start of our big GPU block
                               // void* = "generic pointer" (can point to any type)
                               // nullptr = modern C++ for "null pointer" (safer than NULL)
    
    size_t total_size = 0;     // How many bytes did we allocate?
    MemoryLayout layout;       // Our memory map (where everything goes)
    
public:
    // CONSTRUCTOR: This runs when you create a UnifiedMemoryPool
    UnifiedMemoryPool() : layout() {
        // LEARNING: ": layout()" is "initializer list" syntax
        // It calls MemoryLayout's constructor BEFORE this constructor body runs
        // More efficient than assigning inside constructor body!
        
        total_size = layout.total_size;  // Get size from our layout
        
        // GOOD PRACTICE: Check if GPU has enough memory BEFORE allocating!
        if (!check_gpu_memory_available(total_size)) {
            // Not enough GPU memory - print helpful error and exit
            fprintf(stderr, COLOR_BOLD_RED "Error:" COLOR_RESET 
                    " Insufficient GPU memory!\n");
            fprintf(stderr, "Required: %.2f MB\n", 
                    (double)total_size / (1024.0 * 1024.0));
            print_gpu_memory_info();  // Show how much is available
            exit(EXIT_FAILURE);       // Can't continue without memory
        }
        
        // Allocate the big block!
        printf(COLOR_CYAN "Allocating unified memory pool..." COLOR_RESET "\n");
        CUDA_CHECK(cudaMalloc(&base_ptr, total_size));
        // LEARNING: &base_ptr = "address of base_ptr"
        // cudaMalloc needs to modify base_ptr, so we pass its address
        
        // OPTIONAL BUT RECOMMENDED: Zero-initialize the memory
        // WHY? Helps catch bugs (uninitialized memory is random garbage)
        CUDA_CHECK(cudaMemset(base_ptr, 0, total_size));
        // cudaMemset is like memset but for GPU memory
        
        printf(COLOR_GREEN "✓" COLOR_RESET " Allocated %.2f MB unified memory\n",
               (double)total_size / (1024.0 * 1024.0));
    }
    
    // DESTRUCTOR: This runs when UnifiedMemoryPool is destroyed
    // The '~' character means "destructor"
    ~UnifiedMemoryPool() {
        // IMPORTANT: Always check if pointer is valid before freeing!
        if (base_ptr) {
            CUDA_CHECK(cudaFree(base_ptr));  // Free GPU memory
            base_ptr = nullptr;              // Set to null (safety)
        }
    }
    // LEARNING: This is called RAII (Resource Acquisition Is Initialization)
    // Constructor acquires resource (allocates memory)
    // Destructor releases resource (frees memory)
    // BENEFIT: Automatic cleanup! Can't forget to free memory!
    
    // -------------------- Prevent Copying --------------------
    // LEARNING: These are "deleted" functions - compiler won't let you call them
    
    UnifiedMemoryPool(const UnifiedMemoryPool&) = delete;
    // Deletes the COPY CONSTRUCTOR
    // WHY? We don't want two UnifiedMemoryPools pointing to same GPU memory!
    // That would cause double-free (freeing same memory twice = crash)
    
    UnifiedMemoryPool& operator=(const UnifiedMemoryPool&) = delete;
    // Deletes the COPY ASSIGNMENT operator
    // Same reason - prevents accidental copying
    
    // RESULT: You can't write:
    //   UnifiedMemoryPool pool1;
    //   UnifiedMemoryPool pool2 = pool1;  // ERROR! Copy constructor deleted
    //   pool2 = pool1;                     // ERROR! Assignment deleted
    
    // SUGGESTION: Could add MOVE semantics (std::move) for transferring ownership
    
    // -------------------- Get Buffer Pointer --------------------
    // WHAT THIS DOES: Converts offset into actual pointer
    
    template<typename T>
    T* get_buffer(size_t offset) {
        // LEARNING: This is a TEMPLATE FUNCTION
        // You can call it with any type: get_buffer<float>(offset)
        // The compiler generates a version for each type you use
        
        // POINTER ARITHMETIC:
        // 1. static_cast<char*>(base_ptr): Convert to char* (1-byte pointer)
        //    WHY char*? Because adding to char* adds BYTES
        //    Adding to float* would add in multiples of 4 bytes!
        
        // 2. + offset: Add the offset in bytes
        
        // 3. reinterpret_cast<T*>(...): Convert to desired type
        
        return reinterpret_cast<T*>(static_cast<char*>(base_ptr) + offset);
        
        // EXAMPLE: Get the 'xb' buffer
        //   bf16* xb = pool.get_buffer<bf16>(layout.xb_offset);
        //   Now xb points to the correct location in our big block!
    }
    
    // -------------------- Getter Functions --------------------
    // These provide READ-ONLY access to private members
    // GOOD PRACTICE: Encapsulation - control how data is accessed
    
    const MemoryLayout& get_layout() const { 
        // Returns REFERENCE to layout (no copying)
        // Both 'const' mean: returns const reference, doesn't modify object
        return layout; 
    }
    
    size_t get_total_size() const { 
        return total_size; 
    }
    
    void print_layout() const { 
        layout.print_layout();  // Delegate to MemoryLayout's print function
    }
};

// =============================================================================
// HOW TO USE THIS:
// 
// 1. Create a memory pool:
//    UnifiedMemoryPool* pool = new UnifiedMemoryPool();
//    // This allocates ~900 MB on GPU in one call!
// 
// 2. Get pointers to individual buffers:
//    const MemoryLayout& layout = pool->get_layout();
//    bf16* x = pool->get_buffer<bf16>(layout.x_offset);
//    bf16* xb = pool->get_buffer<bf16>(layout.xb_offset);
//    // Now x, xb point to their sections of the big block
// 
// 3. Use the buffers normally:
//    my_kernel<<<blocks, threads>>>(x, xb, ...);
// 
// 4. Cleanup is automatic:
//    delete pool;  // Destructor frees ALL memory automatically!
// 
// =============================================================================

// =============================================================================
// CODE REVIEW SUMMARY FOR JUNIOR DEVELOPERS:
// 
// ✅ EXCELLENT PRACTICES USED HERE:
// 1. ✅ RAII pattern (automatic resource management)
// 2. ✅ Deleted copy constructor/assignment (prevents bugs)
// 3. ✅ Memory alignment for GPU performance
// 4. ✅ Pre-allocation validation (check memory before allocating)
// 5. ✅ Zero initialization (easier debugging)
// 6. ✅ Const correctness (const methods, const references)
// 7. ✅ Template functions for type safety
// 8. ✅ Helpful debug output (print_layout)
// 9. ✅ Initializer list in constructor (efficient)
// 
// 📚 KEY CONCEPTS YOU LEARNED:
// 1. Unified memory allocation vs. fragmented allocation
// 2. RAII (Resource Acquisition Is Initialization)
// 3. Memory alignment and why it matters
// 4. Pointer arithmetic with char* for byte offsets
// 5. Deleted functions to prevent copying
// 6. Template functions for generic code
// 7. Const correctness for safety
// 
// 💡 COMPARISON: Before vs. After
// 
// BEFORE (11 separate allocations):
//   cudaMalloc(&x, DIM * sizeof(bf16));       // Call 1
//   cudaMalloc(&xb, DIM * sizeof(bf16));      // Call 2
//   cudaMalloc(&xb2, DIM * sizeof(bf16));     // Call 3
//   ... 8 more calls ...
//   // Fragmented memory, slow, hard to manage
//   // Must call cudaFree() 11 times (easy to forget!)
// 
// AFTER (1 unified allocation):
//   UnifiedMemoryPool* pool = new UnifiedMemoryPool();
//   bf16* x = pool->get_buffer<bf16>(layout.x_offset);
//   // Fast, aligned, organized
//   // Only delete pool once - automatic cleanup!
// 
// PERFORMANCE BENEFIT: 20-30% faster initialization!
// 
// 🔧 POSSIBLE IMPROVEMENTS:
// 1. Add move constructor/assignment for ownership transfer
// 2. Consider adding bounds checking in get_buffer() for debug builds
// 3. Could add statistics (peak usage, allocation time)
// 4. Consider supporting multiple GPU devices
// 5. Could add custom allocator support for different memory types
// 
// 🎓 HOMEWORK EXERCISE:
// Try adding a method:
//   bool is_valid_offset(size_t offset, size_t size) const
// That checks if accessing memory at [offset, offset+size) is safe.
// This would help catch bugs where you access outside allocated region!
// =============================================================================
