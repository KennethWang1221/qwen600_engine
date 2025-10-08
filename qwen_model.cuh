// qwen_model.cuh
// 
// This file contains the core transformer model structures and functions
// for the Qwen inference engine.

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "config.h"
#include "static_loader.h"

// ================================================================
// GLOBALS AND TYPE DEFINITIONS
// ================================================================

#define EXIT_SUCCESS 0
constexpr int THREADS_PER_BLOCK = 256;

// Type aliases for clarity
using bf16 = __nv_bfloat16;
using TransformerWeights = qwen_loader::QwenWeights;

// ================================================================
// TRANSFORMER MODEL STRUCTURES
// ================================================================

/**
 * RunState: Runtime buffers for inference
 * 
 * This structure holds all the temporary buffers needed during a forward pass.
 * All pointers point to GPU memory.
 * 
 * Memory Layout:
 * - Activation buffers (x, xb, xb2): Used for layer norm and residual connections
 * - Hidden buffers (hb, hb2): Used for FFN intermediate activations
 * - Query buffer (q): Larger than x because Q_DIM = N_HEADS * HEAD_DIM
 * - Attention buffer (att): Stores attention scores in FP32 for stability
 * - KV cache: Stores past keys and values for efficient autoregressive generation
 * - Logits: Final output predictions
 */
typedef struct
{
    // === Activation Buffers ===
    bf16 *x;       // Main activation at current timestep (DIM,) = (1024,)
    bf16 *xb;      // Buffer for residual branch (DIM,)
    bf16 *xb2;     // Additional buffer for intermediate results (DIM,)
    
    // === FFN Buffers ===
    bf16 *hb;      // FFN hidden dimension buffer (HIDDEN_DIM,) = (3072,)
    bf16 *hb2;     // FFN hidden dimension buffer (HIDDEN_DIM,)
    
    // === Attention Buffers ===
    bf16 *q;       // Query buffer (Q_DIM,) = (2048,) - NOTE: Larger than DIM!
    float *att;    // Attention scores (N_HEADS, SEQ_LEN) - FP32 for numerical stability
    
    // === Output Buffers ===
    bf16 *logits;  // Output logits on GPU (VOCAB_SIZE,) in BF16
    float* d_logits_fp32;  // Output logits in FP32 on GPU
    
    // === KV Cache ===
    // These are the largest allocations (~940MB total)
    // Layout: [layer][position][kv_dim]
    bf16* key_cache;   // (N_LAYERS, SEQ_LEN, KV_DIM) = (28, 8192, 1024)
    bf16* value_cache; // (N_LAYERS, SEQ_LEN, KV_DIM) = (28, 8192, 1024)
} RunState;

/**
 * Transformer: Complete model structure
 * 
 * This is the main structure that holds everything needed for inference:
 * - weights: Model parameters loaded from checkpoint
 * - state: Runtime buffers allocated on GPU
 * - cublas_handle: cuBLAS context for matrix operations
 * - h_logits: Host-side pinned memory for fast GPU→CPU transfer
 */
typedef struct
{
    TransformerWeights weights;    // Model weights (loaded from file)
    RunState state;                // Runtime state (allocated dynamically)
    cublasHandle_t cublas_handle;  // cuBLAS context for matmul operations
    float* h_logits;               // Host-side buffer for final logits (pinned memory)
} Transformer;

// ================================================================
// MEMORY MANAGEMENT FUNCTIONS
// ================================================================

/**
 * malloc_run_state: Allocate GPU memory for runtime state
 * 
 * This function allocates all the temporary buffers needed during inference.
 * All allocations are on GPU device memory.
 * 
 * Total memory allocated: ~940 MB
 * - Small buffers (x, xb, q, etc.): ~1 MB
 * - KV cache: ~938 MB (the bulk of memory)
 * 
 * @param s: Pointer to RunState structure to initialize
 */
void
malloc_run_state(RunState* s)
{
    printf("Allocating GPU memory for runtime state...\n");
    
    // === Activation Buffers ===
    cudaMalloc(&s->x, DIM * sizeof(bf16));
    cudaMalloc(&s->xb, DIM * sizeof(bf16));
    cudaMalloc(&s->xb2, DIM * sizeof(bf16));
    
    // === FFN Buffers ===
    cudaMalloc(&s->hb, HIDDEN_DIM * sizeof(bf16));
    cudaMalloc(&s->hb2, HIDDEN_DIM * sizeof(bf16));
    
    // === Attention Buffers ===
    // Query buffer must be Q_DIM (N_HEADS * HEAD_DIM = 2048)
    cudaMalloc(&s->q, Q_DIM * sizeof(bf16));
    
    // Attention scores: N_HEADS * SEQ_LEN in FP32
    // Using size_t to avoid integer overflow
    cudaMalloc(&s->att, (size_t)N_HEADS * SEQ_LEN * sizeof(float));
    
    // === Output Buffers ===
    cudaMalloc(&s->logits, VOCAB_SIZE * sizeof(bf16));
    cudaMalloc(&s->d_logits_fp32, VOCAB_SIZE * sizeof(float));
    
    // === KV Cache ===
    // These are the largest allocations
    // IMPORTANT: Use size_t cast to avoid integer overflow!
    // Without cast: 28 * 8192 * 1024 * 2 might overflow 32-bit int
    cudaMalloc(&s->key_cache, 
               (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16));
    cudaMalloc(&s->value_cache, 
               (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16));
    
    printf("✓ GPU memory allocated successfully\n");
    printf("  - Activation buffers: ~%zu KB\n", 
           (DIM * 3 + HIDDEN_DIM * 2 + Q_DIM) * sizeof(bf16) / 1024);
    printf("  - Attention buffer: %zu KB\n", 
           (size_t)N_HEADS * SEQ_LEN * sizeof(float) / 1024);
    printf("  - KV cache: %zu MB\n", 
           (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16) * 2 / (1024 * 1024));
}

/**
 * build_transformer: Initialize the complete transformer model
 * 
 * This is the main initialization function that:
 * 1. Loads model weights from checkpoint file to GPU
 * 2. Allocates runtime state buffers on GPU
 * 3. Allocates host-side pinned memory for logits
 * 4. Initializes cuBLAS handle for matrix operations
 * 
 * @param t: Pointer to Transformer structure to initialize
 * @param checkpoint_path: Path to model.safetensors file
 */
void
build_transformer(
    Transformer* t,
    const char* checkpoint_path)
{
    printf("\n=== Initializing Transformer ===\n");
    printf("Checkpoint: %s\n", checkpoint_path);
    
    // Step 1: Load model weights from file to GPU
    // This will allocate ~1.5GB of GPU memory and load all weights
    printf("\nStep 1: Loading model weights...\n");
    qwen_loader::load_qwen_weights(checkpoint_path, t->weights);
    printf("✓ Model weights loaded to GPU\n");
    
    // Step 2: Allocate runtime state buffers
    // This will allocate ~940MB of GPU memory
    printf("\nStep 2: Allocating runtime state...\n");
    malloc_run_state(&t->state);
    
    // Step 3: Allocate host-side pinned memory for logits
    // Pinned memory enables faster GPU→CPU transfers
    // This is needed for sampling (which happens on CPU)
    printf("\nStep 3: Allocating host-side buffers...\n");
    cudaMallocHost((void**)&t->h_logits, VOCAB_SIZE * sizeof(float));
    printf("✓ Pinned host memory allocated: %zu KB\n", 
           VOCAB_SIZE * sizeof(float) / 1024);
    
    // Step 4: Initialize cuBLAS
    // cuBLAS provides highly optimized matrix operations
    // This is critical for performance as matmuls dominate inference time
    printf("\nStep 4: Initializing cuBLAS...\n");
    cublasCreate(&t->cublas_handle);
    printf("✓ cuBLAS initialized\n");
    
    printf("\n=== Transformer Initialized Successfully ===\n");
    printf("Total GPU memory used: ~2.4 GB\n");
    printf("Ready for inference!\n\n");
}

/**
 * free_transformer: Clean up and deallocate all resources
 * 
 * This function releases all memory and destroys handles.
 * IMPORTANT: Always call this before program exit to avoid memory leaks!
 * 
 * @param t: Pointer to Transformer structure to clean up
 */
void
free_transformer(Transformer* t)
{
    printf("\nCleaning up transformer resources...\n");
    
    // Free runtime state buffers
    cudaFree(t->state.x);
    cudaFree(t->state.xb);
    cudaFree(t->state.xb2);
    cudaFree(t->state.hb);
    cudaFree(t->state.hb2);
    cudaFree(t->state.q);
    cudaFree(t->state.att);
    cudaFree(t->state.logits);
    cudaFree(t->state.key_cache);
    cudaFree(t->state.value_cache);
    cudaFree(t->state.d_logits_fp32);
    
    // Free host-side pinned memory
    cudaFreeHost(t->h_logits);
    
    // Destroy cuBLAS handle
    cublasDestroy(t->cublas_handle);
    
    // Note: Weights are freed automatically by QwenWeights destructor
    
    printf("✓ All resources freed\n");
}

// ================================================================
// CUDA KERNELS (To be implemented)
// ================================================================

// TODO: Implement RMSNorm kernel
// TODO: Implement RoPE kernel
// TODO: Implement Attention kernels
// TODO: Implement SwiGLU kernel
// TODO: Implement residual add kernel
// TODO: Implement type conversion kernels

// ================================================================
// FORWARD PASS (To be implemented)
// ================================================================

// TODO: Implement forward() function
// This will be the main inference function that:
// 1. Looks up token embedding
// 2. Runs through all transformer layers
// 3. Applies final layer norm
// 4. Computes output logits
// 5. Returns logits for sampling

/**
 * forward: Run inference for one token (TO BE IMPLEMENTED)
 * 
 * @param transformer: Pointer to initialized Transformer
 * @param token: Input token ID
 * @param pos: Position in sequence
 * @return: Pointer to output logits (on host)
 */
float* forward(
    Transformer* transformer, 
    int token, 
    int pos)
{
    // TODO: Implement this function
    printf("forward() not yet implemented!\n");
    return nullptr;
}
