// static_loader.h
//
// This file implements loading model weights from SafeTensors format.
// SafeTensors is a safe, fast format for storing tensors.
//
// File Format:
// [8-byte header length (little-endian)][JSON metadata][binary tensor data]
//
// Learning objectives:
// 1. Memory-mapped file I/O (mmap)
// 2. CUDA memory management
// 3. Async GPU transfers
// 4. Pointer arithmetic and memory layout


#pragma once

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Memory mapping on Linux/macOS
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "config.h"

// ================================================================
// ERROR CHECKING MACRO
// ================================================================

/**
 * CUDA_CHECK: Macro for checking CUDA errors
 * 
 * This wraps every CUDA call and checks for errors.
 * If an error occurs, it prints the error and exits.
 * 
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace qwen_loader
{
    using bf16 = __nv_bfloat16;

    // ================================================================
    // WEIGHT STRUCTURES
    // ================================================================
    /**
     * AttentionWeights: Weight for one attention block
     * 
     * Qwen uses Grouped Query Attention (GQA):
     * - Q projection: DIM -> Q_DIM (1024 -> 2048)
     * - K projection: DIM -> KV_DIM (1024 -> 1024)
     * - V projection: DIM -> kv_DIM (1024 -> 1024)
     * - O projection: Q_DIM -> DIM (2048 -> 1024)
     * 
     * Additionally, Qwen uses QK-Norm (normalization on Q and K):
     * - q_norm_weight: (HEAD_DIM,) for each head
     * - k_norm_weight: (HEAD_DIM,) for each kv head
     * */

    struct AttentionWeights{
        bf16* q_proj_weight; // (Q_DIM, DIM) = (2048, 1024)
        bf16* k_proj_weight; // (KV_DIM, DIM) = (1024, 1024)
        bf16* v_proj_weight; // (KV_DIM, DIM) = (1024, 1024)
        bf16* o_proj_weight; // (DIM, Q_DIM) = (1024, 1024)
        bf16* q_norm_weight; // (HEAD_DIM,) = (128,)
        bf16* k_norm_weight; // (HEAD_DIM,) = (128,)
    };

    /**
     * FFNWeights: Feed-Forward Network weights
     * 
     * Qwen uses SwiGLU activation:
     * FFN(x) = (SiLU(gate(x)) * up(x)) @ down
     * 
     * Where:
     * - gate: DIM → HIDDEN_DIM
     * - up:   DIM → HIDDEN_DIM
     * - down: HIDDEN_DIM → DIM
    */

    struct FFNWeights{
        bf16* gate_proj_weight; // (HIDDEN_DIM, DIM) = (3072, 1024)
        bf16* up_proj_weight; // (HIDDEN_DIM, DIM) = (3072, 1024)
        bf16* down_proj_weight; // (DIM, HIDDEN_DIM) = (1024, 3072)
    };

    /**
     * TransormerBlcokWeights: Weights for one transformer layer
     * 
     * each layer contains:
     * - Pre-attention LayerNorm
     * - Attention block
     * - Post-attention LayerNorm
     * - FFN block
     */

    struct TransformerBlockWeights{
        bf16* input_layernorm_weight; // (DIM,) = (1024,)
        bf16* post_attention_layernorm_weight; // (DIM,) = (1024,)
        AttentionWeights attention;
        FFNWeights ffn;
    };

    /*
    QwenWeights: Complete model weights
    Total model size: ~/1.5GB for Qwen0.6B
    
    */

    struct QwenWeights{
        bf16* token_embedding_table; // (VOCAB_SIZE, DIM) = (151936, 1024)
        TransformerBlockWeights layers[N_LAYERS]; // 28 layers
        bf16* final_norm_weight; // (DIM,) = (1024,)
        bf16* output_head_weight; // (VOCAB_SIZE, DIM) = (151936, 1024)

        // we allocate one giant GPU memory block and point into it
        void* _gpu_mem_block = nullptr;

        // Destructor: automatically free GPU memory when QwenWeights goes out of scope
        ~QwenWeights()
        {
            if (_gpu_mem_block){
                CUDA_CHECK(cudaFree(_gpu_mem_block));
            }
        }
    };

    // Simple implementation for practice
    void load_qwen_weights(const char* filepath, QwenWeights& weights) {
        printf("Loading weights from: %s\n", filepath);
        
        // Allocate a small amount of GPU memory for testing
        size_t test_size = 1024 * 1024 * 1024;  // 1GB for testing
        CUDA_CHECK(cudaMalloc(&weights._gpu_mem_block, test_size));
        
        // Set up some dummy pointers for testing
        weights.token_embedding_table = (bf16*)weights._gpu_mem_block;
        weights.output_head_weight = (bf16*)((char*)weights._gpu_mem_block + test_size/4);
        weights.final_norm_weight = (bf16*)((char*)weights._gpu_mem_block + test_size/2);
        
        printf("✓ Basic weight loading completed (1GB allocated for testing)\n");
    }

} // namespace qwen_loader