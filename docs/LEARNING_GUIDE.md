# Qwen Inference Engine - Complete Learning Guide

**Building an LLM Inference Engine from Scratch**

This guide will take you step-by-step through implementing a complete GPU-accelerated transformer inference engine, learning CUDA, C++, and LLM internals along the way.

---

## Table of Contents

1. [Project Status](#project-status)
2. [Architecture Understanding](#architecture-understanding)
3. [Implementation Roadmap](#implementation-roadmap)
4. [Learning Path (Week-by-Week)](#learning-path)
5. [Development Workflow](#development-workflow)
6. [Resources](#resources)

---

## Project Status

### ✅ Completed Components

#### 1. Project Structure
- **CMakeLists.txt**: Complete build system with debug support
  - Debug mode: `-O0 -g -G` (no optimization, full debug symbols)
  - CUDA architecture support: 75 (Turing), 86 (Ampere)
  - cuBLAS integration
  - Proper error checking

#### 2. Configuration (`config.h`)
- Model hyperparameters for Qwen 0.6B
- Default sampling parameters
- Color codes for terminal output

#### 3. Weight Loader (`static_loader.h`) ✨
- **Built from scratch!** Complete SafeTensors parser
- Memory-mapped I/O for efficient file reading
- Async CUDA transfers using streams
- Single large GPU allocation (~1.5 GB)
- Full documentation of the loading process

#### 4. Model Core (`qwen_model.cuh`)
- Complete data structures
- `build_transformer()`: Initialization function
- `malloc_run_state()`: GPU memory allocation (~940 MB)
- `free_transformer()`: Cleanup and deallocation

#### 5. Main Application (`main.cu`)
- Command-line argument parsing
- Initialization flow
- User interface structure

**Current Status**: ✅ Project compiles successfully!

---

## Architecture Understanding

### The Transformer Structure

```cpp
typedef struct {
    TransformerWeights weights;    // Model parameters (loaded from file)
    RunState state;                // Runtime buffers (allocated dynamically)
    cublasHandle_t cublas_handle;  // cuBLAS context for matrix ops
    float* h_logits;               // Host-side buffer for final output
} Transformer;
```

**Key Concepts:**
- `weights`: Static parameters, loaded once, never modified
- `state`: Dynamic buffers that change during each forward pass
- `cublas_handle`: CUDA library context for efficient matrix operations
- `h_logits`: CPU-accessible memory for sampling

### The RunState Structure

Runtime buffers needed during inference:

```cpp
typedef struct {
    bf16 *x;       // Main activation buffer (DIM,) = (1024,)
    bf16 *xb;      // Residual branch buffer (DIM,)
    bf16 *xb2;     // Additional buffer (DIM,)
    bf16 *hb;      // FFN hidden dimension (HIDDEN_DIM,) = (3072,)
    bf16 *hb2;     // FFN hidden dimension (HIDDEN_DIM,)
    bf16 *q;       // Query buffer (Q_DIM,) = (2048,)
    
    float *att;    // Attention scores (N_HEADS * SEQ_LEN,)
    bf16 *logits;  // Output logits (VOCAB_SIZE,)
    
    // KV Cache for efficient autoregressive generation
    bf16* key_cache;   // (N_LAYERS, SEQ_LEN, KV_DIM)
    bf16* value_cache; // (N_LAYERS, SEQ_LEN, KV_DIM)
    
    float* d_logits_fp32; // FP32 logits on GPU
} RunState;
```

**Why these buffers?**
- **x, xb, xb2**: Layer normalization and residual connections
- **hb, hb2**: FFN needs larger hidden dimension (3x model dim)
- **q**: Query buffer (Q_DIM = N_HEADS * HEAD_DIM = 2048)
- **att**: Attention scores in FP32 for numerical stability
- **key_cache, value_cache**: Store past keys/values for efficient generation

### Model Dimensions (Qwen 0.6B)

```
DIM = 1024          # Model dimension
HIDDEN_DIM = 3072   # FFN hidden dimension (3x model dim)
N_LAYERS = 28       # Number of transformer blocks
N_HEADS = 16        # Number of attention heads
N_KV_HEADS = 8      # Number of KV heads (GQA)
HEAD_DIM = 128      # Dimension per head
SEQ_LEN = 8192      # Maximum sequence length
VOCAB_SIZE = 151936 # Vocabulary size

Q_DIM = N_HEADS * HEAD_DIM = 2048
KV_DIM = N_KV_HEADS * HEAD_DIM = 1024
```

### Memory Layout

#### GPU Memory Breakdown
```
Model Weights:     ~1,500 MB  (loaded from SafeTensors file)
KV Cache:            ~940 MB  (largest allocation)
Activation Buffers:   ~10 MB  (x, xb, hb, etc.)
Attention Buffers:   ~512 KB  (attention scores)
Output Buffers:      ~600 KB  (logits)
-----------------------------------
Total GPU Memory:  ~2,450 MB
```

#### CPU Memory
```
Pinned Host Memory: ~600 KB (for logits transfer)
Program Memory:     ~10 MB
```

### The build_transformer Function

This is the initialization function that sets up everything:

```cpp
void build_transformer(Transformer* t, const char* checkpoint_path)
{
    // Step 1: Load model weights from file to GPU (~1.5 GB)
    qwen_loader::load_qwen_weights(checkpoint_path, t->weights);
    
    // Step 2: Allocate runtime state buffers (~940 MB)
    malloc_run_state(&t->state);
    
    // Step 3: Allocate host-side pinned memory for logits
    cudaMallocHost((void**)&t->h_logits, VOCAB_SIZE * sizeof(float));
    
    // Step 4: Initialize cuBLAS for matrix operations
    cublasCreate(&t->cublas_handle);
}
```

**What it does:**
1. **Loads weights**: Uses memory-mapped I/O and async CUDA transfers
2. **Allocates runtime buffers**: All the temporary storage needed during inference
3. **Allocates pinned memory**: Faster GPU↔CPU transfers for sampling
4. **Initializes cuBLAS**: NVIDIA's optimized matrix multiplication library

---

## Implementation Roadmap

### Phase 1: Basic CUDA Kernels (Start Here!)

These are the fundamental operations for transformers.

#### 1.1 RMSNorm Kernel ⭐ **IMPLEMENT FIRST**

**Purpose**: Layer normalization (used before attention and FFN)

**Math**: `y = (x / √(mean(x²) + ε)) * weight`

**Learning Objectives**:
- Block-level parallel reduction
- Shared memory usage
- Warp-level primitives (CUB library)
- BF16 ↔ FP32 conversion
- Memory coalescing

**Implementation Template**:
```cuda
template <int THREADS_PER_BLOCK>
__global__ void rms_norm_kernel(
    bf16* output,
    const bf16* input,
    const bf16* weight,
    size_t dim)
{
    // Step 1: Calculate sum of squares (parallel reduction)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim/2; i += THREADS_PER_BLOCK) {
        // Load BF16 pairs, convert to FP32, accumulate
    }
    
    // Step 2: Reduce across block using CUB
    // BlockReduce::Sum(sum_sq)
    
    // Step 3: Compute normalization factor
    // rms = sqrt(sum_sq / dim + eps)
    
    // Step 4: Apply normalization and weight
    for (int i = threadIdx.x; i < dim/2; i += THREADS_PER_BLOCK) {
        // output[i] = (input[i] / rms) * weight[i]
    }
}
```

**Reference**: Lines 118-173 in reference `qwen_model.cuh`

#### 1.2 Element-wise Operations

**Operations to implement**:

1. **add_residual**: `x = x + residual`
   ```cuda
   __global__ void add_residual_kernel(bf16* x, const bf16* residual, int size)
   ```

2. **bf16_to_fp32**: Type conversion for logits
   ```cuda
   __global__ void convert_bf16_to_fp32_kernel(bf16* in, float* out, int n)
   ```

**Reference**: Lines 509-534, 628-636 in reference

#### 1.3 RoPE (Rotary Position Embedding)

**Purpose**: Encode position information into Q and K embeddings

**Math**: Rotate pairs of dimensions by position-dependent angles
```
freq_i = 1 / (theta ^ (i / dim))
angle = position * freq_i
q_real' = q_real * cos(angle) - q_imag * sin(angle)
q_imag' = q_real * sin(angle) + q_imag * cos(angle)
```

**Key Concepts**:
- Complex number rotation
- Frequency-based encoding
- Per-head operations
- Qwen uses a "naive" RoPE layout

**Reference**: Lines 316-377 in reference

---

### Phase 2: Attention Mechanism

The heart of transformers!

#### 2.1 Attention Components

```
Attention(Q, K, V) = softmax(QK^T / √d) V

Steps:
1. Compute scores: QK^T (dot products between queries and keys)
2. Scale and softmax: Normalize attention scores
3. Weighted sum: Multiply by values and aggregate
```

#### 2.2 Multi-Head Attention (MHA)

Qwen uses **Grouped Query Attention (GQA)**:
- 16 query heads
- 8 key/value heads
- Each query head groups with 2 KV heads

**Implementation Steps**:

1. **QK Kernel**: Compute attention scores
   ```cuda
   __global__ void attention_qk_kernel(
       float* att,           // Output: [N_HEADS, SEQ_LEN]
       const bf16* q,        // Query: [Q_DIM]
       const bf16* k_cache,  // Keys: [pos+1, KV_DIM]
       int pos)
   {
       // Each block handles one head
       // Each thread handles one position
       // Compute dot product: score = sum(q[h] * k[t,h])
   }
   ```

2. **Softmax Kernel**: Normalize scores
   ```cuda
   __global__ void softmax_kernel(float* att, int pos)
   {
       // Find max for numerical stability
       // Compute exp and sum
       // Normalize
   }
   ```

3. **Value Aggregation**: Weighted sum
   ```cuda
   __global__ void attention_v_kernel(
       bf16* out,           // Output: [Q_DIM]
       const float* att,    // Attention: [N_HEADS, SEQ_LEN]
       const bf16* v_cache, // Values: [pos+1, KV_DIM]
       int pos)
   {
       // Weighted sum: out[h,i] = sum(att[h,t] * v[t,h,i])
   }
   ```

**Reference**: Lines 417-504 in reference

#### 2.3 QK-Norm

Qwen applies RMSNorm to Q and K before attention:
```cuda
void qk_norm_fused_gpu(
    bf16* q,                    // [Q_DIM]
    bf16* k,                    // [KV_DIM]
    const bf16* q_norm_weight,  // [HEAD_DIM]
    const bf16* k_norm_weight)  // [HEAD_DIM]
{
    // Apply RMSNorm to each head independently
}
```

**Reference**: Lines 195-265 in reference

---

### Phase 3: Feed-Forward Network (FFN)

#### 3.1 SwiGLU Activation

**Math**: `SwiGLU(x) = SiLU(gate(x)) * up(x)`
where `SiLU(x) = x / (1 + exp(-x))`

```cuda
__global__ void swiglu_kernel(bf16* hb, const bf16* hb2, int size)
{
    // hb = gate projection output
    // hb2 = up projection output
    // hb = SiLU(hb) * hb2
}
```

**Reference**: Lines 539-565 in reference

#### 3.2 cuBLAS Matrix Multiplication

Wrap cuBLAS for efficient matrix-vector multiplication:

```cpp
void matmul_cublas(
    cublasHandle_t handle,
    bf16* y,              // Output: [m]
    const bf16* W,        // Weight: [m, n]
    const bf16* x,        // Input: [n]
    int m, int n,
    float alpha = 1.0f,
    float beta = 0.0f)    // beta=1.0 for fused residual
{
    // Call cublasGemmEx with:
    // - BF16 data type
    // - FP32 compute type
    // - Transpose for row-major layout
}
```

**Key Points**:
- Use `cublasGemmEx` for mixed precision
- BF16 data, FP32 accumulation
- `beta=1.0` enables fused residual addition
- Handle row-major to column-major conversion

**Reference**: Lines 572-623 in reference

---

### Phase 4: Complete Forward Pass

#### 4.1 Transformer Layer Structure

```python
# Pseudocode for one transformer layer

def transformer_layer(x, layer_weights, pos):
    # === ATTENTION BLOCK ===
    residual = x
    x = RMSNorm(x, layer.input_layernorm_weight)
    
    # QKV projections
    q = matmul(layer.attention.q_proj_weight, x)  # [Q_DIM]
    k = matmul(layer.attention.k_proj_weight, x)  # [KV_DIM]
    v = matmul(layer.attention.v_proj_weight, x)  # [KV_DIM]
    
    # Store K, V in cache
    k_cache[pos] = k
    v_cache[pos] = v
    
    # QK normalization
    q, k = QK_Norm(q, k, q_norm_weight, k_norm_weight)
    
    # Rotary position embedding
    q, k = RoPE(q, k, pos)
    
    # Multi-head attention
    attn_out = Attention(q, k_cache[:pos+1], v_cache[:pos+1])
    
    # Output projection + residual
    x = residual + matmul(layer.attention.o_proj_weight, attn_out)
    
    # === FFN BLOCK ===
    residual = x
    x = RMSNorm(x, layer.post_attention_layernorm_weight)
    
    # SwiGLU FFN
    gate = matmul(layer.ffn.gate_proj_weight, x)  # [HIDDEN_DIM]
    up = matmul(layer.ffn.up_proj_weight, x)      # [HIDDEN_DIM]
    hidden = SwiGLU(gate, up)
    
    # Down projection + residual
    x = residual + matmul(layer.ffn.down_proj_weight, hidden)
    
    return x
```

#### 4.2 Full Forward Pass

```cpp
float* forward(Transformer* transformer, int token, int pos)
{
    RunState* s = &transformer->state;
    TransformerWeights* w = &transformer->weights;
    cublasHandle_t handle = transformer->cublas_handle;
    
    // 1. Token embedding lookup
    bf16* token_emb = w->token_embedding_table + token * DIM;
    cudaMemcpy(s->x, token_emb, DIM * sizeof(bf16), cudaMemcpyDeviceToDevice);
    
    // 2. Process through all layers
    for (int l = 0; l < N_LAYERS; l++) {
        // Attention block
        // FFN block
    }
    
    // 3. Final layer norm
    rmsnorm_gpu(s->x, s->x, w->final_norm_weight, DIM);
    
    // 4. Output projection
    matmul_cublas(handle, s->logits, w->output_head_weight, s->x, 
                  VOCAB_SIZE, DIM);
    
    // 5. Convert to FP32 and copy to host
    convert_bf16_to_fp32(s->logits, s->d_logits_fp32, VOCAB_SIZE);
    cudaMemcpy(transformer->h_logits, s->d_logits_fp32, 
               VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    return transformer->h_logits;
}
```

**Reference**: Lines 638-717 in reference

---

## Learning Path

### Week 1-2: CUDA Fundamentals & RMSNorm

**Goal**: Understand GPU programming and implement first kernel

**Day 1-2**: CUDA Basics
- Thread hierarchy (blocks, grids, threads, warps)
- Memory hierarchy (global, shared, registers)
- Write simple vector addition
- Experiment with block/thread dimensions

**Day 3-5**: Implement RMSNorm
- Study parallel reduction patterns
- Learn CUB library (`BlockReduce`)
- Implement vectorized BF16 operations
- Benchmark different implementations

**Day 6-7**: Element-wise Operations
- Implement `add_residual`
- Implement `bf16_to_fp32` conversion
- Practice kernel launches
- Profile with `nsight-compute`

**Resources**:
- CUDA C Programming Guide (Chapters 2-3)
- CUB Documentation: BlockReduce
- Reference implementation: Lines 118-173

---

### Week 3-4: RoPE and Attention

**Goal**: Implement positional encoding and attention mechanism

**Day 1-3**: Implement RoPE
- Study rotary embeddings paper
- Understand complex rotation
- Implement position-dependent frequencies
- Visualize embeddings (optional)

**Day 4-7**: Implement Attention
- Understand QKV projections
- Implement QK kernel (dot products)
- Implement softmax kernel
- Implement value aggregation
- Add KV cache management
- Test with small examples

**Resources**:
- "RoFormer" paper (RoPE)
- "Attention Is All You Need" paper
- Reference implementation: Lines 316-504

---

### Week 5-6: FFN and cuBLAS

**Goal**: Complete transformer layer computation

**Day 1-3**: cuBLAS Integration
- Study cuBLAS documentation
- Understand `cublasGemmEx` API
- Implement matrix-vector wrapper
- Handle row-major to column-major
- Benchmark vs naive implementation

**Day 4-7**: Implement FFN
- Implement SwiGLU activation
- Connect FFN components
- Test complete layer forward pass
- Verify correctness

**Resources**:
- cuBLAS Documentation
- "GLU Variants Improve Transformer" paper
- Reference implementation: Lines 539-623

---

### Week 7-8: Integration & Optimization

**Goal**: Complete inference pipeline and optimize

**Day 1-4**: Complete Forward Pass
- Integrate all components
- Implement layer loop
- Add embedding lookup
- Test with real model weights
- Compare outputs with reference

**Day 5-7**: Optimization & Testing
- Profile with Nsight Systems
- Identify bottlenecks
- Optimize critical kernels
- Add error checking
- Write documentation

**Resources**:
- NVIDIA Nsight Tools Documentation
- CUDA Best Practices Guide

---

## Development Workflow

### 1. Build and Test

```bash
cd /PATH/TO/qwen600_engine

# Clean build
rm -rf build && mkdir build && cd build

# Configure (Debug mode for development)
cmake ..

# Compile
make

# Run
./qwen600_engine <path_to_model>
```

### 2. Debugging with cuda-gdb

```bash
# Compile with debug flags (already set in CMakeLists.txt)
cd build && make

# Run with cuda-gdb
cuda-gdb-python3.12-tui ./qwen600_engine

# Common commands:
(cuda-gdb) break rms_norm_kernel  # Set breakpoint
(cuda-gdb) run                     # Start execution
(cuda-gdb) cuda thread (0,0,0)    # Switch to specific thread
(cuda-gdb) print variable          # Print variable
(cuda-gdb) info cuda threads       # Show GPU threads
```

### 3. Profiling

```bash
# Memory checking
cuda-memcheck ./qwen600_engine

# Profile with Nsight Systems
nsys profile -o profile.qdrep ./qwen600_engine

# Profile with Nsight Compute (kernel-level)
ncu --set full -o kernel_profile ./qwen600_engine

# View results
nsight-sys profile.qdrep
```

### 4. Compare with Reference

```bash
# Your implementation
./qwen600_engine <model_path> -i "Hello"

# Reference implementation
/PATH/TO/qwen600/qwen600 <model_path> -i "Hello"

# Compare outputs
```

---

## Resources

### CUDA Programming

1. **Official Documentation**:
   - [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
   - [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
   - [CUB Library Documentation](https://nvidia.github.io/cccl/cub/)

2. **Tutorials**:
   - NVIDIA CUDA Training Series
   - "Programming Massively Parallel Processors" book

3. **Tools**:
   - [Nsight Systems](https://developer.nvidia.com/nsight-systems)
   - [Nsight Compute](https://developer.nvidia.com/nsight-compute)
   - cuda-gdb documentation

### Transformer Architecture

1. **Papers**:
   - "Attention Is All You Need" (original transformer)
   - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
   - "GLU Variants Improve Transformer"
   - "GQA: Training Generalized Multi-Query Transformer"

2. **Model Documentation**:
   - [Qwen2 Model Card](https://huggingface.co/Qwen)
   - Qwen technical report

### cuBLAS & Libraries

1. **Documentation**:
   - [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
   - [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)

2. **Examples**:
   - cuBLAS samples in CUDA Toolkit
   - Matrix multiplication optimization guides

---

## Common Pitfalls & Solutions

### 1. Integer Overflow
```cpp
// ❌ WRONG: May overflow on 32-bit int
cudaMalloc(&ptr, N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16));

// ✅ CORRECT: Use size_t for large allocations
cudaMalloc(&ptr, (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16));
```

### 2. Memory Leaks
```cpp
// Always pair allocations with deallocations
cudaMalloc(&ptr, size);
// ... use ptr ...
cudaFree(ptr);  // Don't forget!
```

### 3. Error Checking
```cpp
// ❌ WRONG: Ignoring errors
cudaMalloc(&ptr, size);

// ✅ CORRECT: Check every CUDA call
CUDA_CHECK(cudaMalloc(&ptr, size));
```

### 4. Synchronization
```cpp
// After launching kernels, synchronize before accessing results
kernel<<<blocks, threads>>>(args);
cudaDeviceSynchronize();  // Wait for kernel to finish
// Now safe to read results
```

### 5. Numerical Stability
```cpp
// Use FP32 for reductions and attention scores
// Use BF16 for weights and activations
// Convert carefully between types
```

---

## Testing Strategy

### Unit Tests

Test each component independently:

```cpp
// Test 1: Memory allocation
Transformer t;
build_transformer(&t, "model.safetensors");
// Check: nvidia-smi shows ~2.4GB allocated
free_transformer(&t);
// Check: memory released

// Test 2: RMSNorm kernel
// Create test input, run kernel, compare with CPU reference

// Test 3: Full layer
// Run one transformer layer, compare with reference
```

### Integration Tests

```cpp
// Test complete forward pass
// Compare your output with reference implementation
// Should match within numerical precision
```

### Benchmarking

```bash
# Measure tokens per second
time ./qwen600_engine -i "Write a story" | grep "tk/s"
```

---

## Summary

You now have:

✅ **Complete foundation**: Build system, weight loader, memory management  
✅ **Clear roadmap**: 8-week implementation plan  
✅ **Learning resources**: Papers, documentation, examples  
✅ **Development workflow**: Build, debug, profile, optimize  

**Your journey**:
1. Start with RMSNorm (Week 1-2)
2. Add RoPE and Attention (Week 3-4)
3. Implement FFN and cuBLAS (Week 5-6)
4. Complete and optimize (Week 7-8)


---

**Next Step**: Open `qwen_model.cuh` and implement `rms_norm_kernel`!
