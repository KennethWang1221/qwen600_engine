#pragma once

// ================================================================
// Model Architecture Parameters
// ================================================================

// Basic model dimensions
constexpr int DIM = 1024;          // Model's base dimension
constexpr int HIDDEN_DIM = 3072;   // FFN hidden layer dimension
constexpr int N_LAYERS = 28;       // Number of transformer layers
constexpr int N_HEADS = 16;        // Number of attention heads
constexpr int N_KV_HEADS = 8;      // Number of key/value heads (for grouped-query attention)
constexpr int HEAD_DIM = 128;      // Dimension per attention head
constexpr int VOCAB_SIZE = 151936; // Size of the vocabulary

// Derived dimensions
constexpr int Q_DIM = N_HEADS * HEAD_DIM;     // Total query dimension
constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;  // Total key/value dimension

// Sequence and buffer sizes
constexpr int SEQ_LEN = 8192;              // Maximum sequence length
constexpr int PROMPT_BUFFER_SIZE = 32768;  // Size of prompt buffer
constexpr int MAX_LINE_WIDTH = 80;         // Maximum line width for output

// Mathematical constants
constexpr float INV_HEAD_DIM = 1.0f / HEAD_DIM;  // Precomputed inverse of head dimension
constexpr float INV_DIM = 1.0f / DIM;           // Precomputed inverse of model dimension
constexpr float ROPE_THETA = 1000000.0f;        // RoPE theta parameter
constexpr float EPS = 1e-6f;                    // Epsilon for numerical stability

// ================================================================
// CUDA Configuration
// ================================================================

// Thread and block configurations
constexpr int MAX_THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;

// Memory configurations
constexpr size_t MAX_GPU_MEMORY = 8ULL * 1024 * 1024 * 1024;  // 8GB default GPU memory limit

// ================================================================
// Output Formatting
// ================================================================

// ANSI color codes for pretty printing
#define COLOR_RESET    "\x1b[0m"
#define COLOR_BOLD_RED "\x1b[1;31m"
#define COLOR_GREEN    "\x1b[32m"
#define COLOR_YELLOW   "\x1b[33m"
#define COLOR_ORANGE   "\x1b[33m"
#define COLOR_CYAN     "\033[36m"

// ================================================================
// Sampling Parameters
// ================================================================

// Default sampling parameters
constexpr float DEFAULT_TEMPERATURE = 0.6f;
constexpr float DEFAULT_TOP_P = 0.95f;
constexpr int DEFAULT_TOP_K = 20;
constexpr float DEFAULT_REPETITION_PENALTY = 1.0f;
