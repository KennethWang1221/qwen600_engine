// config.h
// =============================================================================
// LEARNING NOTE: This is a HEADER FILE - it contains configuration constants
// that are shared across the entire project. Think of it as a "settings file"
// that multiple source files can include and use.
// =============================================================================

#pragma once  // GOOD PRACTICE: Prevents this file from being included multiple times
              // This is modern C++ style. Older code uses #ifndef guards instead.

// ================================================================
// ANSI Color Codes for Terminal Output
// ================================================================
// LEARNING NOTE: These are escape sequences that make terminal text colorful.
// They're defined as macros (#define) so the preprocessor replaces them before compilation.
// Format: \x1b is ESC character, [XXm is the color code

#define COLOR_RESET    "\x1b[0m"        // Resets text to default color
#define COLOR_BOLD_RED "\x1b[1;31m"     // Bold red for errors
#define COLOR_GREEN    "\x1b[32m"       // Green for success messages
#define COLOR_YELLOW   "\x1b[33m"       // Yellow for warnings
#define COLOR_ORANGE   "\x1b[33m"       // Same as yellow (ANSI doesn't have orange)
#define COLOR_CYAN     "\033[36m"       // Cyan for info messages

// WHY USE #define FOR COLORS?
// - Zero runtime cost - replaced at compile time
// - Simple string replacement
// - No memory allocation needed

// ================================================================
// Model Configuration Constants
// ================================================================
// LEARNING NOTE: We use 'constexpr' instead of '#define' for numeric constants
// because it's type-safe and debugger-friendly.

// WHAT IS constexpr?
// - Means "constant expression" - evaluated at compile time
// - Has a type (int, float, etc.) unlike #define
// - Can be debugged (shows in debugger with correct type)
// - Catches type errors at compile time

#define MAX_LINE_WIDTH 80  // SUGGESTION: Could be constexpr int for type safety

// ==================== Sequence and Buffer Sizes ====================
constexpr int SEQ_LEN = 8192;           // Maximum sequence length (context window)
                                        // WHY 8192? Qwen3-0.6B model's maximum context
                                        
constexpr int PROMPT_BUFFER_SIZE = 32768;  // 4x SEQ_LEN for safety margin
                                           // LEARNING: Always allocate extra space for strings
                                           // to avoid buffer overflows
                                           
constexpr int VOCAB_SIZE = 151936;      // Total number of tokens the model understands
                                        // WHY SO LARGE? Covers multiple languages + special tokens

// ==================== Model Architecture ====================
// These define the "shape" of the neural network

constexpr int DIM = 1024;               // Hidden dimension - size of internal representation
                                        // ANALOGY: Like the "width" of the neural network
                                        
constexpr int HIDDEN_DIM = 3072;        // FFN (Feed-Forward Network) hidden size
                                        // TYPICAL PATTERN: 3-4x the main dimension
                                        // WHY? Gives the model more capacity to learn
                                        
constexpr int N_LAYERS = 28;            // Number of transformer blocks stacked
                                        // ANALOGY: Like layers in a cake - deeper = more processing
                                        
constexpr int N_HEADS = 16;             // Number of attention heads (for queries)
                                        // LEARNING: More heads = can attend to more patterns
                                        
constexpr int N_KV_HEADS = 8;           // Grouped Query Attention (GQA)
                                        // OPTIMIZATION: Fewer KV heads than Q heads saves memory
                                        // Each KV head is shared by N_HEADS/N_KV_HEADS Q heads
                                        
constexpr int HEAD_DIM = 128;           // Dimension per attention head
                                        // MATH: N_HEADS * HEAD_DIM should ≈ DIM

// ==================== Precomputed Values ====================
// GOOD PRACTICE: Precompute divisions and multiplications at compile time
// WHY? Division is expensive on GPU - doing it once at compile time is free!

constexpr float INV_HEAD_DIM = 1.0f / HEAD_DIM;  // 1/128 = 0.0078125
                                                 // Used in attention scaling
                                                 
constexpr float INV_DIM = 1.0f / DIM;           // 1/1024 = 0.0009765625
                                                // Used in normalization

// LEARNING NOTE: See how we add 'f' after float literals? (1.0f)
// This tells C++ it's a float, not a double. Important for GPU code!

// ==================== Model Hyperparameters ====================
constexpr float ROPE_THETA = 1000000.0f;  // RoPE (Rotary Position Embedding) base
                                          // WHY 1M? Extends context window capability
                                          // TECHNICAL: Controls position encoding frequency
                                          
constexpr float EPS = 1e-6f;             // Epsilon for numerical stability
                                         // LEARNING: Small value to prevent division by zero
                                         // Used in normalization: x / sqrt(sum + EPS)
                                         // WHY 1e-6? Small enough to not affect results,
                                         // large enough to prevent underflow

// ==================== Derived Dimensions ====================
// GOOD PRACTICE: Calculate these from base values instead of hardcoding
// WHY? If you change N_HEADS or HEAD_DIM, these update automatically!

constexpr int Q_DIM = N_HEADS * HEAD_DIM;        // 16 * 128 = 2048
                                                  // Query vector total dimension
                                                  
constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;    // 8 * 128 = 1024
                                                  // Key/Value vector total dimension
                                                  // NOTICE: Half of Q_DIM (memory optimization)

// =============================================================================
// ARCHITECTURE SUMMARY FOR BEGINNERS:
// 
// Think of the model as a 28-story building (N_LAYERS)
// Each floor has:
//   - 16 attention "cameras" (N_HEADS) looking at the input
//   - But only 8 memory banks (N_KV_HEADS) storing what they see
//   - Each camera sees 128 features (HEAD_DIM)
//   - A processing room with 3072 neurons (HIDDEN_DIM)
// 
// The model can remember 8192 tokens (SEQ_LEN) of context
// It knows 151,936 different words/tokens (VOCAB_SIZE)
// =============================================================================

// SUGGESTION: Consider adding these for better organization:
// constexpr int BYTES_PER_BF16 = 2;  // BF16 = 2 bytes
// constexpr size_t APPROX_MODEL_SIZE_BYTES = ...; // For memory validation
