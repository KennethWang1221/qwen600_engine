// sampler.h - Production-grade sampling with advanced techniques

#pragma once

#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

#include "config.h"

// ================================================================
// Data Structures
// ================================================================

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    // Basic parameters
    float temperature;
    float topp;          // top-p (nucleus) sampling
    float minp;          // min-p sampling
    int top_k;
    
    // Advanced: Repetition penalties
    float repetition_penalty;    // > 1.0 discourages repetition
    float frequency_penalty;     // penalize based on frequency
    float presence_penalty;      // penalize based on presence
    
    // Token history for penalties
    int* recent_tokens;
    int recent_token_count;
    int recent_token_capacity;
    int recent_token_pos;
    float* token_counts;
    
    // Internal
    unsigned long long rng_state;
    ProbIndex* probindex;
} Sampler;

// ================================================================
// Random Number Generation
// ================================================================

static inline unsigned int 
random_u32(unsigned long long *state) {
    *state ^= *state >> 12; 
    *state ^= *state << 25; 
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static inline float 
random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

// ================================================================
// Sorting and Selection
// ================================================================

static int 
compare_prob_desc(const void* a, const void* b) {
    ProbIndex* pa = (ProbIndex*)a;
    ProbIndex* pb = (ProbIndex*)b;
    if (pa->prob > pb->prob) return -1;
    if (pa->prob < pb->prob) return 1;
    return 0;
}

static void 
quick_select(ProbIndex* arr, int n, int k) {
    int l = 0, r = n - 1;
    while (l < r) {
        ProbIndex pivot = arr[k];
        int i = l, j = r;
        do {
            while (arr[i].prob > pivot.prob) i++;
            while (arr[j].prob < pivot.prob) j--;
            if (i <= j) {
                ProbIndex temp = arr[i]; 
                arr[i] = arr[j]; 
                arr[j] = temp;
                i++; j--;
            }
        } while (i <= j);
        if (j < k) l = i;
        if (i > k) r = j;
    }
}

// ================================================================
// Sampling Functions
// ================================================================

static inline int 
sample_argmax(float* logits) {
    int max_i = 0;
    float max_p = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (logits[i] > max_p) {
            max_i = i;
            max_p = logits[i];
        }
    }
    return max_i;
}

// Apply repetition penalties
static void
apply_repetition_penalties(Sampler* sampler, float* logits) {
    if (sampler->recent_token_count == 0) return;
    
    for (int i = 0; i < sampler->recent_token_count; i++) {
        int token = sampler->recent_tokens[i];
        if (token < 0 || token >= VOCAB_SIZE) continue;
        
        // Repetition penalty
        if (sampler->repetition_penalty != 1.0f) {
            if (logits[token] > 0) {
                logits[token] /= sampler->repetition_penalty;
            } else {
                logits[token] *= sampler->repetition_penalty;
            }
        }
        
        // Frequency penalty
        if (sampler->frequency_penalty != 0.0f) {
            logits[token] -= sampler->frequency_penalty * sampler->token_counts[token];
        }
        
        // Presence penalty
        if (sampler->presence_penalty != 0.0f) {
            logits[token] -= sampler->presence_penalty;
        }
    }
}

// Min-p filtering
static int
apply_minp_filter(ProbIndex* candidates, int n_candidates, float min_p) {
    if (min_p <= 0.0f || n_candidates == 0) return n_candidates;
    
    float max_prob = candidates[0].prob;
    for (int i = 1; i < n_candidates; i++) {
        if (candidates[i].prob > max_prob) max_prob = candidates[i].prob;
    }
    
    float threshold = min_p * max_prob;
    int kept = 0;
    for (int i = 0; i < n_candidates; i++) {
        if (candidates[i].prob >= threshold) {
            if (i != kept) candidates[kept] = candidates[i];
            kept++;
        }
    }
    return kept > 0 ? kept : 1;
}

// Main sampling function
static int 
sample(Sampler* sampler, float* logits) {
    // Declare all variables at the top to avoid goto issues
    int next_token;
    float temp_inv, max_logit, sum, sum_inv, r, cumsum;
    int n_cands, top_p_cutoff;
    int i;
    float exp_val;
    
    // Apply penalties
    apply_repetition_penalties(sampler, logits);
    
    // Greedy sampling
    if (sampler->temperature < 1e-6f) {
        next_token = sample_argmax(logits);
        goto record_token;
    }
    
    // Temperature scaling
    temp_inv = 1.0f / sampler->temperature;
    for (i = 0; i < VOCAB_SIZE; i++) {
        logits[i] *= temp_inv;
    }
    
    // Softmax
    max_logit = logits[0];
    for (i = 1; i < VOCAB_SIZE; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    sum = 0.0f;
    for (i = 0; i < VOCAB_SIZE; i++) {
        exp_val = expf(logits[i] - max_logit);
        sampler->probindex[i].prob = exp_val;
        sampler->probindex[i].index = i;
        sum += exp_val;
    }
    
    // Normalize
    sum_inv = 1.0f / sum;
    for (i = 0; i < VOCAB_SIZE; i++) {
        sampler->probindex[i].prob *= sum_inv;
    }
    
    n_cands = VOCAB_SIZE;
    
    // Top-k filtering
    if (sampler->top_k > 0 && sampler->top_k < n_cands) {
        quick_select(sampler->probindex, n_cands, sampler->top_k - 1);
        qsort(sampler->probindex, sampler->top_k, sizeof(ProbIndex), compare_prob_desc);
        n_cands = sampler->top_k;
    } else {
        qsort(sampler->probindex, n_cands, sizeof(ProbIndex), compare_prob_desc);
    }
    
    // Min-p filtering
    if (sampler->minp > 0.0f) {
        n_cands = apply_minp_filter(sampler->probindex, n_cands, sampler->minp);
    }
    
    // Top-p filtering
    if (sampler->topp < 1.0f && sampler->topp > 0.0f) {
        cumsum = 0.0f;
        top_p_cutoff = 0;
        for (i = 0; i < n_cands; i++) {
            cumsum += sampler->probindex[i].prob;
            if (cumsum >= sampler->topp) {
                top_p_cutoff = i + 1;
                break;
            }
        }
        if (top_p_cutoff > 0) n_cands = top_p_cutoff;
    }
    
    // Renormalize
    sum = 0.0f;
    for (i = 0; i < n_cands; i++) sum += sampler->probindex[i].prob;
    sum_inv = 1.0f / sum;
    for (i = 0; i < n_cands; i++) sampler->probindex[i].prob *= sum_inv;
    
    // Sample
    r = random_f32(&sampler->rng_state);
    cumsum = 0.0f;
    for (i = 0; i < n_cands; i++) {
        cumsum += sampler->probindex[i].prob;
        if (r < cumsum) {
            next_token = sampler->probindex[i].index;
            goto record_token;
        }
    }
    next_token = sampler->probindex[n_cands - 1].index;
    
record_token:
    // Update token history with proper NULL checks
    if (sampler->recent_tokens && sampler->token_counts && 
        next_token >= 0 && next_token < VOCAB_SIZE) {
        sampler->token_counts[next_token]++;
        sampler->recent_tokens[sampler->recent_token_pos] = next_token;
        sampler->recent_token_pos = (sampler->recent_token_pos + 1) % sampler->recent_token_capacity;
        if (sampler->recent_token_count < sampler->recent_token_capacity) {
            sampler->recent_token_count++;
        }
    }
    
    return next_token;
}

// ================================================================
// Initialization and Cleanup
// ================================================================

static inline void 
build_sampler(
    Sampler* sampler,
    float temperature,
    float topp,
    float minp,
    int top_k,
    float repetition_penalty,
    float frequency_penalty,
    float presence_penalty,
    int penalty_window_size,
    unsigned long long rng_seed)
{
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->minp = minp;
    sampler->top_k = (top_k >= 0 && top_k <= VOCAB_SIZE) ? top_k : VOCAB_SIZE;
    sampler->repetition_penalty = repetition_penalty;
    sampler->frequency_penalty = frequency_penalty;
    sampler->presence_penalty = presence_penalty;
    sampler->rng_state = rng_seed;
    sampler->probindex = (ProbIndex*)malloc(VOCAB_SIZE * sizeof(ProbIndex));
    if (!sampler->probindex) {
        fprintf(stderr, "Error: Failed to allocate memory for probindex\n");
        exit(EXIT_FAILURE);
    }
    
    if (penalty_window_size > 0) {
        sampler->recent_token_capacity = penalty_window_size;
        sampler->recent_tokens = (int*)malloc(penalty_window_size * sizeof(int));
        if (!sampler->recent_tokens) {
            fprintf(stderr, "Error: Failed to allocate memory for recent_tokens\n");
            free(sampler->probindex);
            exit(EXIT_FAILURE);
        }
        sampler->token_counts = (float*)calloc(VOCAB_SIZE, sizeof(float));
        if (!sampler->token_counts) {
            fprintf(stderr, "Error: Failed to allocate memory for token_counts\n");
            free(sampler->recent_tokens);
            free(sampler->probindex);
            exit(EXIT_FAILURE);
        }
        sampler->recent_token_count = 0;
        sampler->recent_token_pos = 0;
    } else {
        sampler->recent_tokens = NULL;
        sampler->token_counts = NULL;
        sampler->recent_token_count = 0;
        sampler->recent_token_capacity = 0;
        sampler->recent_token_pos = 0;
    }
}

static inline void 
free_sampler(Sampler* sampler) {
    if (sampler->probindex) {
        free(sampler->probindex);
        sampler->probindex = NULL;
    }
    if (sampler->recent_tokens) {
        free(sampler->recent_tokens);
        sampler->recent_tokens = NULL;
    }
    if (sampler->token_counts) {
        free(sampler->token_counts);
        sampler->token_counts = NULL;
    }
}

static inline void
reset_sampler_history(Sampler* sampler) {
    if (sampler->recent_tokens) {
        sampler->recent_token_count = 0;
        sampler->recent_token_pos = 0;
    }
    if (sampler->token_counts) {
        memset(sampler->token_counts, 0, VOCAB_SIZE * sizeof(float));
    }
}
