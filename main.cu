#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "qwen_model.cuh"

// Forward declarations for components not yet implemented
// class Tokenizer;  // TODO: Implement later
// class Sampler;    // TODO: Implement later

// ================================================================
// Utils
// ================================================================

void print_banner() {
    printf("\n" COLOR_ORANGE R"(
   ██████╗ ██╗    ██╗███████╗███╗   ██╗ ██████╗  ██████╗  ██████╗
  ██╔═══██╗██║    ██║██╔════╝████╗  ██║██╔════╝ ██╔═████╗██╔═████╗
  ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║███████╗ ██║██╔██║██║██╔██║
  ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║██╔═══██╗████╔╝██║████╔╝██║
  ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║╚██████╔╝╚██████╔╝╚██████╔╝
   ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝  ╚═════╝
)" COLOR_RESET);
    printf(COLOR_CYAN "                     QWEN Inference Engine\n" COLOR_RESET);
}

// Get current time in milliseconds for benchmarking
long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// Construct file path by combining directory and filename
void construct_path(char* out_path, size_t out_size, const char* dir, const char* filename) {
    size_t len = strlen(dir);
    if (len > 0 && dir[len - 1] == '/') {
        snprintf(out_path, out_size, "%s%s", dir, filename);
    } else {
        snprintf(out_path, out_size, "%s/%s", dir, filename);
    }
}

// ================================================================
// Error handling and usage
// ================================================================

void error_usage() {
    fprintf(stderr, "\nusage:   ./qwen600_engine <model_dir> [options]\n");
    fprintf(stderr, "example: ./qwen600_engine <model_dir> -r 1\n");
    fprintf(stderr, "model directory must contain:\n");
    fprintf(stderr, "  - model.safetensors\n");
    fprintf(stderr, "  - tokenizer.bin\n");
    fprintf(stderr, "  - template_*.txt files\n\n");

    fprintf(stderr, "arguments:\n");
    fprintf(stderr, "----------\n");
    fprintf(stderr, "  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking\n");
    fprintf(stderr, "  -s <int>    random seed, default = current time\n");
    fprintf(stderr, "  -k <int>    k value in top-k sampling, default %d\n", DEFAULT_TOP_K);
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default %.1f\n", DEFAULT_TEMPERATURE);
    fprintf(stderr, "  -p <float>  p value in top-p sampling in [0,1], default %.2f\n", DEFAULT_TOP_P);
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode, default is none\n\n");
    exit(EXIT_FAILURE);
}


// ================================================================
// chat loop
// ================================================================

void read_stdin(const char* guide, char* buffer, size_t bufsize){
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL){
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n'){
            // strip newline
            buffer[len - 1] = '\0';
        }

    }
}

// ================================================================
// Main
// ================================================================

int main(int argc, char *argv[]) {
    print_banner();

    // Default parameters
    char *model_dir = NULL;
    float temperature = DEFAULT_TEMPERATURE;
    float top_p = DEFAULT_TOP_P;
    int top_k = DEFAULT_TOP_K;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *system_prompt = NULL;
    int enable_thinking = 0;

    // Parse command line arguments
    if (argc >= 2) { 
        model_dir = argv[1]; 
    } else { 
        error_usage(); 
    }

    // Parse optional arguments
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }

        switch (argv[i][1]) {
            case 'h': error_usage(); break;
            case 't': temperature = atof(argv[i + 1]); break;
            case 'p': top_p = atof(argv[i + 1]); break;
            case 'k': top_k = atoi(argv[i + 1]); break;
            case 's': rng_seed = atoi(argv[i + 1]); break;
            case 'i': prompt = argv[i + 1]; break;
            case 'y': system_prompt = argv[i + 1]; break;
            case 'r': enable_thinking = atoi(argv[i + 1]); break;
            default: error_usage();
        }
    }

    // Validate and adjust parameters
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0) temperature = 0;
    if (top_p < 0 || top_p > 1.0) top_p = DEFAULT_TOP_P;

    // Construct model path
    char model_path[1024];
    char* model_tensor = "model.safetensors" // a pointer holding the address of the first character , m , when passed to the function, filename receives this address , and inside the function (via snprintf),C dereferences the pointer to read each character sequentially
    construct_path(&model_path[0], sizeof(model_path), model_dir, model_tensor);

    printf("\nInitializing with parameters:\n");
    printf("- Model path: %s\n", model_path);
    printf("- Temperature: %.2f\n", temperature);
    printf("- Top-p: %.2f\n", top_p);
    printf("- Top-k: %d\n", top_k);
    printf("- Thinking mode: %s\n", enable_thinking ? "enabled" : "disabled");
    printf("- Random seed: %llu\n", rng_seed);


    Transformer transformer; // Declares a variable named transformer , 'Transformer' is the TYPE, 'transformer' is the VARIABLE
    build_transformer(&transformer, model_path);
    
    // TODO: Initialize components once implemented
    
    // Tokenizer tokenizer;
    // Sampler sampler;

    printf("\nComponents will be implemented in the following order:\n");
    printf("1. Tokenizer\n");
    printf("2. Model Loader\n");
    printf("3. Transformer\n");
    printf("4. Sampler\n");

    return 0;
}
