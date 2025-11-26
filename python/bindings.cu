// bindings.cpp
// =============================================================================
// Python bindings for Qwen3-0.6B CUDA inference engine using pybind11
// 
// This file bridges C++/CUDA code with Python, allowing users to call
// the inference engine from Python with a simple API.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For automatic conversion of std::map, std::vector
#include <string>
#include <map>
#include <vector>
#include <stdexcept>

// Include only header files (not main.cu which has main() function)
#include "../config.h"           // Must be first (defines constants)
#include "../qwen_model.cuh"
#include "../sampler.h"
#include "../tokenizer.h"
#include "../static_loader.h"

namespace py = pybind11;

// =============================================================================
// Utility Functions (from main.cu, needed here)
// =============================================================================

void construct_path(
    char* out_path, 
    size_t out_size, 
    const char* dir, 
    const char* filename)
{
    size_t len = strlen(dir);
    if (len > 0 && dir[len - 1] == '/')
    {
        // directory already has a slash, so don't add another one
        snprintf(out_path, out_size, "%s%s", dir, filename);
    }
    else
    {
        snprintf(out_path, out_size, "%s/%s", dir, filename);
    }
}

// =============================================================================
// C++ Wrapper Class for Python API
// =============================================================================
// This class wraps your C++ inference engine in a clean interface for Python

class InferenceSession {
private:
    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;
    std::string model_path;
    int device_id;
    int reasoning_mode;
    bool initialized = false;
    std::string system_prompt_str;
    
public:
    // Constructor: Initialize model
    InferenceSession(const std::string& model_dir, int device = 0, int reasoning = 0) {
        model_path = model_dir;
        device_id = device;
        reasoning_mode = reasoning;
        
        // Set CUDA device with error checking
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to set CUDA device ") + 
                std::to_string(device_id) + ": " + 
                cudaGetErrorString(err)
            );
        }
        
        // Construct paths
        char safetensors_path[1024];
        snprintf(safetensors_path, sizeof(safetensors_path), 
                 "%s/model.safetensors", model_dir.c_str());
        
        // Load model
        try {
            build_transformer(&transformer, safetensors_path);
            build_tokenizer(&tokenizer, model_dir.c_str(), reasoning_mode);
            
            // Default sampler config
            build_sampler(&sampler, 
                         0.6f,   // temperature
                         0.95f,  // top_p
                         0.05f,  // min_p
                         20,     // top_k
                         1.1f,   // repetition_penalty
                         0.0f,   // frequency_penalty
                         0.0f,   // presence_penalty
                         64,     // penalty_window
                         (unsigned long long)time(NULL));  // seed
            
            initialized = true;
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("Failed to initialize model: ") + e.what()
            );
        }
    }
    
    // Generate text from prompt
    std::string generate(
        const std::string& prompt,
        const std::map<std::string, float>& config,
        const std::string& system_prompt = ""
    ) {
        if (!initialized) {
            throw std::runtime_error("Session not initialized");
        }
        
        // Update sampler config
        update_sampler_config(config);
        
        // Prepare prompt with overflow checking
        char rendered_prompt[PROMPT_BUFFER_SIZE];
        int written = 0;
        if (!system_prompt.empty()) {
            written = snprintf(rendered_prompt, sizeof(rendered_prompt),
                    tokenizer.system_prompt_template,
                    system_prompt.c_str(), prompt.c_str());
        } else if (!system_prompt_str.empty()) {
            written = snprintf(rendered_prompt, sizeof(rendered_prompt),
                    tokenizer.system_prompt_template,
                    system_prompt_str.c_str(), prompt.c_str());
        } else {
            written = snprintf(rendered_prompt, sizeof(rendered_prompt),
                    tokenizer.prompt_template, prompt.c_str());
        }
        
        // Check for truncation
        if (written >= (int)sizeof(rendered_prompt)) {
            throw std::runtime_error(
                "Prompt too long (" + std::to_string(written) + 
                " bytes), maximum is " + std::to_string(sizeof(rendered_prompt)) + " bytes"
            );
        }
        
        // Encode prompt
        int* prompt_tokens = (int*)malloc(PROMPT_BUFFER_SIZE * sizeof(int));
        if (!prompt_tokens) {
            throw std::runtime_error("Failed to allocate memory for prompt tokens");
        }
        int num_tokens = 0;
        encode(&tokenizer, rendered_prompt, prompt_tokens, &num_tokens);
        
        // Generate tokens
        std::string output;
        int pos = 0;
        int max_tokens = static_cast<int>(config.count("max_tokens") ? 
                                         config.at("max_tokens") : 512);
        
        // First, process all prompt tokens through the transformer (prefill phase)
        int token = 0;
        for (int i = 0; i < num_tokens; i++) {
            token = prompt_tokens[i];
            forward(&transformer, token, pos++);  // Build KV cache from prompt
        }
        
        // Then, generate new tokens (generation phase)
        for (int i = 0; i < max_tokens; i++) {
            token = sample(&sampler, forward(&transformer, token, pos++));
            
            // Check for EOS
            if (token == tokenizer.eos_token_id) {
                break;
            }
            
            char* piece = decode(&tokenizer, token);
            output += piece;
        }
        
        free(prompt_tokens);
        return output;
    }
    
    // Set persistent system prompt
    void set_system_prompt(const std::string& prompt) {
        system_prompt_str = prompt;
    }
    
    // Reset conversation state
    void reset_state() {
        // Reset sampler history
        reset_sampler_history(&sampler);
        system_prompt_str.clear();
    }
    
    // Cleanup
    void cleanup() {
        if (initialized) {
            free_transformer(&transformer);
            free_tokenizer(&tokenizer);
            free_sampler(&sampler);
            initialized = false;
        }
    }
    
    // Destructor
    ~InferenceSession() {
        cleanup();
    }
    
private:
    void update_sampler_config(const std::map<std::string, float>& config) {
        // Update sampler parameters from Python config dict
        if (config.count("temperature")) {
            sampler.temperature = config.at("temperature");
        }
        if (config.count("top_k")) {
            sampler.top_k = static_cast<int>(config.at("top_k"));
        }
        if (config.count("top_p")) {
            sampler.topp = config.at("top_p");
        }
        if (config.count("min_p")) {
            sampler.minp = config.at("min_p");
        }
        if (config.count("repetition_penalty")) {
            sampler.repetition_penalty = config.at("repetition_penalty");
        }
        if (config.count("frequency_penalty")) {
            sampler.frequency_penalty = config.at("frequency_penalty");
        }
        if (config.count("presence_penalty")) {
            sampler.presence_penalty = config.at("presence_penalty");
        }
        if (config.count("seed") && config.at("seed") >= 0) {
            sampler.rng_state = static_cast<unsigned long long>(config.at("seed"));
        }
    }
};

// =============================================================================
// Pybind11 Module Definition
// =============================================================================
// This section exposes C++ classes and functions to Python

PYBIND11_MODULE(_qwen_core, m) {
    m.doc() = "Qwen3-0.6B CUDA inference engine - C++ backend";
    
    // Expose InferenceSession class
    py::class_<InferenceSession>(m, "InferenceSession")
        .def(py::init<const std::string&, int, int>(),
             py::arg("model_path"),
             py::arg("device") = 0,
             py::arg("reasoning_mode") = 0,
             "Initialize inference session\n\n"
             "Args:\n"
             "    model_path: Path to model directory\n"
             "    device: CUDA device ID (default: 0)\n"
             "    reasoning_mode: Enable thinking mode: 0=off, 1=on (default: 0)")
        
        .def("generate", &InferenceSession::generate,
             py::arg("prompt"),
             py::arg("config"),
             py::arg("system_prompt") = "",
             "Generate text from prompt\n\n"
             "Args:\n"
             "    prompt: Input text\n"
             "    config: Dictionary of sampling parameters\n"
             "    system_prompt: Optional system prompt\n"
             "Returns:\n"
             "    Generated text as string")
        
        .def("set_system_prompt", &InferenceSession::set_system_prompt,
             py::arg("prompt"),
             "Set persistent system prompt")
        
        .def("reset_state", &InferenceSession::reset_state,
             "Reset conversation state")
        
        .def("cleanup", &InferenceSession::cleanup,
             "Cleanup and free GPU memory");
    
    // Module version
    m.attr("__version__") = "2.0.0";
}

