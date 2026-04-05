#pragma once

// ane_lm.h — Public C API for ANE-LM inference
//
// Usage (C or any FFI):
//
//   ane_lm_ctx ctx = ane_lm_load("/path/to/model");
//   ane_lm_generate(ctx, "Hello", 100, 0.6f, my_callback, my_data);
//   ane_lm_free(ctx);
//
// Swift example:
//   let ctx = ane_lm_load(modelPath)
//   ane_lm_generate(ctx, "Hello", 100, 0.6, { resp, _ in
//       print(String(cString: resp!.pointee.text!))
//   }, nil)
//   ane_lm_free(ctx)

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Opaque inference context (owns model + tokenizer)
typedef void* ane_lm_ctx;

// Per-token callback payload
typedef struct {
    const char* text;           // decoded text for this token (UTF-8, null-terminated)
    int         token;          // token ID (-1 = generation complete)
    int         prompt_tokens;
    double      prompt_tps;
    int         generation_tokens;
    double      generation_tps;
} ane_lm_response;

// Token callback: called once per decoded token, and once with token=-1 at end
typedef void (*ane_lm_callback)(const ane_lm_response* response, void* user_data);

// --- Lifecycle ---

// Load model from directory (reads config.json, safetensors, compiles ANE kernels).
// Returns NULL on failure. Thread-safe to load multiple contexts.
ane_lm_ctx ane_lm_load(const char* model_dir);

// Enable or disable persistent ANE compile cache (default: enabled).
// Call before ane_lm_load for effect.
void ane_lm_set_cache(int enabled);

// Enable verbose logging (default: off).
void ane_lm_set_verbose(int enabled);

// Free all resources associated with ctx. Safe to call with NULL.
void ane_lm_free(ane_lm_ctx ctx);

// --- Inference ---

// Single-prompt generation. Calls callback for each token.
// max_tokens=0 means unlimited. temperature=0 means greedy.
void ane_lm_generate(
    ane_lm_ctx ctx,
    const char* prompt,
    int max_tokens,
    float temperature,
    ane_lm_callback callback,
    void* user_data);

// Multi-turn chat generation.
// roles[i] and texts[i] form message pairs (n_messages total).
// Typical roles: "user", "assistant", "system".
void ane_lm_chat(
    ane_lm_ctx ctx,
    const char** roles,
    const char** texts,
    int n_messages,
    int max_tokens,
    float temperature,
    ane_lm_callback callback,
    void* user_data);

// --- Model info ---

int  ane_lm_vocab_size(ane_lm_ctx ctx);

// Reset KV cache (call between independent conversations)
void ane_lm_reset(ane_lm_ctx ctx);

#ifdef __cplusplus
}
#endif
