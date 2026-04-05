// ane_lm_c.cpp — C API implementation for ANE-LM shared library
#include <ane_lm/ane_lm.h>
#include <ane_lm/common.h>
#include "../utils.h"
#include "../generate.h"
#include "../core/ane_runtime.h"

#include <memory>
#include <string>
#include <vector>
#include <cstring>

// ObjC autorelease pool (needed when called from non-ObjC code)
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

namespace ane_lm {

struct Context {
    std::unique_ptr<LLMModel> model;
    Tokenizer tokenizer;
};

} // namespace ane_lm

using namespace ane_lm;

// ============ Lifecycle ============

void ane_lm_set_cache(int enabled) {
    ane_set_persist_cache(enabled != 0);
}

void ane_lm_set_verbose(int enabled) {
    g_verbose = (enabled != 0);
}

ane_lm_ctx ane_lm_load(const char* model_dir) {
    if (!model_dir) return nullptr;
    void* pool = objc_autoreleasePoolPush();
    try {
        auto result = load(std::string(model_dir));
        auto* ctx = new Context();
        ctx->model     = std::move(result.first);
        ctx->tokenizer = std::move(result.second);
        objc_autoreleasePoolPop(pool);
        return ctx;
    } catch (const std::exception& e) {
        fprintf(stderr, "ane_lm_load: %s\n", e.what());
        objc_autoreleasePoolPop(pool);
        return nullptr;
    }
}

void ane_lm_free(ane_lm_ctx ctx) {
    if (!ctx) return;
    delete static_cast<Context*>(ctx);
}

// ============ Inference ============

void ane_lm_generate(
    ane_lm_ctx ctx,
    const char* prompt,
    int max_tokens,
    float temperature,
    ane_lm_callback callback,
    void* user_data)
{
    if (!ctx || !prompt) return;
    auto* c = static_cast<Context*>(ctx);

    void* pool = objc_autoreleasePoolPush();

    SamplingParams sampling;
    sampling.temperature = temperature;

    stream_generate(*c->model, c->tokenizer, std::string(prompt),
        max_tokens, false, sampling,
        [callback, user_data](const GenerationResponse& r) {
            if (!callback) return;
            ane_lm_response resp{};
            resp.text              = r.text.c_str();
            resp.token             = r.token;
            resp.prompt_tokens     = r.prompt_tokens;
            resp.prompt_tps        = r.prompt_tps;
            resp.generation_tokens = r.generation_tokens;
            resp.generation_tps    = r.generation_tps;
            callback(&resp, user_data);
        });

    objc_autoreleasePoolPop(pool);
}

void ane_lm_chat(
    ane_lm_ctx ctx,
    const char** roles,
    const char** texts,
    int n_messages,
    int max_tokens,
    float temperature,
    ane_lm_callback callback,
    void* user_data)
{
    if (!ctx || !roles || !texts || n_messages <= 0) return;
    auto* c = static_cast<Context*>(ctx);

    void* pool = objc_autoreleasePoolPush();

    std::vector<std::pair<std::string, std::string>> messages;
    messages.reserve(n_messages);
    for (int i = 0; i < n_messages; i++) {
        messages.push_back({
            roles[i] ? roles[i] : "user",
            texts[i] ? texts[i] : ""
        });
    }

    SamplingParams sampling;
    sampling.temperature = temperature;

    stream_generate(*c->model, c->tokenizer, messages,
        max_tokens, false, sampling,
        [callback, user_data](const GenerationResponse& r) {
            if (!callback) return;
            ane_lm_response resp{};
            resp.text              = r.text.c_str();
            resp.token             = r.token;
            resp.prompt_tokens     = r.prompt_tokens;
            resp.prompt_tps        = r.prompt_tps;
            resp.generation_tokens = r.generation_tokens;
            resp.generation_tps    = r.generation_tps;
            callback(&resp, user_data);
        });

    objc_autoreleasePoolPop(pool);
}

// ============ Model info ============

int ane_lm_vocab_size(ane_lm_ctx ctx) {
    if (!ctx) return 0;
    return static_cast<Context*>(ctx)->model->vocab_size();
}

void ane_lm_reset(ane_lm_ctx ctx) {
    if (!ctx) return;
    static_cast<Context*>(ctx)->model->reset();
}
