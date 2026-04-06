#pragma once

#include "qwen3_5.h"
#include "../../core/ane_runtime.h"
#include "../../core/model_loader.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ane_lm {

struct Gemma3Args {
    int hidden_size = 1152;
    int num_hidden_layers = 18;
    int num_attention_heads = 4;
    int num_key_value_heads = 1;
    int head_dim = 256;
    int intermediate_size = 6912;
    int vocab_size = 262144;
    float rms_norm_eps = 1e-6f;
    float rope_local_base_freq = 10000.0f;
    float rope_global_base_freq = 1000000.0f;
    int sliding_window = 512;
    int sliding_window_pattern = 0;  // 0 = all global
    float query_pre_attn_scalar = 256.0f;
    bool tie_word_embeddings = true;

    static Gemma3Args from_json(const nlohmann::json& j);
};

class Gemma3Model : public LLMModel {
public:
    ~Gemma3Model() override;
    bool load(const std::string& model_dir) override;
    float* forward(int token_id, int pos) override;
    void reset() override;
    int vocab_size() const override { return vocab_size_; }

private:
    int hidden_size_ = 0;
    int intermediate_size_ = 0;
    int vocab_size_ = 0;
    int num_layers_ = 0;
    int num_q_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int sliding_window_ = 512;
    int sliding_window_pattern_ = 0;
    float rope_local_theta_ = 10000.0f;
    float rope_global_theta_ = 1000000.0f;
    float attn_scale_ = 0.0625f;  // 1/sqrt(query_pre_attn_scalar)
    float embed_scale_ = 1.0f;    // sqrt(hidden_size)
    float rms_eps_ = 1e-6f;
    bool tie_word_embeddings_ = true;
    bool ffn_is_fused_ = true;    // false when gelu fused compile fails

    int q_proj_dim_ = 0;
    int kv_proj_dim_ = 0;
    int rope_cache_len_ = 0;

    static constexpr int KV_CACHE_CAPACITY = 2048;

    // per-layer: is this layer global attention (vs sliding window)?
    std::vector<bool> is_global_attn_;

    struct LayerWeights {
        float* input_layernorm = nullptr;
        float* post_attention_layernorm = nullptr;
        float* pre_feedforward_layernorm = nullptr;
        float* post_feedforward_layernorm = nullptr;
        // Per-head Q/K norms (Gemma3 specific, shape=[head_dim])
        float* q_norm = nullptr;
        float* k_norm = nullptr;
        // BF16 projection weights for CPU fallback (point into mmap'd safetensors)
        const uint16_t* q_proj_w = nullptr;
        const uint16_t* k_proj_w = nullptr;
        const uint16_t* v_proj_w = nullptr;
        const uint16_t* o_proj_w = nullptr;
        const uint16_t* gate_proj_w = nullptr;
        const uint16_t* up_proj_w = nullptr;
        const uint16_t* down_proj_w = nullptr;
    };

    struct KVCache {
        float* k_cache = nullptr;
        float* v_cache = nullptr;
        int len = 0;
        int start = 0;
        int capacity = 0;
    };

    std::vector<LayerWeights> layers_;
    std::vector<KVCache> kv_caches_;
    std::vector<LayerANEKernels> ane_layers_;

    float* embed_tokens_ = nullptr;
    float* lm_head_ = nullptr;
    float* final_norm_ = nullptr;

    // Keep safetensors mmap alive so BF16 CPU weight pointers remain valid
    std::unique_ptr<ModelWeights> weights_;
    // True when num_layers * 2 <= ANE_LOAD_LIMIT so first_proj can run on ANE
    bool use_ane_first_proj_ = false;

    float* x_ = nullptr;
    float* x_norm_ = nullptr;
    float* logits_ = nullptr;
    float* scratch_qkv_ = nullptr;
    float* scratch_attn_ = nullptr;
    float* scratch_ffn_ = nullptr;  // gate+up output when split FFN

    // Separate RoPE tables for local (short theta) and global (long theta) layers
    float* rope_local_cos_ = nullptr;
    float* rope_local_sin_ = nullptr;
    float* rope_global_cos_ = nullptr;
    float* rope_global_sin_ = nullptr;

    void apply_args(const Gemma3Args& a);
    bool load_weights(ModelWeights* sf);
    bool compile_ane(ModelWeights* sf, const std::string& blob_dir);
    void build_rope_table(float* cos_out, float* sin_out, int len, float theta) const;
};

} // namespace ane_lm
