#include "gemma3.h"
#include "../../core/cpu_ops.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sys/stat.h>

namespace ane_lm {

using json = nlohmann::json;

// Half-split RoPE (same style as LLaMA/Gemma/Qwen): pairs (i, i+half)
static void apply_rope_gemma3(
    float* q, float* k,
    int n_q_heads, int n_kv_heads,
    int head_dim, int pos,
    const float* cos_row, const float* sin_row, float theta)
{
    int half = head_dim / 2;
    for (int h = 0; h < n_q_heads + n_kv_heads; h++) {
        float* v = (h < n_q_heads) ? q + (size_t)h * head_dim
                                   : k + (size_t)(h - n_q_heads) * head_dim;
        for (int i = 0; i < half; i++) {
            float cos_a, sin_a;
            if (cos_row && sin_row) {
                cos_a = cos_row[i];
                sin_a = sin_row[i];
            } else {
                float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
                cos_a = cosf(pos * freq);
                sin_a = sinf(pos * freq);
            }
            float v0 = v[i];
            float v1 = v[i + half];
            v[i]        = v0 * cos_a - v1 * sin_a;
            v[i + half] = v1 * cos_a + v0 * sin_a;
        }
    }
}

// ============ Config parsing ============

Gemma3Args Gemma3Args::from_json(const json& j) {
    Gemma3Args a;

    // Gemma3 may nest text config inside a "text_config" key (multimodal variants)
    const json& tc = j.contains("text_config") ? j["text_config"] : j;

    a.hidden_size           = tc.value("hidden_size",           a.hidden_size);
    a.num_hidden_layers     = tc.value("num_hidden_layers",     a.num_hidden_layers);
    a.num_attention_heads   = tc.value("num_attention_heads",   a.num_attention_heads);
    a.num_key_value_heads   = tc.value("num_key_value_heads",   a.num_key_value_heads);
    a.head_dim              = tc.value("head_dim",              a.head_dim);
    a.intermediate_size     = tc.value("intermediate_size",     a.intermediate_size);
    a.vocab_size            = tc.value("vocab_size",            a.vocab_size);
    a.rms_norm_eps          = tc.value("rms_norm_eps",          a.rms_norm_eps);
    a.sliding_window        = tc.value("sliding_window",        a.sliding_window);
    a.sliding_window_pattern = tc.value("sliding_window_pattern", a.sliding_window_pattern);
    a.query_pre_attn_scalar = tc.value("query_pre_attn_scalar", a.query_pre_attn_scalar);
    a.tie_word_embeddings   = tc.value("tie_word_embeddings",
                                        j.value("tie_word_embeddings", a.tie_word_embeddings));

    if (!tc.contains("head_dim") && a.num_attention_heads > 0)
        a.head_dim = a.hidden_size / a.num_attention_heads;

    // RoPE bases
    if (tc.contains("rope_local_base_freq"))
        a.rope_local_base_freq = tc["rope_local_base_freq"].get<float>();
    else if (tc.contains("rope_theta"))
        a.rope_local_base_freq = tc["rope_theta"].get<float>();

    if (tc.contains("rope_global_base_freq"))
        a.rope_global_base_freq = tc["rope_global_base_freq"].get<float>();

    return a;
}

// ============ Destructor ============

Gemma3Model::~Gemma3Model() {
    free(embed_tokens_);
    free(final_norm_);
    if (!tie_word_embeddings_) free(lm_head_);

    free(x_);
    free(x_norm_);
    free(logits_);
    free(scratch_qkv_);
    free(scratch_attn_);
    free(scratch_ffn_);
    free(rope_local_cos_);
    free(rope_local_sin_);
    free(rope_global_cos_);
    free(rope_global_sin_);

    for (auto& lw : layers_) {
        free(lw.input_layernorm);
        free(lw.post_attention_layernorm);
        free(lw.pre_feedforward_layernorm);
        free(lw.post_feedforward_layernorm);
        free(lw.q_norm);
        free(lw.k_norm);
    }
    for (auto& kv : kv_caches_) {
        free(kv.k_cache);
        free(kv.v_cache);
    }
    for (auto& lk : ane_layers_) ane_free_layer(&lk);

    for (auto* k : ffn_down_) ane_free(k);
    ffn_down_.clear();

    free_lm_head_ane();
}

void Gemma3Model::reset() {
    for (auto& kv : kv_caches_) {
        kv.len = 0;
        kv.start = 0;
        memset(kv.k_cache, 0, (size_t)kv.capacity * num_kv_heads_ * head_dim_ * sizeof(float));
        memset(kv.v_cache, 0, (size_t)kv.capacity * num_kv_heads_ * head_dim_ * sizeof(float));
    }
}

void Gemma3Model::apply_args(const Gemma3Args& a) {
    hidden_size_            = a.hidden_size;
    intermediate_size_      = a.intermediate_size;
    vocab_size_             = a.vocab_size;
    num_layers_             = a.num_hidden_layers;
    num_q_heads_            = a.num_attention_heads;
    num_kv_heads_           = a.num_key_value_heads;
    head_dim_               = a.head_dim;
    sliding_window_         = a.sliding_window;
    sliding_window_pattern_ = a.sliding_window_pattern;
    rope_local_theta_       = a.rope_local_base_freq;
    rope_global_theta_      = a.rope_global_base_freq;
    attn_scale_             = 1.0f / sqrtf(a.query_pre_attn_scalar);
    embed_scale_            = sqrtf((float)a.hidden_size);
    rms_eps_                = a.rms_norm_eps;
    tie_word_embeddings_    = a.tie_word_embeddings;

    q_proj_dim_ = num_q_heads_ * head_dim_;
    kv_proj_dim_ = num_kv_heads_ * head_dim_;
}

// ============ RoPE table builder ============

void Gemma3Model::build_rope_table(float* cos_out, float* sin_out, int len, float theta) const {
    int half = head_dim_ / 2;
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; i++)
        inv_freq[i] = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim_);
    for (int pos = 0; pos < len; pos++) {
        float* crow = cos_out + (size_t)pos * half;
        float* srow = sin_out + (size_t)pos * half;
        for (int i = 0; i < half; i++) {
            float angle = pos * inv_freq[i];
            crow[i] = cosf(angle);
            srow[i] = sinf(angle);
        }
    }
}

// ============ Weight loading ============

bool Gemma3Model::load_weights(ModelWeights* sf) {
    char name[256];

    embed_tokens_ = sf->load_bf16_to_f32("model.embed_tokens.weight",
                                          (int64_t)vocab_size_ * hidden_size_);
    if (!embed_tokens_) return false;

    if (tie_word_embeddings_) {
        lm_head_ = embed_tokens_;
    } else {
        lm_head_ = sf->load_bf16_to_f32("lm_head.weight",
                                          (int64_t)vocab_size_ * hidden_size_);
        if (!lm_head_) return false;
    }

    final_norm_ = sf->load_bf16_to_f32("model.norm.weight", hidden_size_);
    if (!final_norm_) return false;

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];

        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", L);
        lw.input_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        if (!lw.input_layernorm) return false;

        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", L);
        lw.post_attention_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        if (!lw.post_attention_layernorm) return false;

        snprintf(name, sizeof(name), "model.layers.%d.pre_feedforward_layernorm.weight", L);
        lw.pre_feedforward_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        if (!lw.pre_feedforward_layernorm) return false;

        snprintf(name, sizeof(name), "model.layers.%d.post_feedforward_layernorm.weight", L);
        lw.post_feedforward_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        if (!lw.post_feedforward_layernorm) return false;

        // Per-head Q/K norms (head_dim-sized, shared across all heads)
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", L);
        lw.q_norm = sf->load_bf16_to_f32(name, head_dim_);
        if (!lw.q_norm) return false;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", L);
        lw.k_norm = sf->load_bf16_to_f32(name, head_dim_);
        if (!lw.k_norm) return false;
    }

    LOG("All Gemma3 weights loaded successfully\n");
    return true;
}

// ============ ANE compilation ============

static std::string blob_path(const std::string& dir, const char* tensor_name) {
    std::string p = dir + "/";
    for (const char* c = tensor_name; *c; c++)
        p += (*c == '.') ? '/' : *c;
    p += ".bin";
    return p;
}

bool Gemma3Model::compile_ane(ModelWeights* sf, const std::string& blob_dir) {
    if (!ane_available()) {
        fprintf(stderr, "ANE not available\n");
        return false;
    }

    bool use_blobs = !blob_dir.empty();
    LOG("Compiling Gemma3 ANE kernels%s...\n", use_blobs ? " (from blobs)" : "");

    // Try fused GELU FFN kernel on the first layer to detect support
    ffn_is_fused_ = true;
    {
        const char* g0 = "model.layers.0.mlp.gate_proj.weight";
        const char* u0 = "model.layers.0.mlp.up_proj.weight";
        const char* d0 = "model.layers.0.mlp.down_proj.weight";
        ANEKernel* test = use_blobs
            ? ane_compile_fused_ffn_gelu_blob(
                blob_path(blob_dir, g0), blob_path(blob_dir, u0),
                blob_path(blob_dir, d0), hidden_size_, intermediate_size_)
            : ane_compile_fused_ffn_gelu(
                sf->get_bf16_ptr(g0), sf->get_bf16_ptr(u0),
                sf->get_bf16_ptr(d0), hidden_size_, intermediate_size_);
        if (!test) {
            LOG("  GELU fused FFN not supported, falling back to split gate+up/down\n");
            ffn_is_fused_ = false;
        } else {
            ane_free(test);
        }
    }

    if (!ffn_is_fused_) {
        scratch_ffn_ = (float*)calloc((size_t)2 * intermediate_size_, sizeof(float));
        ffn_down_.resize(num_layers_, nullptr);
    }

    char name[256], name2[256], name3[256];

    for (int L = 0; L < num_layers_; L++) {
        LOG("  Layer %d/%d...\r", L + 1, num_layers_);

        // QKV projection (fused)
        snprintf(name,  sizeof(name),  "model.layers.%d.self_attn.q_proj.weight", L);
        snprintf(name2, sizeof(name2), "model.layers.%d.self_attn.k_proj.weight", L);
        snprintf(name3, sizeof(name3), "model.layers.%d.self_attn.v_proj.weight", L);

        if (use_blobs) {
            ane_layers_[L].first_proj = ane_compile_fused_3_blob(
                blob_path(blob_dir, name),  q_proj_dim_,
                blob_path(blob_dir, name2), kv_proj_dim_,
                blob_path(blob_dir, name3), kv_proj_dim_,
                hidden_size_);
        } else {
            ane_layers_[L].first_proj = ane_compile_fused_3(
                sf->get_bf16_ptr(name),  q_proj_dim_,
                sf->get_bf16_ptr(name2), kv_proj_dim_,
                sf->get_bf16_ptr(name3), kv_proj_dim_,
                hidden_size_);
        }
        if (!ane_layers_[L].first_proj) {
            fprintf(stderr, "ANE first_proj failed at layer %d\n", L);
            return false;
        }

        // Output projection
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", L);
        if (use_blobs) {
            ane_layers_[L].o_proj = ane_compile_matmul_blob(
                blob_path(blob_dir, name), hidden_size_, q_proj_dim_);
        } else {
            ane_layers_[L].o_proj = ane_compile_matmul(
                sf->get_bf16_ptr(name), hidden_size_, q_proj_dim_);
        }
        if (!ane_layers_[L].o_proj) {
            fprintf(stderr, "ANE o_proj failed at layer %d\n", L);
            return false;
        }

        // FFN
        snprintf(name,  sizeof(name),  "model.layers.%d.mlp.gate_proj.weight", L);
        snprintf(name2, sizeof(name2), "model.layers.%d.mlp.up_proj.weight",   L);
        snprintf(name3, sizeof(name3), "model.layers.%d.mlp.down_proj.weight", L);

        if (ffn_is_fused_) {
            if (use_blobs) {
                ane_layers_[L].fused_ffn = ane_compile_fused_ffn_gelu_blob(
                    blob_path(blob_dir, name),
                    blob_path(blob_dir, name2),
                    blob_path(blob_dir, name3),
                    hidden_size_, intermediate_size_);
            } else {
                ane_layers_[L].fused_ffn = ane_compile_fused_ffn_gelu(
                    sf->get_bf16_ptr(name),
                    sf->get_bf16_ptr(name2),
                    sf->get_bf16_ptr(name3),
                    hidden_size_, intermediate_size_);
            }
            if (!ane_layers_[L].fused_ffn) {
                fprintf(stderr, "ANE fused_ffn_gelu failed at layer %d\n", L);
                return false;
            }
        } else {
            // Split: gate+up fused, down separate
            if (use_blobs) {
                ane_layers_[L].fused_ffn = ane_compile_fused_2_blob(
                    blob_path(blob_dir, name),  intermediate_size_,
                    blob_path(blob_dir, name2), intermediate_size_,
                    hidden_size_);
                ffn_down_[L] = ane_compile_matmul_blob(
                    blob_path(blob_dir, name3), hidden_size_, intermediate_size_);
            } else {
                ane_layers_[L].fused_ffn = ane_compile_fused_2(
                    sf->get_bf16_ptr(name),  intermediate_size_,
                    sf->get_bf16_ptr(name2), intermediate_size_,
                    hidden_size_);
                ffn_down_[L] = ane_compile_matmul(
                    sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_);
            }
            if (!ane_layers_[L].fused_ffn || !ffn_down_[L]) {
                fprintf(stderr, "ANE split FFN failed at layer %d\n", L);
                return false;
            }
        }
    }

    int compiled = ane_compile_count();
    int cached = ane_cache_loads();
    LOG("  %d ANE layer kernels ready (compiled=%d, cached=%d)\n",
        compiled + cached, compiled, cached);

    if (!compile_lm_head_ane(sf, blob_dir))
        LOG("ANE LM head disabled, falling back to CPU\n");
    else
        LOG("  LM head ANE enabled (%d chunks)\n", (int)lm_head_kernels_.size());

    return true;
}

bool Gemma3Model::compile_lm_head_ane(ModelWeights* sf, const std::string& blob_dir) {
    bool use_blobs = !blob_dir.empty();
    const char* lm_name = tie_word_embeddings_ ? "model.embed_tokens.weight" : "lm_head.weight";

    const uint16_t* lm_bf16 = sf->get_bf16_ptr(lm_name);
    if (!lm_bf16) return false;

    int chunk = lm_head_chunk_;
    if (chunk > vocab_size_) chunk = vocab_size_;
    int chunks = (vocab_size_ + chunk - 1) / chunk;
    lm_head_kernels_.resize(chunks, nullptr);

    LOG("  LM head ANE: %d chunks (chunk=%d)\n", chunks, chunk);
    for (int c = 0; c < chunks; c++) {
        int offset = c * chunk;
        int rows = std::min(chunk, vocab_size_ - offset);
        (void)use_blobs;
        const uint16_t* chunk_w = lm_bf16 + (int64_t)offset * hidden_size_;
        lm_head_kernels_[c] = ane_compile_matmul(chunk_w, rows, hidden_size_);
        if (!lm_head_kernels_[c]) {
            free_lm_head_ane();
            return false;
        }
    }

    ane_lm_head_enabled_ = true;
    lm_head_chunk_ = chunk;
    return true;
}

void Gemma3Model::free_lm_head_ane() {
    for (auto* k : lm_head_kernels_) ane_free(k);
    lm_head_kernels_.clear();
    ane_lm_head_enabled_ = false;
}

// ============ Model loading ============

bool Gemma3Model::load(const std::string& model_dir) {
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s\n", config_path.c_str());
        return false;
    }
    json j = json::parse(f);
    Gemma3Args args = Gemma3Args::from_json(j);
    apply_args(args);

    auto sf = ModelWeights::open(model_dir);
    if (!sf) {
        fprintf(stderr, "Failed to open model weights in %s\n", model_dir.c_str());
        return false;
    }

    // Infer dims from safetensors
    const SFTensor* embed = sf->find("model.embed_tokens.weight");
    if (!embed || embed->ndims != 2) {
        fprintf(stderr, "Missing/invalid model.embed_tokens.weight\n");
        return false;
    }
    const SFTensor* gate = sf->find("model.layers.0.mlp.gate_proj.weight");
    if (!gate || gate->ndims != 2) {
        fprintf(stderr, "Missing/invalid gate_proj.weight\n");
        return false;
    }

    hidden_size_      = (int)embed->shape[1];
    vocab_size_       = (int)embed->shape[0];
    intermediate_size_ = (int)gate->shape[0];
    if (head_dim_ <= 0 && num_q_heads_ > 0)
        head_dim_ = hidden_size_ / num_q_heads_;
    q_proj_dim_  = num_q_heads_  * head_dim_;
    kv_proj_dim_ = num_kv_heads_ * head_dim_;
    embed_scale_ = sqrtf((float)hidden_size_);

    LOG("Gemma3 dims: hidden=%d inter=%d vocab=%d layers=%d heads=%d/%d\n",
        hidden_size_, intermediate_size_, vocab_size_, num_layers_,
        num_q_heads_, num_kv_heads_);

    ane_init();

    // Allocate scratch buffers
    x_          = (float*)calloc(hidden_size_, sizeof(float));
    x_norm_     = (float*)calloc(hidden_size_, sizeof(float));
    logits_     = (float*)calloc(vocab_size_, sizeof(float));
    scratch_qkv_ = (float*)calloc((size_t)q_proj_dim_ + 2 * kv_proj_dim_, sizeof(float));
    scratch_attn_ = (float*)calloc(std::max(q_proj_dim_, hidden_size_), sizeof(float));

    // RoPE tables
    int half_rot = head_dim_ / 2;
    rope_cache_len_ = 8192;

    rope_local_cos_  = (float*)calloc((size_t)rope_cache_len_ * half_rot, sizeof(float));
    rope_local_sin_  = (float*)calloc((size_t)rope_cache_len_ * half_rot, sizeof(float));
    rope_global_cos_ = (float*)calloc((size_t)rope_cache_len_ * half_rot, sizeof(float));
    rope_global_sin_ = (float*)calloc((size_t)rope_cache_len_ * half_rot, sizeof(float));

    if (rope_local_cos_ && rope_local_sin_)
        build_rope_table(rope_local_cos_, rope_local_sin_, rope_cache_len_, rope_local_theta_);
    if (rope_global_cos_ && rope_global_sin_)
        build_rope_table(rope_global_cos_, rope_global_sin_, rope_cache_len_, rope_global_theta_);

    // Determine per-layer global/local attention
    is_global_attn_.resize(num_layers_);
    for (int L = 0; L < num_layers_; L++) {
        // Layer is global if (L+1) % sliding_window_pattern == 0, or pattern == 0
        bool global = (sliding_window_pattern_ == 0) ||
                      ((L + 1) % sliding_window_pattern_ == 0);
        is_global_attn_[L] = global;
    }

    layers_.resize(num_layers_);
    kv_caches_.resize(num_layers_);
    ane_layers_.resize(num_layers_);

    for (int L = 0; L < num_layers_; L++) {
        auto& kv = kv_caches_[L];
        kv.capacity = KV_CACHE_CAPACITY;
        kv.k_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
        kv.v_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
        kv.len = 0;
        kv.start = 0;
    }

    if (!load_weights(sf.get())) return false;

    std::string blob_dir = model_dir + "/ane_weights";
    struct stat st;
    bool has_blobs = (stat(blob_dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
    if (has_blobs)
        LOG("Using pre-converted ANE blobs from %s\n", blob_dir.c_str());

    if (!compile_ane(sf.get(), has_blobs ? blob_dir : "")) return false;

    return true;
}

// ============ Forward pass ============

float* Gemma3Model::forward(int token_id, int pos) {
    // Embed + scale
    memcpy(x_, embed_tokens_ + (int64_t)token_id * hidden_size_, hidden_size_ * sizeof(float));
    for (int i = 0; i < hidden_size_; i++) x_[i] *= embed_scale_;

    float* pre_oproj = scratch_attn_;

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];
        auto& cache = kv_caches_[L];
        bool global = is_global_attn_[L];

        // --- Attention ---
        rmsnorm_gemma(x_norm_, x_, lw.input_layernorm, hidden_size_, rms_eps_);

        // QKV projection
        float* qkv_buf = scratch_qkv_;
        if (!ane_matvec(ane_layers_[L].first_proj, qkv_buf, x_norm_,
                        hidden_size_, q_proj_dim_ + 2 * kv_proj_dim_)) {
            fprintf(stderr, "ANE first_proj failed at layer %d\n", L);
            return nullptr;
        }
        float* q_raw = qkv_buf;
        float* k_raw = qkv_buf + q_proj_dim_;
        float* v_raw = qkv_buf + q_proj_dim_ + kv_proj_dim_;

        // Per-head Q/K norms (Gemma3: same weight shared across heads)
        for (int h = 0; h < num_q_heads_; h++)
            rmsnorm_gemma(q_raw + h * head_dim_, q_raw + h * head_dim_,
                          lw.q_norm, head_dim_, rms_eps_);
        for (int h = 0; h < num_kv_heads_; h++)
            rmsnorm_gemma(k_raw + h * head_dim_, k_raw + h * head_dim_,
                          lw.k_norm, head_dim_, rms_eps_);

        // RoPE (use local or global table)
        const float* cos_row = nullptr;
        const float* sin_row = nullptr;
        if (pos < rope_cache_len_) {
            int half = head_dim_ / 2;
            if (global && rope_global_cos_) {
                cos_row = rope_global_cos_ + (size_t)pos * half;
                sin_row = rope_global_sin_ + (size_t)pos * half;
            } else if (!global && rope_local_cos_) {
                cos_row = rope_local_cos_ + (size_t)pos * half;
                sin_row = rope_local_sin_ + (size_t)pos * half;
            }
        }
        // Gemma3 uses half-split RoPE (same as LLaMA/Qwen3)
        apply_rope_gemma3(q_raw, k_raw, num_q_heads_, num_kv_heads_,
                          head_dim_, pos, cos_row, sin_row,
                          global ? rope_global_theta_ : rope_local_theta_);

        // KV cache update
        int slot;
        if (cache.len < cache.capacity) {
            slot = cache.start + cache.len;
            if (slot >= cache.capacity) slot -= cache.capacity;
            cache.len++;
        } else {
            slot = cache.start;
            cache.start++;
            if (cache.start >= cache.capacity) cache.start = 0;
        }
        size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
        memcpy(cache.k_cache + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
        memcpy(cache.v_cache + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));

        // Compute effective window for local attention
        int eff_len, eff_start;
        if (!global && sliding_window_ > 0 && cache.len > sliding_window_) {
            eff_len = sliding_window_;
            eff_start = (cache.start + (cache.len - sliding_window_)) % cache.capacity;
        } else {
            eff_len = cache.len;
            eff_start = cache.start;
        }

        // Scale Q if query_pre_attn_scalar differs from head_dim
        float default_scale = 1.0f / sqrtf((float)head_dim_);
        if (fabsf(attn_scale_ - default_scale) > 1e-6f) {
            float factor = attn_scale_ * sqrtf((float)head_dim_);
            for (int i = 0; i < q_proj_dim_; i++) q_raw[i] *= factor;
        }

        gqa_attention(pre_oproj, q_raw, cache.k_cache, cache.v_cache,
                      num_q_heads_, num_kv_heads_, head_dim_, head_dim_,
                      eff_start, eff_len, cache.capacity);

        // Output projection
        float* attn_out = x_norm_;
        if (!ane_matvec(ane_layers_[L].o_proj, attn_out, pre_oproj, q_proj_dim_, hidden_size_)) {
            fprintf(stderr, "ANE o_proj failed at layer %d\n", L);
            return nullptr;
        }

        // Post-attention norm + residual
        rmsnorm_gemma(attn_out, attn_out, lw.post_attention_layernorm, hidden_size_, rms_eps_);
        for (int i = 0; i < hidden_size_; i++) x_[i] += attn_out[i];

        // --- FFN ---
        rmsnorm_gemma(x_norm_, x_, lw.pre_feedforward_layernorm, hidden_size_, rms_eps_);

        float* ffn_out = scratch_attn_;

        if (ffn_is_fused_) {
            // Fused GELU: gate + gelu + up + down all in one ANE kernel
            if (!ane_matvec(ane_layers_[L].fused_ffn, ffn_out, x_norm_, hidden_size_, hidden_size_)) {
                fprintf(stderr, "ANE fused_ffn_gelu failed at layer %d\n", L);
                return nullptr;
            }
        } else {
            // Split: gate+up → CPU GELU → down
            float* gate_up = scratch_ffn_;
            if (!ane_matvec(ane_layers_[L].fused_ffn, gate_up, x_norm_,
                            hidden_size_, 2 * intermediate_size_)) {
                fprintf(stderr, "ANE gate+up failed at layer %d\n", L);
                return nullptr;
            }
            float* gate_out = gate_up;
            float* up_out   = gate_up + intermediate_size_;

            // GELU(gate) * up (in-place on gate_out)
            for (int i = 0; i < intermediate_size_; i++) {
                // GELU tanh approximation
                float g = gate_out[i];
                float g3 = g * g * g;
                float inner = 0.7978845608f * (g + 0.044715f * g3);
                float tanh_val = tanhf(inner);
                gate_out[i] = 0.5f * g * (1.0f + tanh_val) * up_out[i];
            }

            if (!ane_matvec(ffn_down_[L], ffn_out, gate_out, intermediate_size_, hidden_size_)) {
                fprintf(stderr, "ANE ffn_down failed at layer %d\n", L);
                return nullptr;
            }
        }

        // Post-FFN norm + residual
        rmsnorm_gemma(ffn_out, ffn_out, lw.post_feedforward_layernorm, hidden_size_, rms_eps_);
        for (int i = 0; i < hidden_size_; i++) x_[i] += ffn_out[i];
    }

    // Final norm (Gemma variant: multiply by 1 + weight)
    rmsnorm_gemma(x_, x_, final_norm_, hidden_size_, rms_eps_);

    // LM head
    if (ane_lm_head_enabled_ && !lm_head_kernels_.empty()) {
        bool ok = true;
        int chunks = (int)lm_head_kernels_.size();
        for (int c = 0; c < chunks && ok; c++) {
            int offset = c * lm_head_chunk_;
            int rows = std::min(lm_head_chunk_, vocab_size_ - offset);
            if (!ane_matvec(lm_head_kernels_[c], logits_ + offset, x_, hidden_size_, rows)) {
                ok = false;
            }
        }
        if (!ok) {
            free_lm_head_ane();
            matvec(logits_, lm_head_, x_, vocab_size_, hidden_size_);
        }
    } else {
        matvec(logits_, lm_head_, x_, vocab_size_, hidden_size_);
    }

    return logits_;
}

} // namespace ane_lm
