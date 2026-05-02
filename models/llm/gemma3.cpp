#include "gemma3.h"
#include "../../core/cpu_ops.h"
#include "../../core/metal_ops.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sys/stat.h>

namespace ane_lm {

using json = nlohmann::json;

static void apply_rope_gemma3(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim, int pos, const float* cos_row, const float* sin_row, float theta) {
    int half = head_dim / 2;
    for (int h = 0; h < n_q_heads + n_kv_heads; h++) {
        float* v = (h < n_q_heads) ? q + (size_t)h * head_dim : k + (size_t)(h - n_q_heads) * head_dim;
        for (int i = 0; i < half; i++) {
            float cos_a = cos_row ? cos_row[i] : cosf(pos / powf(theta, (float)(2 * i) / head_dim));
            float sin_a = sin_row ? sin_row[i] : sinf(pos / powf(theta, (float)(2 * i) / head_dim));
            float v0 = v[i], v1 = v[i + half];
            v[i] = v0 * cos_a - v1 * sin_a; v[i + half] = v1 * cos_a + v0 * sin_a;
        }
    }
}

Gemma3Args Gemma3Args::from_json(const json& j) {
    Gemma3Args a; const json& tc = j.contains("text_config") ? j["text_config"] : j;
    a.hidden_size = tc.value("hidden_size", a.hidden_size); a.num_hidden_layers = tc.value("num_hidden_layers", a.num_hidden_layers);
    a.num_attention_heads = tc.value("num_attention_heads", a.num_attention_heads); a.num_key_value_heads = tc.value("num_key_value_heads", a.num_key_value_heads);
    a.head_dim = tc.value("head_dim", a.head_dim); a.intermediate_size = tc.value("intermediate_size", a.intermediate_size);
    a.vocab_size = tc.value("vocab_size", a.vocab_size); a.rms_norm_eps = tc.value("rms_norm_eps", a.rms_norm_eps);
    a.sliding_window = tc.value("sliding_window", a.sliding_window); a.sliding_window_pattern = tc.value("sliding_window_pattern", a.sliding_window_pattern);
    a.query_pre_attn_scalar = tc.value("query_pre_attn_scalar", a.query_pre_attn_scalar);
    a.tie_word_embeddings = tc.value("tie_word_embeddings", j.value("tie_word_embeddings", a.tie_word_embeddings));
    if (!tc.contains("head_dim") && a.num_attention_heads > 0) a.head_dim = a.hidden_size / a.num_attention_heads;
    if (tc.contains("rope_local_base_freq")) a.rope_local_base_freq = tc["rope_local_base_freq"].get<float>();
    if (tc.contains("rope_global_base_freq")) a.rope_global_base_freq = tc["rope_global_base_freq"].get<float>();
    return a;
}

Gemma3Model::~Gemma3Model() {
    free(embed_tokens_); free(final_norm_); if (!tie_word_embeddings_) free(lm_head_);
    free(x_); free(x_norm_); free(logits_); free(scratch_qkv_); free(scratch_attn_); free(scratch_ffn_);
    free(rope_local_cos_); free(rope_local_sin_); free(rope_global_cos_); free(rope_global_sin_);
    for (auto& lw : layers_) {
        free(lw.input_layernorm); free(lw.post_attention_layernorm); free(lw.pre_feedforward_layernorm); free(lw.post_feedforward_layernorm);
        free(lw.q_norm); free(lw.k_norm);
        metal_free_buffer(lw.m_input_norm_w); metal_free_buffer(lw.m_post_attn_norm_w); metal_free_buffer(lw.m_pre_ffn_norm_w); metal_free_buffer(lw.m_post_ffn_norm_w);
    }
    for (auto& kv : kv_caches_) { free(kv.k_cache); free(kv.v_cache); metal_free_buffer(kv.m_k_cache); metal_free_buffer(kv.m_v_cache); }
    for (auto& lk : ane_layers_) ane_free_layer(&lk);
}

void Gemma3Model::reset() { for (auto& kv : kv_caches_) { kv.len = 0; kv.start = 0; } }

void Gemma3Model::apply_args(const Gemma3Args& a) {
    hidden_size_ = a.hidden_size; intermediate_size_ = a.intermediate_size; vocab_size_ = a.vocab_size;
    num_layers_ = a.num_hidden_layers; num_q_heads_ = a.num_attention_heads; num_kv_heads_ = a.num_key_value_heads;
    head_dim_ = a.head_dim; sliding_window_ = a.sliding_window; sliding_window_pattern_ = a.sliding_window_pattern;
    rope_local_theta_ = a.rope_local_base_freq; rope_global_theta_ = a.rope_global_base_freq;
    attn_scale_ = 1.0f / sqrtf(a.query_pre_attn_scalar); embed_scale_ = sqrtf((float)a.hidden_size);
    rms_eps_ = a.rms_norm_eps; tie_word_embeddings_ = a.tie_word_embeddings;
    q_proj_dim_ = num_q_heads_ * head_dim_; kv_proj_dim_ = num_kv_heads_ * head_dim_;
}

void Gemma3Model::build_rope_table(float* cos_out, float* sin_out, int len, float theta) const {
    int half = head_dim_ / 2;
    for (int pos = 0; pos < len; pos++) {
        for (int i = 0; i < half; i++) {
            float angle = pos / powf(theta, (float)(2 * i) / head_dim_);
            cos_out[pos * half + i] = cosf(angle); sin_out[pos * half + i] = sinf(angle);
        }
    }
}

bool Gemma3Model::load_weights(ModelWeights* sf) {
    const char* p = weight_prefix_.c_str(); char name[256];
    snprintf(name, 256, "%s.embed_tokens.weight", p); embed_tokens_ = sf->load_bf16_to_f32(name, (int64_t)vocab_size_ * hidden_size_);
    if (!embed_tokens_) return false;
    if (tie_word_embeddings_) lm_head_ = embed_tokens_;
    else { snprintf(name, 256, "%slm_head.weight", strcmp(p, "model") == 0 ? "" : "language_model."); lm_head_ = sf->load_bf16_to_f32(name, (int64_t)vocab_size_ * hidden_size_); }
    snprintf(name, 256, "%s.norm.weight", p); final_norm_ = sf->load_bf16_to_f32(name, hidden_size_);
    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];
        snprintf(name, 256, "%s.layers.%d.input_layernorm.weight", p, L); lw.input_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        snprintf(name, 256, "%s.layers.%d.post_attention_layernorm.weight", p, L); lw.post_attention_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        snprintf(name, 256, "%s.layers.%d.pre_feedforward_layernorm.weight", p, L); lw.pre_feedforward_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        snprintf(name, 256, "%s.layers.%d.post_feedforward_layernorm.weight", p, L); lw.post_feedforward_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        snprintf(name, 256, "%s.layers.%d.self_attn.q_norm.weight", p, L); lw.q_norm = sf->load_bf16_to_f32(name, head_dim_);
        snprintf(name, 256, "%s.layers.%d.self_attn.k_norm.weight", p, L); lw.k_norm = sf->load_bf16_to_f32(name, head_dim_);
        if (!lw.input_layernorm || !lw.post_attention_layernorm || !lw.q_norm || !lw.k_norm) return false;
        if (metal_available()) {
            lw.m_input_norm_w = metal_new_buffer(lw.input_layernorm, hidden_size_ * 4);
            lw.m_post_attn_norm_w = metal_new_buffer(lw.post_attention_layernorm, hidden_size_ * 4);
            lw.m_pre_ffn_norm_w = metal_new_buffer(lw.pre_feedforward_layernorm, hidden_size_ * 4);
            lw.m_post_ffn_norm_w = metal_new_buffer(lw.post_feedforward_layernorm, hidden_size_ * 4);
        }
        snprintf(name, 256, "%s.layers.%d.self_attn.q_proj.weight", p, L); lw.q_proj_w = sf->get_bf16_ptr(name);
        snprintf(name, 256, "%s.layers.%d.self_attn.k_proj.weight", p, L); lw.k_proj_w = sf->get_bf16_ptr(name);
        snprintf(name, 256, "%s.layers.%d.self_attn.v_proj.weight", p, L); lw.v_proj_w = sf->get_bf16_ptr(name);
        snprintf(name, 256, "%s.layers.%d.self_attn.o_proj.weight", p, L); lw.o_proj_w = sf->get_bf16_ptr(name);
        snprintf(name, 256, "%s.layers.%d.mlp.gate_proj.weight", p, L); lw.gate_proj_w = sf->get_bf16_ptr(name);
        snprintf(name, 256, "%s.layers.%d.mlp.up_proj.weight", p, L); lw.up_proj_w = sf->get_bf16_ptr(name);
        snprintf(name, 256, "%s.layers.%d.mlp.down_proj.weight", p, L); lw.down_proj_w = sf->get_bf16_ptr(name);
    }
    return true;
}

static std::string blob_path(const std::string& dir, const char* tensor_name) {
    std::string p = dir + "/"; for (const char* c = tensor_name; *c; c++) p += (*c == '.') ? '/' : *c;
    p += ".bin"; return p;
}

bool Gemma3Model::compile_ane(ModelWeights* sf, const std::string& blob_dir) {
    if (!ane_available()) return false;
    use_ane_first_proj_ = (num_layers_ * 2 <= 51);
    const char* p = weight_prefix_.c_str(); ffn_is_fused_ = true;
    {
        char g0[256], u0[256], d0[256]; snprintf(g0, 256, "%s.layers.0.mlp.gate_proj.weight", p); snprintf(u0, 256, "%s.layers.0.mlp.up_proj.weight", p); snprintf(d0, 256, "%s.layers.0.mlp.down_proj.weight", p);
        ANEKernel* test = !blob_dir.empty() ? ane_compile_fused_ffn_gelu_blob(blob_path(blob_dir, g0), blob_path(blob_dir, u0), blob_path(blob_dir, d0), hidden_size_, intermediate_size_)
                                            : ane_compile_fused_ffn_gelu(sf->get_bf16_ptr(g0), sf->get_bf16_ptr(u0), sf->get_bf16_ptr(d0), hidden_size_, intermediate_size_);
        if (!test) ffn_is_fused_ = false; else ane_free(test);
    }
    if (!ffn_is_fused_) scratch_ffn_ = (float*)calloc(intermediate_size_, 4);
    for (int L = 0; L < num_layers_; L++) {
        char n1[256], n2[256], n3[256];
        if (use_ane_first_proj_) {
            snprintf(n1, 256, "%s.layers.%d.self_attn.q_proj.weight", p, L); snprintf(n2, 256, "%s.layers.%d.self_attn.k_proj.weight", p, L); snprintf(n3, 256, "%s.layers.%d.self_attn.v_proj.weight", p, L);
            ane_layers_[L].first_proj = !blob_dir.empty() ? ane_compile_fused_3_blob(blob_path(blob_dir, n1), q_proj_dim_, blob_path(blob_dir, n2), kv_proj_dim_, blob_path(blob_dir, n3), kv_proj_dim_, hidden_size_)
                                                          : ane_compile_fused_3(sf->get_bf16_ptr(n1), q_proj_dim_, sf->get_bf16_ptr(n2), kv_proj_dim_, sf->get_bf16_ptr(n3), kv_proj_dim_, hidden_size_);
        }
        if (ffn_is_fused_) {
            snprintf(n1, 256, "%s.layers.%d.mlp.gate_proj.weight", p, L); snprintf(n2, 256, "%s.layers.%d.mlp.up_proj.weight", p, L); snprintf(n3, 256, "%s.layers.%d.mlp.down_proj.weight", p, L);
            ane_layers_[L].fused_ffn = !blob_dir.empty() ? ane_compile_fused_ffn_gelu_blob(blob_path(blob_dir, n1), blob_path(blob_dir, n2), blob_path(blob_dir, n3), hidden_size_, intermediate_size_)
                                                        : ane_compile_fused_ffn_gelu(sf->get_bf16_ptr(n1), sf->get_bf16_ptr(n2), sf->get_bf16_ptr(n3), hidden_size_, intermediate_size_);
        }
    }
    return true;
}

bool Gemma3Model::load(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json"); if (!f.is_open()) return false;
    json j = json::parse(f); apply_args(Gemma3Args::from_json(j));
    auto sf = ModelWeights::open(model_dir); if (!sf) return false;
    weight_prefix_ = sf->find("model.embed_tokens.weight") ? "model" : "language_model.model";
    const char* p = weight_prefix_.c_str(); char b[256];
    snprintf(b, 256, "%s.embed_tokens.weight", p); const SFTensor* e = sf->find(b);
    snprintf(b, 256, "%s.layers.0.mlp.gate_proj.weight", p); const SFTensor* g = sf->find(b);
    if (!e || !g) return false;
    hidden_size_ = (int)e->shape[1]; vocab_size_ = (int)e->shape[0]; intermediate_size_ = (int)g->shape[0];
    if (head_dim_ <= 0 && num_q_heads_ > 0) head_dim_ = hidden_size_ / num_q_heads_;
    q_proj_dim_ = num_q_heads_ * head_dim_; kv_proj_dim_ = num_kv_heads_ * head_dim_; embed_scale_ = sqrtf((float)hidden_size_);
    ane_init(); metal_init();
    x_ = (float*)calloc(hidden_size_, 4); x_norm_ = (float*)calloc(hidden_size_, 4); logits_ = (float*)calloc(vocab_size_, 4);
    scratch_qkv_ = (float*)calloc((size_t)q_proj_dim_ + 2 * kv_proj_dim_, 4);
    scratch_attn_ = (float*)calloc(std::max(q_proj_dim_, hidden_size_), 4);
    rope_cache_len_ = 8192; int half = head_dim_ / 2;
    rope_local_cos_ = (float*)calloc((size_t)rope_cache_len_ * half, 4); rope_local_sin_ = (float*)calloc((size_t)rope_cache_len_ * half, 4);
    rope_global_cos_ = (float*)calloc((size_t)rope_cache_len_ * half, 4); rope_global_sin_ = (float*)calloc((size_t)rope_cache_len_ * half, 4);
    build_rope_table(rope_local_cos_, rope_local_sin_, rope_cache_len_, rope_local_theta_);
    build_rope_table(rope_global_cos_, rope_global_sin_, rope_cache_len_, rope_global_theta_);
    is_global_attn_.resize(num_layers_); for (int L = 0; L < num_layers_; L++) is_global_attn_[L] = (sliding_window_pattern_ == 0) || ((L + 1) % sliding_window_pattern_ == 0);
    layers_.resize(num_layers_); kv_caches_.resize(num_layers_); ane_layers_.resize(num_layers_);
    for (int L = 0; L < num_layers_; L++) {
        auto& kv = kv_caches_[L]; kv.capacity = KV_CACHE_CAPACITY;
        kv.k_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, 4);
        kv.v_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, 4);
        if (metal_available()) {
            kv.m_k_cache = metal_new_buffer(kv.k_cache, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * 4);
            kv.m_v_cache = metal_new_buffer(kv.v_cache, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * 4);
        }
    }
    if (!load_weights(sf.get())) return false;
    if (!compile_ane(sf.get(), "")) return false;
    weights_ = std::move(sf); return true;
}

float* Gemma3Model::forward(int token_id, int pos) {
    for (int i = 0; i < hidden_size_; i++) x_[i] = embed_tokens_[(int64_t)token_id * hidden_size_ + i] * embed_scale_;
    float* pre_oproj = scratch_attn_;
    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L]; auto& cache = kv_caches_[L];
        if (metal_available()) metal_rmsnorm(x_norm_, x_, lw.m_input_norm_w, hidden_size_, rms_eps_);
        else rmsnorm_gemma(x_norm_, x_, lw.input_layernorm, hidden_size_, rms_eps_);
        float* qkv_buf = scratch_qkv_;
        if (use_ane_first_proj_) { if (!ane_matvec(ane_layers_[L].first_proj, qkv_buf, x_norm_, hidden_size_, q_proj_dim_ + 2 * kv_proj_dim_)) return nullptr; }
        else { matvec_bf16(qkv_buf, lw.q_proj_w, x_norm_, q_proj_dim_, hidden_size_); matvec_bf16(qkv_buf + q_proj_dim_, lw.k_proj_w, x_norm_, kv_proj_dim_, hidden_size_); matvec_bf16(qkv_buf + q_proj_dim_ + kv_proj_dim_, lw.v_proj_w, x_norm_, kv_proj_dim_, hidden_size_); }
        float* q_raw = qkv_buf; float* k_raw = qkv_buf + q_proj_dim_; float* v_raw = qkv_buf + q_proj_dim_ + kv_proj_dim_;
        for (int h = 0; h < num_q_heads_; h++) rmsnorm_gemma(q_raw + h * head_dim_, q_raw + h * head_dim_, lw.q_norm, head_dim_, rms_eps_);
        for (int h = 0; h < num_kv_heads_; h++) rmsnorm_gemma(k_raw + h * head_dim_, k_raw + h * head_dim_, lw.k_norm, head_dim_, rms_eps_);
        const float* cos_row = nullptr, *sin_row = nullptr;
        if (pos < rope_cache_len_) {
            int half = head_dim_ / 2;
            if (is_global_attn_[L]) { cos_row = rope_global_cos_ + (size_t)pos * half; sin_row = rope_global_sin_ + (size_t)pos * half; }
            else { cos_row = rope_local_cos_ + (size_t)pos * half; sin_row = rope_local_sin_ + (size_t)pos * half; }
        }
        apply_rope_gemma3(q_raw, k_raw, num_q_heads_, num_kv_heads_, head_dim_, pos, cos_row, sin_row, is_global_attn_[L] ? rope_global_theta_ : rope_local_theta_);
        int slot; if (cache.len < cache.capacity) { slot = (cache.start + cache.len) % cache.capacity; cache.len++; } else { slot = cache.start; cache.start = (cache.start + 1) % cache.capacity; }
        size_t kv_stride = (size_t)num_kv_heads_ * head_dim_; memcpy(cache.k_cache + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float)); memcpy(cache.v_cache + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));
        int eff_len = cache.len, eff_start = cache.start;
        if (!is_global_attn_[L] && sliding_window_ > 0 && cache.len > sliding_window_) { eff_len = sliding_window_; eff_start = (cache.start + (cache.len - sliding_window_)) % cache.capacity; }
        float factor = attn_scale_ * sqrtf((float)head_dim_); if (fabsf(factor - 1.0f) > 1e-6f) for (int i = 0; i < q_proj_dim_; i++) q_raw[i] *= factor;
        if (metal_available()) metal_gqa_attention(pre_oproj, q_raw, cache.m_k_cache, cache.m_v_cache, num_q_heads_, num_kv_heads_, head_dim_, eff_start, eff_len, cache.capacity);
        else gqa_attention(pre_oproj, q_raw, cache.k_cache, cache.v_cache, num_q_heads_, num_kv_heads_, head_dim_, head_dim_, eff_start, eff_len, cache.capacity);
        float* attn_out = x_norm_; matvec_bf16(attn_out, lw.o_proj_w, pre_oproj, hidden_size_, q_proj_dim_);
        if (metal_available()) metal_rmsnorm(attn_out, attn_out, lw.m_post_attn_norm_w, hidden_size_, rms_eps_);
        else rmsnorm_gemma(attn_out, attn_out, lw.post_attention_layernorm, hidden_size_, rms_eps_);
        for (int i = 0; i < hidden_size_; i++) x_[i] += attn_out[i];
        if (metal_available()) metal_rmsnorm(x_norm_, x_, lw.m_pre_ffn_norm_w, hidden_size_, rms_eps_);
        else rmsnorm_gemma(x_norm_, x_, lw.pre_feedforward_layernorm, hidden_size_, rms_eps_);
        float* ffn_out = scratch_attn_;
        if (ffn_is_fused_) { if (!ane_matvec(ane_layers_[L].fused_ffn, ffn_out, x_norm_, hidden_size_, hidden_size_)) return nullptr; }
        else {
            float* gate_out = scratch_ffn_; matvec_bf16(gate_out, lw.gate_proj_w, x_norm_, intermediate_size_, hidden_size_); matvec_bf16(ffn_out, lw.up_proj_w, x_norm_, intermediate_size_, hidden_size_);
            for (int i = 0; i < intermediate_size_; i++) { float g = gate_out[i]; gate_out[i] = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g))) * ffn_out[i]; }
            matvec_bf16(ffn_out, lw.down_proj_w, gate_out, hidden_size_, intermediate_size_);
        }
        if (metal_available()) metal_rmsnorm(ffn_out, ffn_out, lw.m_post_ffn_norm_w, hidden_size_, rms_eps_);
        else rmsnorm_gemma(ffn_out, ffn_out, lw.post_feedforward_layernorm, hidden_size_, rms_eps_);
        for (int i = 0; i < hidden_size_; i++) x_[i] += ffn_out[i];
    }
    rmsnorm_gemma(x_, x_, final_norm_, hidden_size_, rms_eps_);
    matvec(logits_, lm_head_, x_, vocab_size_, hidden_size_); return logits_;
}

} // namespace ane_lm
