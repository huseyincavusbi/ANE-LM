#pragma once

#include <IOSurface/IOSurfaceRef.h>

namespace ane_lm {

// Global Metal state
bool metal_init();
bool metal_available();

// Buffer management
void* metal_new_buffer(const void* data, size_t size);
void* metal_map_iosurface(IOSurfaceRef surface);
void  metal_free_buffer(void* buffer);

// Kernel dispatchers
void metal_rmsnorm(float* out, const float* x, void* weight_buf, int dim, float eps);
void metal_gqa_attention(float* out, const float* q, void* k_cache_buf, void* v_cache_buf,
                         int n_heads, int n_kv_heads, int head_dim, int cache_start, int cache_len, int cache_capacity);

} // namespace ane_lm
