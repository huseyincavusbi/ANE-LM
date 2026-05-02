#import "metal_ops.h"
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <ane_lm/common.h>

namespace ane_lm {

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> pso_rmsnorm = nil;
static id<MTLComputePipelineState> pso_attention = nil;

static const char* METAL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

kernel void kernel_rmsnorm_gemma(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       dim     [[buffer(3)]],
    constant float&     eps     [[buffer(4)]],
    uint                tpig    [[thread_position_in_grid]])
{
    if (tpig >= (uint)dim) return;
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrt(ss / dim + eps);
    out[tpig] = x[tpig] * scale * (1.0f + weight[tpig]);
}

kernel void kernel_gqa_attention(
    device float*       out      [[buffer(0)]],
    device const float* q        [[buffer(1)]],
    device const float* k_cache  [[buffer(2)]],
    device const float* v_cache  [[buffer(3)]],
    constant int&       n_heads  [[buffer(4)]],
    constant int&       n_kv_heads [[buffer(5)]],
    constant int&       head_dim [[buffer(6)]],
    constant int&       cache_start [[buffer(7)]],
    constant int&       cache_len  [[buffer(8)]],
    constant int&       cache_cap  [[buffer(9)]],
    uint                h        [[thread_position_in_grid]])
{
    if (h >= (uint)n_heads) return;
    int groups = n_heads / n_kv_heads;
    int kv_h = h / groups;
    float scale = 1.0f / sqrt((float)head_dim);
    device const float* qh = q + h * head_dim;
    device float* oh = out + h * head_dim;

    float scores[2048]; // Still stack-based but now the dispatch is safer
    float max_score = -1e20f;
    for (int s = 0; s < cache_len; s++) {
        int slot = (cache_start + s) % cache_cap;
        device const float* kh = k_cache + (slot * n_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; i++) dot += qh[i] * kh[i];
        float score = dot * scale;
        scores[s] = score;
        if (score > max_score) max_score = score;
    }
    float sum_exp = 0.0f;
    for (int s = 0; s < cache_len; s++) {
        scores[s] = exp(scores[s] - max_score);
        sum_exp += scores[s];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < head_dim; i++) oh[i] = 0.0f;
    for (int s = 0; s < cache_len; s++) {
        int slot = (cache_start + s) % cache_cap;
        device const float* vh = v_cache + (slot * n_kv_heads + kv_h) * head_dim;
        float sv = scores[s] * inv_sum;
        for (int i = 0; i < head_dim; i++) oh[i] += sv * vh[i];
    }
}
)";

bool metal_init() {
    @autoreleasepool {
        if (g_device) return true;
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return false;
        g_queue = [g_device newCommandQueue];
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:METAL_SOURCE];
        id<MTLLibrary> lib = [g_device newLibraryWithSource:src options:nil error:&error];
        if (!lib) return false;
        auto create_pso = [&](const char* name) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            return [g_device newComputePipelineStateWithFunction:fn error:nil];
        };
        pso_rmsnorm = create_pso("kernel_rmsnorm_gemma");
        pso_attention = create_pso("kernel_gqa_attention");
        LOG("Metal: Using %s\n", [g_device.name UTF8String]);
        return (pso_rmsnorm != nil);
    }
}

bool metal_available() { return g_device != nil; }

void* metal_new_buffer(const void* data, size_t size) {
    if (!g_device) return nullptr;
    id<MTLBuffer> buf = [g_device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    return (void*)CFRetain((CFTypeRef)buf);
}

void* metal_map_iosurface(IOSurfaceRef surface) {
    if (!g_device || !surface) return nullptr;
    SEL sel = sel_registerName("newBufferWithIOSurface:");
    id<MTLBuffer> buf = ((id(*)(id, SEL, IOSurfaceRef))objc_msgSend)(g_device, sel, surface);
    if (buf) return (void*)CFRetain((CFTypeRef)buf);
    return nullptr;
}

void metal_free_buffer(void* buffer) {
    if (buffer) CFRelease((CFTypeRef)buffer);
}

void metal_rmsnorm(float* out, const float* x, void* weight_buf, int dim, float eps) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso_rmsnorm];
        [enc setBytes:x length:dim * 4 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:1];
        [enc setBytes:out length:dim * 4 atIndex:2];
        [enc setBytes:&dim length:4 atIndex:3];
        [enc setBytes:&eps length:4 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }
}

void metal_gqa_attention(float* out, const float* q, void* k_cache_buf, void* v_cache_buf,
                         int n_heads, int n_kv_heads, int head_dim, int cache_start, int cache_len, int cache_capacity) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso_attention];
        [enc setBytes:out length:n_heads * head_dim * 4 atIndex:0];
        [enc setBytes:q length:n_heads * head_dim * 4 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)k_cache_buf offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)v_cache_buf offset:0 atIndex:3];
        [enc setBytes:&n_heads length:4 atIndex:4];
        [enc setBytes:&n_kv_heads length:4 atIndex:5];
        [enc setBytes:&head_dim length:4 atIndex:6];
        [enc setBytes:&cache_start length:4 atIndex:7];
        [enc setBytes:&cache_len length:4 atIndex:8];
        [enc setBytes:&cache_capacity length:4 atIndex:9];
        [enc dispatchThreads:MTLSizeMake(n_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }
}

} // namespace ane_lm
