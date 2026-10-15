// Fused unbiased auxiliaries inside a 2DGS-style tile raster loop.
// Loop structure follows Apache-2.0 gsplat RasterizeToPixels2DGS.
// Depth/G/corrected-depth/convergence math is SPEC.md, independently written.
// No Unbiased_Surfel or PGSR source.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr float kAlphaThreshold = 1.0f / 255.0f;
constexpr float kFilterInvSquare = 2.0f;

__device__ inline bool finite_f(float x) { return isfinite(x); }

// SPEC ray-splat depth VJP for a scalar d-gradient. G is stop-gradient.
__device__ void add_depth_grad(
    int32_t g,
    float px,
    float py,
    float mx,
    float my,
    float ux, float uy, float uz,
    float vx, float vy, float vz,
    float wx, float wy, float wz,
    float g_d,
    float* __restrict__ g_ray,
    float* __restrict__ g_means) {
    if (g_d == 0.f || g < 0) return;
    float hux = px * wx - ux, huy = px * wy - uy, huz = px * wz - uz;
    float hvx = py * wx - vx, hvy = py * wy - vy, hvz = py * wz - vz;
    float hx = huy * hvz - huz * hvy;
    float hy = huz * hvx - hux * hvz;
    float hz = hux * hvy - huy * hvx;
    if (hz == 0.f) return;
    float sx = hx / hz, sy = hy / hz;
    float rho3 = sx * sx + sy * sy;
    float ddx = mx - px, ddy = my - py;
    float rho2 = 2.0f * (ddx * ddx + ddy * ddy);
    bool plane = rho3 <= rho2;
    float g_sx = 0.f, g_sy = 0.f;
    float* gr = g_ray + g * 9;
    if (plane) {
        g_sx += g_d * wx;
        g_sy += g_d * wy;
        atomicAdd(gr + 6, g_d * sx);
        atomicAdd(gr + 7, g_d * sy);
        atomicAdd(gr + 8, g_d);
    } else {
        atomicAdd(gr + 8, g_d);
        return;
    }
    float inv_hz = 1.0f / hz;
    float g_hx = g_sx * inv_hz;
    float g_hy = g_sy * inv_hz;
    float g_hz = -(g_sx * sx + g_sy * sy) * inv_hz;
    float g_hux = hvy * g_hz - hvz * g_hy;
    float g_huy = hvz * g_hx - hvx * g_hz;
    float g_huz = hvx * g_hy - hvy * g_hx;
    float g_hvx = g_hy * huz - g_hz * huy;
    float g_hvy = g_hz * hux - g_hx * huz;
    float g_hvz = g_hx * huy - g_hy * hux;
    atomicAdd(gr + 0, -g_hux);
    atomicAdd(gr + 1, -g_huy);
    atomicAdd(gr + 2, -g_huz);
    atomicAdd(gr + 3, -g_hvx);
    atomicAdd(gr + 4, -g_hvy);
    atomicAdd(gr + 5, -g_hvz);
    atomicAdd(gr + 6, px * g_hux + py * g_hvx);
    atomicAdd(gr + 7, px * g_huy + py * g_hvy);
    atomicAdd(gr + 8, px * g_huz + py * g_hvz);
}

__global__ void rasterize_unbiased_aux_fwd_kernel(
    const int32_t I,
    const int32_t n_isects,
    const float* __restrict__ means2d,         // [I*N, 2]
    const float* __restrict__ ray_transforms,  // [I*N, 9]
    const float* __restrict__ opacities,       // [I*N]
    const int32_t image_width,
    const int32_t image_height,
    const int32_t tile_size,
    const int32_t tile_width,
    const int32_t tile_height,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const float opacity_eps,
    const float threshold,
    const float radius_quarter,
    float* __restrict__ out_depth,     // [I,H,W]
    bool* __restrict__ out_valid,      // [I,H,W]
    float* __restrict__ out_conv,      // [I,H,W]
    int32_t* __restrict__ out_sel_g    // [I,H,W]
) {
    const int32_t image_id = blockIdx.x;
    const int32_t tile_y = blockIdx.y;
    const int32_t tile_x = blockIdx.z;
    const int32_t tile_id = tile_y * tile_width + tile_x;
    const int32_t i = tile_y * tile_size + threadIdx.y;
    const int32_t j = tile_x * tile_size + threadIdx.x;
    const int32_t pix_id = i * image_width + j;
    const bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;

    const int32_t* offsets = tile_offsets + image_id * tile_height * tile_width;
    const int32_t range_start = offsets[tile_id];
    const int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : offsets[tile_id + 1];

    const int32_t block_size = blockDim.x * blockDim.y;
    const int32_t tr = threadIdx.y * blockDim.x + threadIdx.x;
    const int32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // LIMITATION (v2 note, no ABI change): flatten_ids holds per-image Gaussian
    // ids in [0, N), but means2d/ray_transforms/opacities are flattened [I*N].
    // Indexing g directly is only exact for a single image (I == 1). The Python
    // wrapper now rejects I != 1. A general fix needs an explicit num_gaussians
    // kernel arg and base = image_id * N + g (deferred to the next rebuild).
    extern __shared__ float smem[];
    int32_t* id_batch = reinterpret_cast<int32_t*>(smem);
    float* xy_opac = reinterpret_cast<float*>(id_batch + block_size);  // 3 per
    float* u_M = xy_opac + block_size * 3;
    float* v_M = u_M + block_size * 3;
    float* w_M = v_M + block_size * 3;

    float T = 1.0f;
    float cum = 0.0f;
    float depth = 0.0f;
    bool have_depth = false;
    int32_t sel_g = -1;
    float conv = 0.0f;
    float prev_d = 0.0f;
    float prev_G = 0.0f;
    bool have_prev = false;

    for (int32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) break;
        int32_t batch_start = range_start + block_size * b;
        int32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            xy_opac[tr * 3 + 0] = means2d[g * 2 + 0];
            xy_opac[tr * 3 + 1] = means2d[g * 2 + 1];
            xy_opac[tr * 3 + 2] = opacities[g];
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                u_M[tr * 3 + k] = ray_transforms[g * 9 + k];
                v_M[tr * 3 + k] = ray_transforms[g * 9 + 3 + k];
                w_M[tr * 3 + k] = ray_transforms[g * 9 + 6 + k];
            }
        }
        __syncthreads();
        int32_t batch_size = min(block_size, range_end - batch_start);
        for (int32_t t = 0; t < batch_size && !done; ++t) {
            float opac = xy_opac[t * 3 + 2];
            float ux = u_M[t * 3 + 0], uy = u_M[t * 3 + 1], uz = u_M[t * 3 + 2];
            float vx = v_M[t * 3 + 0], vy = v_M[t * 3 + 1], vz = v_M[t * 3 + 2];
            float wx = w_M[t * 3 + 0], wy = w_M[t * 3 + 1], wz = w_M[t * 3 + 2];
            float hux = px * wx - ux, huy = px * wy - uy, huz = px * wz - uz;
            float hvx = py * wx - vx, hvy = py * wy - vy, hvz = py * wz - vz;
            float hx = huy * hvz - huz * hvy;
            float hy = huz * hvx - hux * hvz;
            float hz = hux * hvy - huy * hvx;
            if (hz == 0.f) continue;
            float sx = hx / hz, sy = hy / hz;
            float rho3 = sx * sx + sy * sy;
            float dx = xy_opac[t * 3 + 0] - px;
            float dy = xy_opac[t * 3 + 1] - py;
            float rho2 = kFilterInvSquare * (dx * dx + dy * dy);
            float rho = rho3 < rho2 ? rho3 : rho2;
            float G = expf(-0.5f * rho);
            float alpha = fminf(0.999f, opac * G);
            if (rho < 0.f || alpha < kAlphaThreshold) continue;
            float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            float d = (rho3 <= rho2) ? (sx * wx + sy * wy + wz) : wz;
            if (!(finite_f(G) && finite_f(d))) {
                T = next_T;
                continue;
            }
            float contrib = (opac + opacity_eps) * G;
            if (finite_f(contrib)) cum += contrib;
            if (!have_depth && finite_f(cum) && cum > threshold) {
                have_depth = true;
                depth = d;
                sel_g = id_batch[t];
            }
            if (have_prev) {
                float delta = d - prev_d;
                if (finite_f(delta) && fabsf(delta) <= radius_quarter) {
                    float w = G < prev_G ? G : prev_G;
                    conv += w * delta * delta;
                }
            }
            prev_d = d;
            prev_G = G;
            have_prev = true;
            T = next_T;
        }
        __syncthreads();
    }

    if (inside) {
        int32_t out = image_id * image_height * image_width + pix_id;
        out_depth[out] = have_depth ? depth : 0.f;
        out_valid[out] = have_depth;
        out_conv[out] = conv;
        out_sel_g[out] = sel_g;
    }
}

}  // namespace

std::vector<torch::Tensor> rasterize_unbiased_aux_cuda(
    torch::Tensor means2d,
    torch::Tensor ray_transforms,
    torch::Tensor opacities,
    torch::Tensor tile_offsets,
    torch::Tensor flatten_ids,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    double opacity_eps,
    double threshold,
    double scene_radius) {
    TORCH_CHECK(means2d.is_cuda() && means2d.scalar_type() == torch::kFloat32,
                "fused aux raster expects float32 CUDA tensors");
    const auto I = tile_offsets.size(0);
    const auto tile_height = tile_offsets.size(1);
    const auto tile_width = tile_offsets.size(2);
    const int32_t n_isects = static_cast<int32_t>(flatten_ids.numel());
    auto depth = torch::zeros({I, image_height, image_width}, means2d.options());
    auto valid = torch::zeros({I, image_height, image_width}, torch::dtype(torch::kBool).device(means2d.device()));
    auto conv = torch::zeros({I, image_height, image_width}, means2d.options());
    auto sel = torch::full({I, image_height, image_width}, -1, torch::dtype(torch::kInt32).device(means2d.device()));
    const c10::cuda::CUDAGuard guard(means2d.device());
    dim3 block(static_cast<unsigned>(tile_size), static_cast<unsigned>(tile_size), 1);
    dim3 grid(static_cast<unsigned>(I), static_cast<unsigned>(tile_height), static_cast<unsigned>(tile_width));
    int32_t block_size = static_cast<int32_t>(tile_size * tile_size);
    size_t smem = block_size * (sizeof(int32_t) + sizeof(float) * (3 + 9));
    rasterize_unbiased_aux_fwd_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        static_cast<int32_t>(I),
        n_isects,
        means2d.contiguous().data_ptr<float>(),
        ray_transforms.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        static_cast<int32_t>(image_width),
        static_cast<int32_t>(image_height),
        static_cast<int32_t>(tile_size),
        static_cast<int32_t>(tile_width),
        static_cast<int32_t>(tile_height),
        tile_offsets.contiguous().data_ptr<int32_t>(),
        flatten_ids.contiguous().data_ptr<int32_t>(),
        static_cast<float>(opacity_eps),
        static_cast<float>(threshold),
        static_cast<float>(scene_radius / 4.0),
        depth.data_ptr<float>(),
        valid.data_ptr<bool>(),
        conv.data_ptr<float>(),
        sel.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {depth, valid, conv, sel};
}

__global__ void rasterize_unbiased_aux_bwd_kernel(
    const int32_t I,
    const int32_t n_isects,
    const float* __restrict__ means2d,
    const float* __restrict__ ray_transforms,
    const float* __restrict__ opacities,
    const int32_t image_width,
    const int32_t image_height,
    const int32_t tile_size,
    const int32_t tile_width,
    const int32_t tile_height,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const float opacity_eps,
    const float threshold,
    const float radius_quarter,
    const float far_k,
    const float* __restrict__ g_depth,
    const float* __restrict__ g_conv,
    float* __restrict__ g_ray,
    float* __restrict__ g_means,
    float* __restrict__ g_opac
) {
    const int32_t image_id = blockIdx.x;
    const int32_t tile_y = blockIdx.y;
    const int32_t tile_x = blockIdx.z;
    const int32_t tile_id = tile_y * tile_width + tile_x;
    const int32_t i = tile_y * tile_size + threadIdx.y;
    const int32_t j = tile_x * tile_size + threadIdx.x;
    const bool inside = (i < image_height && j < image_width);
    bool done = !inside;
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = i * image_width + j;
    const int32_t out = image_id * image_height * image_width + pix_id;
    const float gd = inside ? g_depth[out] : 0.f;
    const float gc = inside ? g_conv[out] : 0.f;

    const int32_t* offsets = tile_offsets + image_id * tile_height * tile_width;
    const int32_t range_start = offsets[tile_id];
    const int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : offsets[tile_id + 1];
    const int32_t block_size = blockDim.x * blockDim.y;
    const int32_t tr = threadIdx.y * blockDim.x + threadIdx.x;
    const int32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ float smem_b[];
    int32_t* id_batch = reinterpret_cast<int32_t*>(smem_b);
    float* xy_opac = reinterpret_cast<float*>(id_batch + block_size);
    float* u_M = xy_opac + block_size * 3;
    float* v_M = u_M + block_size * 3;
    float* w_M = v_M + block_size * 3;

    float T = 1.0f;
    float cum = 0.0f;
    bool have_depth = false;
    float prev_d = 0.0f, prev_G = 0.0f;
    int32_t prev_g = -1;
    float prev_mx = 0.f, prev_my = 0.f;
    float prev_ux = 0.f, prev_uy = 0.f, prev_uz = 0.f;
    float prev_vx = 0.f, prev_vy = 0.f, prev_vz = 0.f;
    float prev_wx = 0.f, prev_wy = 0.f, prev_wz = 0.f;
    bool have_prev = false;

    for (int32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) break;
        int32_t batch_start = range_start + block_size * b;
        int32_t idx = batch_start + tr;
        if (idx < range_end) {
            // Same single-image limitation as the forward kernel (see note above).
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            xy_opac[tr * 3 + 0] = means2d[g * 2 + 0];
            xy_opac[tr * 3 + 1] = means2d[g * 2 + 1];
            xy_opac[tr * 3 + 2] = opacities[g];
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                u_M[tr * 3 + k] = ray_transforms[g * 9 + k];
                v_M[tr * 3 + k] = ray_transforms[g * 9 + 3 + k];
                w_M[tr * 3 + k] = ray_transforms[g * 9 + 6 + k];
            }
        }
        __syncthreads();
        int32_t batch_size = min(block_size, range_end - batch_start);
        for (int32_t t = 0; t < batch_size && !done; ++t) {
            float opac = xy_opac[t * 3 + 2];
            float ux = u_M[t * 3 + 0], uy = u_M[t * 3 + 1], uz = u_M[t * 3 + 2];
            float vx = v_M[t * 3 + 0], vy = v_M[t * 3 + 1], vz = v_M[t * 3 + 2];
            float wx = w_M[t * 3 + 0], wy = w_M[t * 3 + 1], wz = w_M[t * 3 + 2];
            float hux = px * wx - ux, huy = px * wy - uy, huz = px * wz - uz;
            float hvx = py * wx - vx, hvy = py * wy - vy, hvz = py * wz - vz;
            float hx = huy * hvz - huz * hvy;
            float hy = huz * hvx - hux * hvz;
            float hz = hux * hvy - huy * hvx;
            if (hz == 0.f) continue;
            float sx = hx / hz, sy = hy / hz;
            float rho3 = sx * sx + sy * sy;
            float ddx = xy_opac[t * 3 + 0] - px;
            float ddy = xy_opac[t * 3 + 1] - py;
            float rho2 = 2.0f * (ddx * ddx + ddy * ddy);
            float rho = rho3 < rho2 ? rho3 : rho2;
            float G = expf(-0.5f * rho);
            float alpha = fminf(0.999f, opac * G);
            if (rho < 0.f || alpha < (1.0f / 255.0f)) continue;
            float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            float d = (rho3 <= rho2) ? (sx * wx + sy * wy + wz) : wz;
            int32_t g = id_batch[t];
            float contrib = (opac + opacity_eps) * G;
            bool crossing = false;
            if (finite_f(contrib)) cum += contrib;
            if (!have_depth && finite_f(cum) && cum > threshold) {
                have_depth = true;
                crossing = true;
            }
            float g_d = crossing ? gd : 0.f;
            if (have_prev && finite_f(d - prev_d) && fabsf(d - prev_d) <= radius_quarter) {
                float delta = d - prev_d;
                float w = G < prev_G ? G : prev_G;
                float common = 2.f * delta * gc * w;
                g_d += common * far_k;
                add_depth_grad(
                    prev_g, px, py, prev_mx, prev_my,
                    prev_ux, prev_uy, prev_uz, prev_vx, prev_vy, prev_vz,
                    prev_wx, prev_wy, prev_wz, -common, g_ray, g_means);
            }
            if (g_d != 0.f && finite_f(d) && finite_f(G)) {
                add_depth_grad(
                    g, px, py, xy_opac[t * 3 + 0], xy_opac[t * 3 + 1],
                    ux, uy, uz, vx, vy, vz, wx, wy, wz, g_d, g_ray, g_means);
            }
            prev_d = d;
            prev_G = G;
            prev_g = g;
            prev_mx = xy_opac[t * 3 + 0];
            prev_my = xy_opac[t * 3 + 1];
            prev_ux = ux; prev_uy = uy; prev_uz = uz;
            prev_vx = vx; prev_vy = vy; prev_vz = vz;
            prev_wx = wx; prev_wy = wy; prev_wz = wz;
            have_prev = true;
            T = next_T;
        }
        __syncthreads();
    }
}

std::vector<torch::Tensor> rasterize_unbiased_aux_backward_cuda(
    torch::Tensor means2d,
    torch::Tensor ray_transforms,
    torch::Tensor opacities,
    torch::Tensor tile_offsets,
    torch::Tensor flatten_ids,
    torch::Tensor g_depth,
    torch::Tensor g_conv,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    double opacity_eps,
    double threshold,
    double scene_radius,
    double far_k) {
    auto g_ray = torch::zeros_like(ray_transforms);
    auto g_means = torch::zeros_like(means2d);
    auto g_opac = torch::zeros_like(opacities);
    const auto I = tile_offsets.size(0);
    const auto tile_height = tile_offsets.size(1);
    const auto tile_width = tile_offsets.size(2);
    const int32_t n_isects = static_cast<int32_t>(flatten_ids.numel());
    const c10::cuda::CUDAGuard guard(means2d.device());
    dim3 block(static_cast<unsigned>(tile_size), static_cast<unsigned>(tile_size), 1);
    dim3 grid(static_cast<unsigned>(I), static_cast<unsigned>(tile_height), static_cast<unsigned>(tile_width));
    int32_t block_size = static_cast<int32_t>(tile_size * tile_size);
    size_t smem = block_size * (sizeof(int32_t) + sizeof(float) * 12);
    rasterize_unbiased_aux_bwd_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        static_cast<int32_t>(I),
        n_isects,
        means2d.contiguous().data_ptr<float>(),
        ray_transforms.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        static_cast<int32_t>(image_width),
        static_cast<int32_t>(image_height),
        static_cast<int32_t>(tile_size),
        static_cast<int32_t>(tile_width),
        static_cast<int32_t>(tile_height),
        tile_offsets.contiguous().data_ptr<int32_t>(),
        flatten_ids.contiguous().data_ptr<int32_t>(),
        static_cast<float>(opacity_eps),
        static_cast<float>(threshold),
        static_cast<float>(scene_radius / 4.0),
        static_cast<float>(far_k),
        g_depth.contiguous().data_ptr<float>(),
        g_conv.contiguous().data_ptr<float>(),
        g_ray.data_ptr<float>(),
        g_means.data_ptr<float>(),
        g_opac.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_means, g_ray, g_opac};
}
