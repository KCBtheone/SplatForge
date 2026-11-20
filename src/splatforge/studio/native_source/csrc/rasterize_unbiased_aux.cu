// Fused unbiased auxiliaries inside a 2DGS-style tile raster loop.
// Loop structure follows Apache-2.0 gsplat RasterizeToPixels2DGS.
// Depth/G/corrected-depth/convergence math is SPEC.md, independently written.
// No Unbiased_Surfel or PGSR source.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cub/device/device_radix_sort.cuh>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

bool front_prefix_forward_enabled() {
#if defined(FORGE_EXACT_FRONT_PREFIX_FORWARD)
    return true;
#else
    return false;
#endif
}

namespace {

constexpr float kAlphaThreshold = 1.0f / 255.0f;
constexpr float kFilterInvSquare = 2.0f;

__device__ inline bool finite_f(float x) { return isfinite(x); }

// Scalar accumulation used by the ordinary production auxiliary VJP.
__device__ inline void accumulate_ray_vjp(int32_t g,float* values,float* g_ray) {
#pragma unroll
    for(int k=0;k<9;++k) {
        if(values[k]!=0.f) atomicAdd(g_ray+g*9+k,values[k]);
    }
}

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
    float values[9] = {};
    if (plane) {
        g_sx += g_d * wx;
        g_sy += g_d * wy;
        values[6] = g_d * sx;
        values[7] = g_d * sy;
        values[8] = g_d;
    } else {
        values[8] = g_d;
        accumulate_ray_vjp(g, values, g_ray);
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
    values[0] = -g_hux;
    values[1] = -g_huy;
    values[2] = -g_huz;
    values[3] = -g_hvx;
    values[4] = -g_hvy;
    values[5] = -g_hvz;
    values[6] += px * g_hux + py * g_hvx;
    values[7] += px * g_huy + py * g_hvy;
    values[8] += px * g_huz + py * g_hvz;
    accumulate_ray_vjp(g, values, g_ray);
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
    const bool opacity_independent,
    const bool surface_midpoint,
    const bool front_convergence,
    const bool terminal_surface,
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
#if defined(FORGE_EXACT_FRONT_PREFIX_FORWARD)
            // Depth/selection are fixed after crossing. Front energy is inactive
            // once the original T domain has ended. Keep terminal fallbacks live
            // when no crossing exists, and retain the ordinary non-front path.
            if (front_convergence && have_depth && T <= 0.09f) {
                done = true;
                break;
            }
#endif
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
            if (rho < 0.f || (opacity_independent ? G < kAlphaThreshold : alpha < kAlphaThreshold)) continue;
            float next_T = alpha < kAlphaThreshold ? T : T * (1.0f - alpha);
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
            if (!have_depth && ((finite_f(cum) && cum > threshold) || terminal_surface)) {
                depth = surface_midpoint && have_prev ? 0.5f*(d+prev_d) : d;
                sel_g = id_batch[t];
                have_depth = finite_f(cum) && cum > threshold;
            }
            if (have_prev && (!front_convergence || T > 0.09f)) {
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
        const bool surface_valid = have_depth || (terminal_surface && have_prev);
        out_depth[out] = surface_valid ? depth : 0.f;
        out_valid[out] = surface_valid;
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
    double scene_radius,
    bool opacity_independent,
    bool surface_midpoint,
    bool front_convergence,
    bool terminal_surface) {
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
        opacity_independent,
        surface_midpoint,
        front_convergence,
        terminal_surface,
        depth.data_ptr<float>(),
        valid.data_ptr<bool>(),
        conv.data_ptr<float>(),
        sel.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {depth, valid, conv, sel};
}

template <bool CarryDepthVjp>
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
    const bool opacity_independent,
    const bool front_convergence,
    const bool terminal_surface,
    const bool coherent_midpoint,
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
    bool prev_has_previous = false;
    float pending_depth_vjp = 0.f;
    // Needed only when the final (non-crossing) surface averages two hits.
    int32_t terminal_prior_g = -1;
    float terminal_prior[11];

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
            if (rho < 0.f || (opacity_independent ? G < kAlphaThreshold : alpha < kAlphaThreshold)) continue;
            float next_T = alpha < kAlphaThreshold ? T : T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            float d = (rho3 <= rho2) ? (sx * wx + sy * wy + wz) : wz;
            if (terminal_surface && !(finite_f(G) && finite_f(d))) {
                T = next_T;
                continue;
            }
            int32_t g = id_batch[t];
            float contrib = (opac + opacity_eps) * G;
            bool crossing = false;
            if (finite_f(contrib)) cum += contrib;
            if (!have_depth && finite_f(cum) && cum > threshold) {
                have_depth = true;
                crossing = true;
            }
            float g_d = crossing ? (coherent_midpoint && have_prev ? .5f*gd : gd) : 0.f;
            float previous_vjp = crossing && coherent_midpoint && have_prev ? .5f*gd : 0.f;
            if (have_prev && (!front_convergence || T > 0.09f) && finite_f(d - prev_d) && fabsf(d - prev_d) <= radius_quarter) {
                float delta = d - prev_d;
                float w = G < prev_G ? G : prev_G;
                float common = 2.f * delta * gc * w;
                float current_scale = front_convergence && delta < 0.f ? 1.f : far_k;
                float previous_scale = front_convergence && delta < 0.f ? far_k : 1.f;
                previous_vjp += front_convergence && !prev_has_previous ? 0.f : -common * previous_scale;
                g_d += common * current_scale;
            }
            if (have_prev && previous_vjp != 0.f) {
                if constexpr (CarryDepthVjp) {
                    pending_depth_vjp += previous_vjp;
                } else {
                    add_depth_grad(
                        prev_g, px, py, prev_mx, prev_my,
                        prev_ux, prev_uy, prev_uz, prev_vx, prev_vy, prev_vz,
                        prev_wx, prev_wy, prev_wz, previous_vjp, g_ray, g_means);
                }
            }
            if constexpr (CarryDepthVjp) {
                // The previous hit has now received both its own selected/far
                // contribution and this pair's near contribution. Emit once.
                if (have_prev) {
                    add_depth_grad(
                        prev_g, px, py, prev_mx, prev_my,
                        prev_ux, prev_uy, prev_uz, prev_vx, prev_vy, prev_vz,
                        prev_wx, prev_wy, prev_wz, pending_depth_vjp, g_ray, g_means);
                }
                pending_depth_vjp = finite_f(d) && finite_f(G) ? g_d : 0.f;
            } else {
                if (g_d != 0.f && finite_f(d) && finite_f(G)) {
                    add_depth_grad(
                        g, px, py, xy_opac[t * 3 + 0], xy_opac[t * 3 + 1],
                        ux, uy, uz, vx, vy, vz, wx, wy, wz, g_d, g_ray, g_means);
                }
            }
            if (coherent_midpoint && terminal_surface && !have_depth && have_prev) {
                terminal_prior_g = prev_g;
                terminal_prior[0]=prev_mx; terminal_prior[1]=prev_my;
                terminal_prior[2]=prev_ux; terminal_prior[3]=prev_uy; terminal_prior[4]=prev_uz;
                terminal_prior[5]=prev_vx; terminal_prior[6]=prev_vy; terminal_prior[7]=prev_vz;
                terminal_prior[8]=prev_wx; terminal_prior[9]=prev_wy; terminal_prior[10]=prev_wz;
            }
            prev_d = d;
            prev_G = G;
            prev_g = g;
            prev_mx = xy_opac[t * 3 + 0];
            prev_my = xy_opac[t * 3 + 1];
            prev_ux = ux; prev_uy = uy; prev_uz = uz;
            prev_vx = vx; prev_vy = vy; prev_vz = vz;
            prev_wx = wx; prev_wy = wy; prev_wz = wz;
            prev_has_previous = have_prev;
            have_prev = true;
            T = next_T;
        }
        __syncthreads();
    }
    if (terminal_surface && !have_depth && have_prev) {
        const bool split_terminal = coherent_midpoint && terminal_prior_g >= 0;
        const float terminal_gd = split_terminal ? .5f*gd : gd;
        if (split_terminal) add_depth_grad(
            terminal_prior_g, px, py, terminal_prior[0], terminal_prior[1],
            terminal_prior[2], terminal_prior[3], terminal_prior[4],
            terminal_prior[5], terminal_prior[6], terminal_prior[7],
            terminal_prior[8], terminal_prior[9], terminal_prior[10], .5f*gd, g_ray, g_means);
        if constexpr (CarryDepthVjp) pending_depth_vjp += terminal_gd;
        else add_depth_grad(
            prev_g, px, py, prev_mx, prev_my,
            prev_ux, prev_uy, prev_uz, prev_vx, prev_vy, prev_vz,
            prev_wx, prev_wy, prev_wz, terminal_gd, g_ray, g_means);
    }
    if constexpr (CarryDepthVjp) {
        // Also flush at opacity termination or the end of the tile list.
        if (have_prev) {
            add_depth_grad(
                prev_g, px, py, prev_mx, prev_my,
                prev_ux, prev_uy, prev_uz, prev_vx, prev_vy, prev_vz,
                prev_wx, prev_wy, prev_wz, pending_depth_vjp, g_ray, g_means);
        }
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
    double far_k,
    bool carry_depth_vjp,
    bool opacity_independent,
    bool front_convergence,
    bool terminal_surface,
    bool coherent_midpoint) {
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
    auto launch = [&](auto carry_tag) {
    rasterize_unbiased_aux_bwd_kernel<decltype(carry_tag)::value><<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
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
        opacity_independent,
        front_convergence,
        terminal_surface,
        coherent_midpoint,
        g_depth.contiguous().data_ptr<float>(),
        g_conv.contiguous().data_ptr<float>(),
        g_ray.data_ptr<float>(),
        g_means.data_ptr<float>(),
        g_opac.data_ptr<float>());
    };
    if(carry_depth_vjp) launch(std::true_type{});
    else launch(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_means, g_ray, g_opac};
}

// Optional physical ray-depth order. Compact every finite hit from the
// original RGB-accepted prefix; no per-pixel hit limit or truncated sort.
namespace {
template <bool Fill>
__global__ void collect_ray_order_hits(
    const float* means, const float* rays, const float* opac,
    const int32_t* offsets, const int32_t* ids, int nids,
    int width, int height, int tile, int tw, int th,
    int64_t* counts, const int64_t* prefix, int64_t* keys, int64_t* values) {
    int pix=blockIdx.x*blockDim.x+threadIdx.x;
    if (pix>=width*height) return;
    int x=pix%width,y=pix/width,tid=(y/tile)*tw+x/tile;
    int begin=offsets[tid],end=(tid+1<tw*th)?offsets[tid+1]:nids;
    float px=x+.5f,py=y+.5f,T=1.f;
    int64_t n=0,base=Fill?prefix[pix]:0;
    for (int j=begin;j<end;++j) {
        int g=ids[j];const float* m=rays+g*9;
        float hux=px*m[6]-m[0],huy=px*m[7]-m[1],huz=px*m[8]-m[2];
        float hvx=py*m[6]-m[3],hvy=py*m[7]-m[4],hvz=py*m[8]-m[5];
        float hx=huy*hvz-huz*hvy,hy=huz*hvx-hux*hvz,hz=hux*hvy-huy*hvx;
        if (hz==0.f) continue;
        float u=hx/hz,v=hy/hz,rho3=u*u+v*v;
        float dx=means[2*g]-px,dy=means[2*g+1]-py;
        float rho2=2.f*(dx*dx+dy*dy),rho=rho3<rho2?rho3:rho2;
        float G=expf(-.5f*rho),alpha=fminf(.999f,opac[g]*G);
        if (rho<0.f || alpha<kAlphaThreshold) continue;
        float next=T*(1.f-alpha);
        if (next<=1e-4f) break;
        float d=rho3<=rho2?u*m[6]+v*m[7]+m[8]:m[8];
        T=next;
        if (!(isfinite(G)&&isfinite(d))) continue;
        if constexpr (Fill) {
            uint32_t bits=__float_as_uint(d);
            uint32_t ordered=(bits&0x80000000u)?~bits:(bits^0x80000000u);
            keys[base+n]=static_cast<int64_t>((static_cast<uint64_t>(pix)<<32)|ordered);
            values[base+n]=static_cast<int64_t>((static_cast<uint64_t>(g)<<32)|__float_as_uint(G));
        }
        ++n;
    }
    if constexpr (!Fill) counts[pix+1]=n;
}

__device__ float ray_order_depth(int64_t key) {
    uint32_t ordered=static_cast<uint32_t>(key);
    uint32_t bits=(ordered&0x80000000u)?(ordered^0x80000000u):~ordered;
    return __uint_as_float(bits);
}

__global__ void reduce_ray_order_hits(
    const int64_t* keys,const int64_t* values,const int64_t* prefix,
    const float* opac,int pixels,float eps,float threshold,float quarter,bool surface_midpoint,bool terminal_surface,
    float* depth,bool* valid,float* conv,int32_t* selected) {
    int pix=blockIdx.x*blockDim.x+threadIdx.x;if (pix>=pixels) return;
    float cumulative=0.f,loss=0.f,prior_d=0.f,prior_G=0.f;bool have=false,have_prior=false;
    for (int64_t j=prefix[pix];j<prefix[pix+1];++j) {
        int g=static_cast<uint64_t>(values[j])>>32;
        float G=__uint_as_float(static_cast<uint32_t>(values[j]));float d=ray_order_depth(keys[j]);
        float contribution=(opac[g]+eps)*G;
        if (isfinite(contribution)) cumulative+=contribution;
        if (!have&&((isfinite(cumulative)&&cumulative>threshold)||terminal_surface)) {depth[pix]=surface_midpoint&&have_prior?0.5f*(d+prior_d):d;selected[pix]=g;have=isfinite(cumulative)&&cumulative>threshold;}
        float delta=d-prior_d;
        if (have_prior&&isfinite(delta)&&fabsf(delta)<=quarter) loss+=fminf(G,prior_G)*delta*delta;
        prior_d=d;prior_G=G;have_prior=true;
    }
    valid[pix]=have||(terminal_surface&&have_prior);conv[pix]=loss;
}

__device__ void emit_ray_order_vjp(int g,int pix,int width,float grad,
    const float* means,const float* rays,float* g_ray,float* g_means) {
    const float* m=rays+g*9;
    add_depth_grad(g,pix%width+.5f,pix/width+.5f,means[2*g],means[2*g+1],
        m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8],grad,g_ray,g_means);
}

__global__ void ray_order_hits_backward(
    const float* means,const float* rays,const float* opac,
    const int64_t* keys,const int64_t* values,const int64_t* prefix,
    const float* g_depth,const float* g_conv,int width,int pixels,
    float eps,float threshold,float quarter,float far_k,bool terminal_surface,float* g_ray,float* g_means) {
    int pix=blockIdx.x*blockDim.x+threadIdx.x;if (pix>=pixels) return;
    if (g_depth[pix]==0.f&&g_conv[pix]==0.f) return;
    float cumulative=0.f,prior_d=0.f,prior_G=0.f;int prior_g=-1;bool have=false;
    for (int64_t j=prefix[pix];j<prefix[pix+1];++j) {
        int g=static_cast<uint64_t>(values[j])>>32;
        float G=__uint_as_float(static_cast<uint32_t>(values[j]));float d=ray_order_depth(keys[j]);
        float contribution=(opac[g]+eps)*G;
        if (isfinite(contribution)) cumulative+=contribution;
        float grad=0.f;
        if (!have&&isfinite(cumulative)&&cumulative>threshold) {grad=g_depth[pix];have=true;}
        float delta=d-prior_d;
        if (prior_g>=0&&isfinite(delta)&&fabsf(delta)<=quarter) {
            float common=2.f*delta*g_conv[pix]*fminf(G,prior_G);
            grad+=far_k*common;
            emit_ray_order_vjp(prior_g,pix,width,-common,means,rays,g_ray,g_means);
        }
        emit_ray_order_vjp(g,pix,width,grad,means,rays,g_ray,g_means);
        prior_d=d;prior_G=G;prior_g=g;
    }
    if (terminal_surface&&!have&&prior_g>=0)
        emit_ray_order_vjp(prior_g,pix,width,g_depth[pix],means,rays,g_ray,g_means);
}
} // namespace

std::vector<torch::Tensor> rasterize_unbiased_aux_ray_order_cuda(
    torch::Tensor means,torch::Tensor rays,torch::Tensor opac,
    torch::Tensor offsets,torch::Tensor ids,int64_t width,int64_t height,
    int64_t tile,double eps,double threshold,double radius,bool surface_midpoint,bool terminal_surface) {
    TORCH_CHECK(means.is_cuda()&&means.scalar_type()==torch::kFloat32&&
        rays.is_cuda()&&rays.scalar_type()==torch::kFloat32&&opac.is_cuda()&&opac.scalar_type()==torch::kFloat32,
        "ray-order auxiliaries require float32 CUDA inputs");
    TORCH_CHECK(offsets.size(0)==1&&offsets.scalar_type()==torch::kInt32&&ids.scalar_type()==torch::kInt32,
        "ray-order auxiliaries require one image and int32 tile metadata");
    TORCH_CHECK(width>0&&height>0&&tile>0&&width*height<INT32_MAX&&ids.numel()<INT32_MAX,
        "ray-order auxiliary dimensions exceed supported integer indexing");
    const c10::cuda::CUDAGuard guard(means.device());
    const auto stream=at::cuda::getCurrentCUDAStream();int pixels=width*height,blocks=(pixels+255)/256;
    auto counts=torch::zeros({pixels+1},means.options().dtype(torch::kInt64));
    collect_ray_order_hits<false><<<blocks,256,0,stream>>>(means.data_ptr<float>(),rays.data_ptr<float>(),opac.data_ptr<float>(),
        offsets.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),ids.numel(),width,height,tile,offsets.size(2),offsets.size(1),
        counts.data_ptr<int64_t>(),nullptr,nullptr,nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto prefix=counts.cumsum(0);int64_t n=prefix[-1].item<int64_t>();
    TORCH_CHECK(n<INT32_MAX,"ray-order hit count exceeds native radix-sort integer indexing; no hits were truncated");
    auto keys=torch::empty({n},counts.options()),values=torch::empty_like(keys);
    auto sorted_keys=torch::empty_like(keys),sorted_values=torch::empty_like(keys);
    if (n) {
        collect_ray_order_hits<true><<<blocks,256,0,stream>>>(means.data_ptr<float>(),rays.data_ptr<float>(),opac.data_ptr<float>(),
            offsets.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),ids.numel(),width,height,tile,offsets.size(2),offsets.size(1),
            nullptr,prefix.data_ptr<int64_t>(),keys.data_ptr<int64_t>(),values.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        size_t bytes=0;
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,bytes,keys.data_ptr<int64_t>(),sorted_keys.data_ptr<int64_t>(),
            values.data_ptr<int64_t>(),sorted_values.data_ptr<int64_t>(),static_cast<int>(n),0,64,stream));
        auto temporary=torch::empty({static_cast<int64_t>(bytes)},means.options().dtype(torch::kUInt8));
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temporary.data_ptr(),bytes,keys.data_ptr<int64_t>(),sorted_keys.data_ptr<int64_t>(),
            values.data_ptr<int64_t>(),sorted_values.data_ptr<int64_t>(),static_cast<int>(n),0,64,stream));
    }
    auto depth=torch::zeros({1,height,width},means.options()),conv=torch::zeros_like(depth);
    auto valid=torch::zeros({1,height,width},means.options().dtype(torch::kBool));
    auto selected=torch::full({1,height,width},-1,means.options().dtype(torch::kInt32));
    reduce_ray_order_hits<<<blocks,256,0,stream>>>(sorted_keys.data_ptr<int64_t>(),sorted_values.data_ptr<int64_t>(),prefix.data_ptr<int64_t>(),
        opac.data_ptr<float>(),pixels,eps,threshold,radius/4.,surface_midpoint,terminal_surface,depth.data_ptr<float>(),valid.data_ptr<bool>(),conv.data_ptr<float>(),selected.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {depth,valid,conv,selected,sorted_keys,sorted_values,prefix};
}

std::vector<torch::Tensor> rasterize_unbiased_aux_ray_order_backward_cuda(
    torch::Tensor means,torch::Tensor rays,torch::Tensor opac,
    torch::Tensor keys,torch::Tensor values,torch::Tensor prefix,
    torch::Tensor g_depth,torch::Tensor g_conv,int64_t width,int64_t height,
    double eps,double threshold,double radius,double far_k,bool terminal_surface) {
    const c10::cuda::CUDAGuard guard(means.device());
    auto g_means=torch::zeros_like(means),g_ray=torch::zeros_like(rays),g_opac=torch::zeros_like(opac);
    int pixels=width*height;
    ray_order_hits_backward<<<(pixels+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<float>(),rays.data_ptr<float>(),opac.data_ptr<float>(),keys.data_ptr<int64_t>(),values.data_ptr<int64_t>(),
        prefix.data_ptr<int64_t>(),g_depth.data_ptr<float>(),g_conv.data_ptr<float>(),width,pixels,eps,threshold,radius/4.,far_k,terminal_surface,
        g_ray.data_ptr<float>(),g_means.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_means,g_ray,g_opac};
}
