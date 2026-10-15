// Independently written CUDA for remaining SPEC operators.
// Apache-2.0. No Unbiased_Surfel or PGSR source.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

int64_t blocks_for(int64_t n) { return (n + 255) / 256; }

template <typename T>
__device__ inline bool finite_t(T x) {
  return isfinite(static_cast<double>(x));
}

template <typename scalar_t>
__global__ void contribution_kernel(
    const scalar_t* __restrict__ opacities,
    const scalar_t* __restrict__ weights,
    const bool* __restrict__ valid,
    scalar_t* __restrict__ contrib,
    scalar_t eps,
    int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  scalar_t raw = (opacities[i] + eps) * weights[i];
  bool keep = valid[i] && finite_t(raw);
  contrib[i] = keep ? raw : scalar_t(0);
}

__global__ void starts_kernel(const int64_t* __restrict__ seg, bool* __restrict__ starts, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  starts[i] = (i == 0) || (seg[i] != seg[i - 1]);
}

template <typename scalar_t>
__global__ void first_crossing_kernel(
    const scalar_t* __restrict__ cumulative,
    const bool* __restrict__ valid,
    const int64_t* __restrict__ seg,
    int64_t* __restrict__ selected,
    scalar_t threshold,
    int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  if (valid[i] && finite_t(cumulative[i]) && cumulative[i] > threshold) {
    unsigned long long* slot =
        reinterpret_cast<unsigned long long*>(selected + seg[i]);
    atomicMin(slot, static_cast<unsigned long long>(i));
  }
}

template <typename scalar_t>
__global__ void convergence_pairs_kernel(
    const scalar_t* __restrict__ depths,
    const scalar_t* __restrict__ weights,
    const int64_t* __restrict__ seg,
    const bool* __restrict__ valid,
    scalar_t* __restrict__ per_pair,
    bool* __restrict__ pair_mask,
    int64_t* __restrict__ prev_idx,
    int64_t* __restrict__ curr_idx,
    scalar_t radius_quarter,
    int64_t n_pairs) {
  int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_pairs) return;
  int64_t i = p + 1;
  prev_idx[p] = i - 1;
  curr_idx[p] = i;
  bool same = seg[i] == seg[i - 1];
  scalar_t delta = depths[i] - depths[i - 1];
  bool ok = same && valid[i] && valid[i - 1] && finite_t(delta) &&
            fabs(static_cast<double>(delta)) <= static_cast<double>(radius_quarter);
  pair_mask[p] = ok;
  if (!ok) {
    per_pair[p] = 0;
    return;
  }
  scalar_t w = weights[i] < weights[i - 1] ? weights[i] : weights[i - 1];
  per_pair[p] = w * delta * delta;
}

template <typename scalar_t>
__global__ void zncc_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    bool* __restrict__ valid,
    int64_t N,
    int64_t P,
    scalar_t var_eps) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const scalar_t* pa = a + n * P;
  const scalar_t* pb = b + n * P;
  scalar_t mean_a = 0, mean_b = 0;
  bool ok = true;
  for (int64_t k = 0; k < P; ++k) {
    ok = ok && finite_t(pa[k]) && finite_t(pb[k]);
    mean_a += pa[k];
    mean_b += pb[k];
  }
  scalar_t invP = static_cast<scalar_t>(1) / static_cast<scalar_t>(P);
  mean_a *= invP;
  mean_b *= invP;
  scalar_t ea = 0, eb = 0, dot = 0;
  for (int64_t k = 0; k < P; ++k) {
    scalar_t da = pa[k] - mean_a;
    scalar_t db = pb[k] - mean_b;
    ea += da * da;
    eb += db * db;
    dot += da * db;
  }
  ok = ok && (ea > var_eps) && (eb > var_eps);
  valid[n] = ok;
  if (!ok) {
    out[n] = 0;
    return;
  }
  scalar_t den = sqrt(ea * eb);
  if (den < var_eps) den = var_eps;
  scalar_t c = dot / den;
  if (c > 1) c = 1;
  if (c < -1) c = -1;
  out[n] = c;
}

template <typename scalar_t>
__global__ void depth_to_points_kernel(
    const scalar_t* __restrict__ depth,
    const bool* __restrict__ valid,
    const scalar_t* __restrict__ c2w,
    const scalar_t* __restrict__ K,
    scalar_t* __restrict__ points,
    bool* __restrict__ point_valid,
    int64_t N,
    int64_t H,
    int64_t W) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t pixels = H * W;
  if (idx >= N * pixels) return;
  int64_t n = idx / pixels;
  int64_t pix = idx % pixels;
  int64_t y = pix / W;
  int64_t x = pix % W;
  const scalar_t* Kn = K + n * 9;
  const scalar_t* Rn = c2w + n * 16;
  scalar_t fx = Kn[0], fy = Kn[4], cx = Kn[2], cy = Kn[5];
  scalar_t z = depth[idx];
  bool ok = valid[idx] && finite_t(z);
  scalar_t camx = (static_cast<scalar_t>(x) - cx + static_cast<scalar_t>(0.5)) / fx * z;
  scalar_t camy = (static_cast<scalar_t>(y) - cy + static_cast<scalar_t>(0.5)) / fy * z;
  scalar_t camz = z;
  scalar_t wx = Rn[0] * camx + Rn[1] * camy + Rn[2] * camz + Rn[3];
  scalar_t wy = Rn[4] * camx + Rn[5] * camy + Rn[6] * camz + Rn[7];
  scalar_t wz = Rn[8] * camx + Rn[9] * camy + Rn[10] * camz + Rn[11];
  ok = ok && finite_t(wx) && finite_t(wy) && finite_t(wz);
  points[idx * 3 + 0] = ok ? wx : scalar_t(0);
  points[idx * 3 + 1] = ok ? wy : scalar_t(0);
  points[idx * 3 + 2] = ok ? wz : scalar_t(0);
  point_valid[idx] = ok;
}

}  // namespace

std::vector<torch::Tensor> corrected_depth_forward_cuda(
    torch::Tensor depths,
    torch::Tensor opacities,
    torch::Tensor weights,
    torch::Tensor segment_ids,
    torch::Tensor valid,
    int64_t num_segments,
    double opacity_epsilon,
    double threshold) {
  TORCH_CHECK(depths.is_cuda(), "corrected_depth CUDA requires CUDA tensors");
  const auto M = depths.size(0);
  auto dense_depth = torch::zeros({num_segments}, depths.options());
  auto dense_valid = torch::zeros({num_segments}, depths.options().dtype(torch::kBool));
  auto selected = torch::full({num_segments}, M, segment_ids.options());
  auto contrib = torch::empty({M}, depths.options());
  auto cumulative = torch::empty({M}, depths.options());
  if (M == 0) {
    selected.fill_(-1);
    return {dense_depth, dense_valid, selected, contrib, cumulative};
  }
  const c10::cuda::CUDAGuard guard(depths.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(depths.scalar_type(), "contrib", [&] {
    contribution_kernel<scalar_t><<<blocks_for(M), 256, 0, stream>>>(
        opacities.data_ptr<scalar_t>(),
        weights.data_ptr<scalar_t>(),
        valid.data_ptr<bool>(),
        contrib.data_ptr<scalar_t>(),
        static_cast<scalar_t>(opacity_epsilon),
        M);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto cumulative_global = contrib.cumsum(0);
  auto starts = torch::empty({M}, torch::dtype(torch::kBool).device(depths.device()));
  starts_kernel<<<blocks_for(M), 256, 0, stream>>>(
      segment_ids.data_ptr<int64_t>(), starts.data_ptr<bool>(), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto local_index = starts.to(torch::kLong).cumsum(0) - 1;
  auto start_pos = torch::nonzero(starts).squeeze(-1);
  auto offsets = cumulative_global.index({start_pos}) - contrib.index({start_pos});
  cumulative = cumulative_global - offsets.index({local_index});
  AT_DISPATCH_FLOATING_TYPES(depths.scalar_type(), "crossing", [&] {
    first_crossing_kernel<scalar_t><<<blocks_for(M), 256, 0, stream>>>(
        cumulative.data_ptr<scalar_t>(),
        valid.data_ptr<bool>(),
        segment_ids.data_ptr<int64_t>(),
        selected.data_ptr<int64_t>(),
        static_cast<scalar_t>(threshold),
        M);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dense_valid = selected < M;
  auto safe = selected.clamp_max(M - 1);
  dense_depth = torch::where(dense_valid, depths.index({safe}), torch::zeros_like(dense_depth));
  selected = torch::where(dense_valid, selected, torch::full_like(selected, -1));
  return {dense_depth, dense_valid, selected, contrib, cumulative};
}

std::vector<torch::Tensor> convergence_pairs_cuda(
    torch::Tensor depths,
    torch::Tensor weights,
    torch::Tensor segment_ids,
    torch::Tensor valid,
    double scene_radius) {
  const auto M = depths.size(0);
  int64_t n_pairs = M > 0 ? M - 1 : 0;
  auto per_pair = torch::empty({n_pairs}, depths.options());
  auto pair_mask = torch::empty({n_pairs}, torch::dtype(torch::kBool).device(depths.device()));
  auto prev_idx = torch::empty({n_pairs}, segment_ids.options());
  auto curr_idx = torch::empty({n_pairs}, segment_ids.options());
  if (n_pairs == 0) {
    return {prev_idx, curr_idx, pair_mask, per_pair};
  }
  const c10::cuda::CUDAGuard guard(depths.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(depths.scalar_type(), "conv_pairs", [&] {
    convergence_pairs_kernel<scalar_t><<<blocks_for(n_pairs), 256, 0, stream>>>(
        depths.data_ptr<scalar_t>(),
        weights.data_ptr<scalar_t>(),
        segment_ids.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),
        per_pair.data_ptr<scalar_t>(),
        pair_mask.data_ptr<bool>(),
        prev_idx.data_ptr<int64_t>(),
        curr_idx.data_ptr<int64_t>(),
        static_cast<scalar_t>(scene_radius / 4.0),
        n_pairs);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {prev_idx, curr_idx, pair_mask, per_pair};
}

std::vector<torch::Tensor> zncc_forward_cuda(
    torch::Tensor reference_patches,
    torch::Tensor neighbor_patches,
    double variance_epsilon) {
  TORCH_CHECK(reference_patches.is_cuda(), "zncc CUDA requires CUDA tensors");
  const auto N = reference_patches.size(0);
  const auto P = reference_patches.size(1);
  auto out = torch::empty({N}, reference_patches.options());
  auto valid = torch::empty({N}, torch::dtype(torch::kBool).device(reference_patches.device()));
  if (N == 0) {
    return {out, valid};
  }
  const c10::cuda::CUDAGuard guard(reference_patches.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(reference_patches.scalar_type(), "zncc", [&] {
    zncc_kernel<scalar_t><<<blocks_for(N), 256, 0, stream>>>(
        reference_patches.data_ptr<scalar_t>(),
        neighbor_patches.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        valid.data_ptr<bool>(),
        N,
        P,
        static_cast<scalar_t>(variance_epsilon));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, valid};
}

std::vector<torch::Tensor> depth_to_points_cuda(
    torch::Tensor depth,
    torch::Tensor valid,
    torch::Tensor camera_to_world,
    torch::Tensor intrinsics) {
  TORCH_CHECK(depth.is_cuda(), "depth_to_points CUDA requires CUDA tensors");
  auto H = depth.size(-2);
  auto W = depth.size(-1);
  auto N = depth.numel() / (H * W);
  auto depth_flat = depth.reshape({N, H, W}).contiguous();
  auto valid_flat = valid.reshape({N, H, W}).contiguous();
  auto c2w = camera_to_world.reshape({N, 4, 4}).contiguous();
  auto K = intrinsics.reshape({N, 3, 3}).contiguous();
  auto points = torch::empty({N, H, W, 3}, depth.options());
  auto point_valid = torch::empty({N, H, W}, torch::dtype(torch::kBool).device(depth.device()));
  int64_t total = N * H * W;
  if (total == 0) {
    auto point_shape = depth.sizes().vec();
    point_shape.push_back(3);
    return {torch::empty(point_shape, depth.options()), torch::empty_like(valid)};
  }
  const c10::cuda::CUDAGuard guard(depth.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "d2p", [&] {
    depth_to_points_kernel<scalar_t><<<blocks_for(total), 256, 0, stream>>>(
        depth_flat.data_ptr<scalar_t>(),
        valid_flat.data_ptr<bool>(),
        c2w.data_ptr<scalar_t>(),
        K.data_ptr<scalar_t>(),
        points.data_ptr<scalar_t>(),
        point_valid.data_ptr<bool>(),
        N,
        H,
        W);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto point_shape = depth.sizes().vec();
  point_shape.push_back(3);
  return {points.reshape(point_shape), point_valid.reshape_as(valid)};
}

template <typename scalar_t>
__global__ void zncc_backward_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ g_corr,
    const bool* __restrict__ valid,
    scalar_t* __restrict__ g_a,
    scalar_t* __restrict__ g_b,
    int64_t N,
    int64_t P,
    scalar_t var_eps) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  scalar_t* ga = g_a + n * P;
  scalar_t* gb = g_b + n * P;
  for (int64_t k = 0; k < P; ++k) {
    ga[k] = 0;
    gb[k] = 0;
  }
  if (!valid[n]) return;
  const scalar_t* pa = a + n * P;
  const scalar_t* pb = b + n * P;
  scalar_t mean_a = 0, mean_b = 0;
  for (int64_t k = 0; k < P; ++k) {
    mean_a += pa[k];
    mean_b += pb[k];
  }
  scalar_t invP = static_cast<scalar_t>(1) / static_cast<scalar_t>(P);
  mean_a *= invP;
  mean_b *= invP;
  scalar_t ea = 0, eb = 0, dot = 0;
  for (int64_t k = 0; k < P; ++k) {
    scalar_t da = pa[k] - mean_a;
    scalar_t db = pb[k] - mean_b;
    ea += da * da;
    eb += db * db;
    dot += da * db;
  }
  scalar_t den = sqrt(ea * eb);
  if (den < var_eps) den = var_eps;
  scalar_t raw = dot / den;
  if (raw > 1 || raw < -1) return;
  scalar_t gc = g_corr[n];
  scalar_t g_dot = gc / den;
  scalar_t g_s = -gc * raw / den;
  scalar_t g_ea = g_s * eb / den;
  scalar_t g_eb = g_s * ea / den;
  scalar_t sum_ga = 0, sum_gb = 0;
  for (int64_t k = 0; k < P; ++k) {
    scalar_t da = pa[k] - mean_a;
    scalar_t db = pb[k] - mean_b;
    scalar_t dda = g_dot * db + static_cast<scalar_t>(2) * g_ea * da;
    scalar_t ddb = g_dot * da + static_cast<scalar_t>(2) * g_eb * db;
    ga[k] = dda;
    gb[k] = ddb;
    sum_ga += dda;
    sum_gb += ddb;
  }
  scalar_t mean_ga = sum_ga * invP;
  scalar_t mean_gb = sum_gb * invP;
  for (int64_t k = 0; k < P; ++k) {
    ga[k] -= mean_ga;
    gb[k] -= mean_gb;
  }
}

template <typename scalar_t>
__global__ void depth_to_points_backward_kernel(
    const scalar_t* __restrict__ depth,
    const bool* __restrict__ point_valid,
    const scalar_t* __restrict__ c2w,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ g_points,
    scalar_t* __restrict__ g_depth,
    scalar_t* __restrict__ g_c2w,
    scalar_t* __restrict__ g_K,
    int64_t N,
    int64_t H,
    int64_t W) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t pixels = H * W;
  if (idx >= N * pixels) return;
  g_depth[idx] = 0;
  if (!point_valid[idx]) return;
  int64_t n = idx / pixels;
  int64_t pix = idx % pixels;
  int64_t y = pix / W;
  int64_t x = pix % W;
  const scalar_t* Kn = K + n * 9;
  const scalar_t* Rn = c2w + n * 16;
  scalar_t fx = Kn[0], fy = Kn[4], cx = Kn[2], cy = Kn[5];
  scalar_t z = depth[idx];
  scalar_t dirx = (static_cast<scalar_t>(x) - cx + static_cast<scalar_t>(0.5)) / fx;
  scalar_t diry = (static_cast<scalar_t>(y) - cy + static_cast<scalar_t>(0.5)) / fy;
  scalar_t camx = dirx * z, camy = diry * z, camz = z;
  scalar_t gwx = g_points[idx * 3 + 0];
  scalar_t gwy = g_points[idx * 3 + 1];
  scalar_t gwz = g_points[idx * 3 + 2];
  scalar_t gcamx = Rn[0] * gwx + Rn[4] * gwy + Rn[8] * gwz;
  scalar_t gcamy = Rn[1] * gwx + Rn[5] * gwy + Rn[9] * gwz;
  scalar_t gcamz = Rn[2] * gwx + Rn[6] * gwy + Rn[10] * gwz;
  g_depth[idx] = gcamx * dirx + gcamy * diry + gcamz;
  scalar_t gdirx = gcamx * z;
  scalar_t gdiry = gcamy * z;
  scalar_t* gR = g_c2w + n * 16;
  atomicAdd(gR + 0, gwx * camx);
  atomicAdd(gR + 1, gwx * camy);
  atomicAdd(gR + 2, gwx * camz);
  atomicAdd(gR + 3, gwx);
  atomicAdd(gR + 4, gwy * camx);
  atomicAdd(gR + 5, gwy * camy);
  atomicAdd(gR + 6, gwy * camz);
  atomicAdd(gR + 7, gwy);
  atomicAdd(gR + 8, gwz * camx);
  atomicAdd(gR + 9, gwz * camy);
  atomicAdd(gR + 10, gwz * camz);
  atomicAdd(gR + 11, gwz);
  scalar_t* gKn = g_K + n * 9;
  scalar_t u = static_cast<scalar_t>(x) - cx + static_cast<scalar_t>(0.5);
  scalar_t v = static_cast<scalar_t>(y) - cy + static_cast<scalar_t>(0.5);
  atomicAdd(gKn + 0, gdirx * (-u / (fx * fx)));
  atomicAdd(gKn + 2, gdirx * (-static_cast<scalar_t>(1) / fx));
  atomicAdd(gKn + 4, gdiry * (-v / (fy * fy)));
  atomicAdd(gKn + 5, gdiry * (-static_cast<scalar_t>(1) / fy));
}

std::vector<torch::Tensor> zncc_backward_cuda(
    torch::Tensor reference_patches,
    torch::Tensor neighbor_patches,
    torch::Tensor grad_corr,
    torch::Tensor valid,
    double variance_epsilon) {
  const auto N = reference_patches.size(0);
  const auto P = reference_patches.size(1);
  auto g_a = torch::zeros_like(reference_patches);
  auto g_b = torch::zeros_like(neighbor_patches);
  if (N == 0) return {g_a, g_b};
  const c10::cuda::CUDAGuard guard(reference_patches.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(reference_patches.scalar_type(), "zncc_bwd", [&] {
    zncc_backward_kernel<scalar_t><<<(N + 255) / 256, 256, 0, stream>>>(
        reference_patches.data_ptr<scalar_t>(),
        neighbor_patches.data_ptr<scalar_t>(),
        grad_corr.data_ptr<scalar_t>(),
        valid.data_ptr<bool>(),
        g_a.data_ptr<scalar_t>(),
        g_b.data_ptr<scalar_t>(),
        N,
        P,
        static_cast<scalar_t>(variance_epsilon));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {g_a, g_b};
}

std::vector<torch::Tensor> depth_to_points_backward_cuda(
    torch::Tensor depth,
    torch::Tensor point_valid,
    torch::Tensor camera_to_world,
    torch::Tensor intrinsics,
    torch::Tensor grad_points) {
  auto H = depth.size(-2);
  auto W = depth.size(-1);
  auto N = depth.numel() / (H * W);
  auto depth_flat = depth.reshape({N, H, W}).contiguous();
  auto valid_flat = point_valid.reshape({N, H, W}).contiguous();
  auto c2w = camera_to_world.reshape({N, 4, 4}).contiguous();
  auto K = intrinsics.reshape({N, 3, 3}).contiguous();
  auto g_pts = grad_points.reshape({N, H, W, 3}).contiguous();
  auto g_depth = torch::zeros_like(depth_flat);
  auto g_c2w = torch::zeros_like(c2w);
  auto g_K = torch::zeros_like(K);
  int64_t total = N * H * W;
  if (total == 0) {
    return {g_depth.reshape_as(depth), g_c2w.reshape_as(camera_to_world), g_K.reshape_as(intrinsics)};
  }
  const c10::cuda::CUDAGuard guard(depth.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "d2p_bwd", [&] {
    depth_to_points_backward_kernel<scalar_t><<<(total + 255) / 256, 256, 0, stream>>>(
        depth_flat.data_ptr<scalar_t>(),
        valid_flat.data_ptr<bool>(),
        c2w.data_ptr<scalar_t>(),
        K.data_ptr<scalar_t>(),
        g_pts.data_ptr<scalar_t>(),
        g_depth.data_ptr<scalar_t>(),
        g_c2w.data_ptr<scalar_t>(),
        g_K.data_ptr<scalar_t>(),
        N,
        H,
        W);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {g_depth.reshape_as(depth), g_c2w.reshape_as(camera_to_world), g_K.reshape_as(intrinsics)};
}
