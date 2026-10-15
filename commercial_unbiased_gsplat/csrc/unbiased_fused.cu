// Independently written fused ray/surfel depth kernels.
// Equations: SPEC.md only. Apache-2.0. No Unbiased_Surfel or PGSR source.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

namespace {

template <typename T>
__device__ inline bool is_finite_t(T x) {
  return isfinite(static_cast<double>(x));
}

template <typename scalar_t>
__global__ void ray_splat_forward_kernel(
    const scalar_t* __restrict__ ray_t,
    const scalar_t* __restrict__ means,
    const int64_t* __restrict__ pixel_ids,
    scalar_t* __restrict__ out_weight,
    scalar_t* __restrict__ out_depth,
    scalar_t* __restrict__ out_rho3,
    scalar_t* __restrict__ out_rho2,
    scalar_t* __restrict__ out_uv,
    bool* __restrict__ out_plane,
    bool* __restrict__ out_valid,
    int64_t M,
    int64_t W,
    scalar_t hz_epsilon) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= M) {
    return;
  }
  const int64_t pid = pixel_ids[i];
  const scalar_t px = static_cast<scalar_t>(pid % W) + static_cast<scalar_t>(0.5);
  const scalar_t py = static_cast<scalar_t>(pid / W) + static_cast<scalar_t>(0.5);
  const scalar_t* R = ray_t + i * 9;
  const scalar_t ux = R[0], uy = R[1], uz = R[2];
  const scalar_t vx = R[3], vy = R[4], vz = R[5];
  const scalar_t wx = R[6], wy = R[7], wz = R[8];
  const scalar_t hux = px * wx - ux, huy = px * wy - uy, huz = px * wz - uz;
  const scalar_t hvx = py * wx - vx, hvy = py * wy - vy, hvz = py * wz - vz;
  const scalar_t hx = huy * hvz - huz * hvy;
  const scalar_t hy = huz * hvx - hux * hvz;
  const scalar_t hz = hux * hvy - huy * hvx;
  bool ok = is_finite_t(hx) && is_finite_t(hy) && is_finite_t(hz) &&
            (fabs(static_cast<double>(hz)) > static_cast<double>(hz_epsilon));
  scalar_t sx = 0, sy = 0, rho3 = 0, rho2 = 0, depth = 0, weight = 0;
  bool plane = false;
  if (ok) {
    sx = hx / hz;
    sy = hy / hz;
    rho3 = sx * sx + sy * sy;
    const scalar_t dx = means[i * 2] - px;
    const scalar_t dy = means[i * 2 + 1] - py;
    rho2 = static_cast<scalar_t>(2) * (dx * dx + dy * dy);
    const scalar_t rho = rho3 < rho2 ? rho3 : rho2;
    plane = rho3 <= rho2;
    const scalar_t dplane = sx * wx + sy * wy + wz;
    depth = plane ? dplane : wz;
    ok = is_finite_t(rho3) && is_finite_t(rho2) && is_finite_t(rho) && is_finite_t(depth) &&
         rho >= static_cast<scalar_t>(0);
    if (ok) {
      weight = exp(-rho / static_cast<scalar_t>(2));
    }
  }
  if (!ok) {
    sx = 0;
    sy = 0;
    rho3 = 0;
    rho2 = 0;
    depth = 0;
    weight = 0;
    plane = false;
  }
  out_weight[i] = weight;
  out_depth[i] = depth;
  out_rho3[i] = rho3;
  out_rho2[i] = rho2;
  out_uv[i * 2] = sx;
  out_uv[i * 2 + 1] = sy;
  out_plane[i] = plane;
  out_valid[i] = ok;
}

template <typename scalar_t>
__global__ void ray_splat_backward_kernel(
    const scalar_t* __restrict__ ray_t,
    const scalar_t* __restrict__ means,
    const int64_t* __restrict__ pixel_ids,
    const scalar_t* __restrict__ local_uv,
    const scalar_t* __restrict__ weight,
    const bool* __restrict__ valid,
    const bool* __restrict__ plane,
    const scalar_t* __restrict__ grad_weight,
    const scalar_t* __restrict__ grad_depth,
    const scalar_t* __restrict__ grad_rho3,
    const scalar_t* __restrict__ grad_rho2,
    const scalar_t* __restrict__ grad_uv,
    scalar_t* __restrict__ grad_ray,
    scalar_t* __restrict__ grad_means,
    int64_t M,
    int64_t W) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= M) {
    return;
  }
  scalar_t* gR = grad_ray + i * 9;
  for (int k = 0; k < 9; ++k) {
    gR[k] = 0;
  }
  grad_means[i * 2] = 0;
  grad_means[i * 2 + 1] = 0;
  if (!valid[i]) {
    return;
  }

  const int64_t pid = pixel_ids[i];
  const scalar_t px = static_cast<scalar_t>(pid % W) + static_cast<scalar_t>(0.5);
  const scalar_t py = static_cast<scalar_t>(pid / W) + static_cast<scalar_t>(0.5);
  const scalar_t* R = ray_t + i * 9;
  const scalar_t ux = R[0], uy = R[1], uz = R[2];
  const scalar_t vx = R[3], vy = R[4], vz = R[5];
  const scalar_t wx = R[6], wy = R[7], wz = R[8];
  const scalar_t sx = local_uv[i * 2];
  const scalar_t sy = local_uv[i * 2 + 1];
  const scalar_t mx = means[i * 2];
  const scalar_t my = means[i * 2 + 1];
  const scalar_t dx = mx - px;
  const scalar_t dy = my - py;
  const scalar_t rho3 = sx * sx + sy * sy;
  const scalar_t rho2 = static_cast<scalar_t>(2) * (dx * dx + dy * dy);
  const bool use_plane = plane[i];
  const scalar_t G = weight[i];

  scalar_t g_rho3 = grad_rho3[i];
  scalar_t g_rho2 = grad_rho2[i];
  const scalar_t g_G = grad_weight[i];
  const scalar_t g_d = grad_depth[i];
  scalar_t g_sx = grad_uv[i * 2];
  scalar_t g_sy = grad_uv[i * 2 + 1];

  // rho = min(rho3, rho2); Torch minimum sends the equal-case gradient to rho3
  // via (rho3 <= rho2) for the first operand.
  const scalar_t g_rho = g_G * (-G / static_cast<scalar_t>(2));
  if (rho3 <= rho2) {
    g_rho3 += g_rho;
  } else {
    g_rho2 += g_rho;
  }

  if (use_plane) {
    g_sx += g_d * wx;
    g_sy += g_d * wy;
    gR[6] += g_d * sx;
    gR[7] += g_d * sy;
    gR[8] += g_d;
  } else {
    gR[8] += g_d;
  }

  g_sx += g_rho3 * (static_cast<scalar_t>(2) * sx);
  g_sy += g_rho3 * (static_cast<scalar_t>(2) * sy);

  grad_means[i * 2] = g_rho2 * (static_cast<scalar_t>(4) * dx);
  grad_means[i * 2 + 1] = g_rho2 * (static_cast<scalar_t>(4) * dy);

  const scalar_t hux = px * wx - ux, huy = px * wy - uy, huz = px * wz - uz;
  const scalar_t hvx = py * wx - vx, hvy = py * wy - vy, hvz = py * wz - vz;
  const scalar_t hz = hux * hvy - huy * hvx;
  const scalar_t inv_hz = static_cast<scalar_t>(1) / hz;
  const scalar_t g_hx = g_sx * inv_hz;
  const scalar_t g_hy = g_sy * inv_hz;
  const scalar_t g_hz = -(g_sx * sx + g_sy * sy) * inv_hz;

  // h = hu x hv; g_hu = hv x g_h; g_hv = g_h x hu
  const scalar_t g_hux = hvy * g_hz - hvz * g_hy;
  const scalar_t g_huy = hvz * g_hx - hvx * g_hz;
  const scalar_t g_huz = hvx * g_hy - hvy * g_hx;
  const scalar_t g_hvx = g_hy * huz - g_hz * huy;
  const scalar_t g_hvy = g_hz * hux - g_hx * huz;
  const scalar_t g_hvz = g_hx * huy - g_hy * hux;

  gR[0] += -g_hux;
  gR[1] += -g_huy;
  gR[2] += -g_huz;
  gR[3] += -g_hvx;
  gR[4] += -g_hvy;
  gR[5] += -g_hvz;
  gR[6] += px * g_hux + py * g_hvx;
  gR[7] += px * g_huy + py * g_hvy;
  gR[8] += px * g_huz + py * g_hvz;
}

int64_t blocks_for(int64_t M) {
  return (M + 255) / 256;
}

}  // namespace

std::vector<torch::Tensor> ray_splat_forward_cuda(
    torch::Tensor ray_transforms,
    torch::Tensor means2d,
    torch::Tensor pixel_ids,
    int64_t image_width,
    double hz_epsilon) {
  TORCH_CHECK(ray_transforms.is_cuda(), "ray_transforms must be CUDA");
  TORCH_CHECK(means2d.is_cuda() && pixel_ids.is_cuda(), "means2d and pixel_ids must be CUDA");
  TORCH_CHECK(ray_transforms.dim() == 3 && ray_transforms.size(-1) == 3 && ray_transforms.size(-2) == 3,
              "ray_transforms must be [M,3,3]");
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(-1) == 2, "means2d must be [M,2]");
  TORCH_CHECK(pixel_ids.dim() == 1 && pixel_ids.dtype() == torch::kLong, "pixel_ids must be int64 [M]");
  TORCH_CHECK(image_width > 0, "image_width must be positive");
  TORCH_CHECK(ray_transforms.size(0) == means2d.size(0) && ray_transforms.size(0) == pixel_ids.size(0),
              "M mismatch");
  const auto M = ray_transforms.size(0);
  auto opts = ray_transforms.options();
  auto weight = torch::empty({M}, opts);
  auto depth = torch::empty({M}, opts);
  auto rho3 = torch::empty({M}, opts);
  auto rho2 = torch::empty({M}, opts);
  auto local_uv = torch::empty({M, 2}, opts);
  auto plane = torch::empty({M}, ray_transforms.options().dtype(torch::kBool));
  auto valid = torch::empty({M}, ray_transforms.options().dtype(torch::kBool));
  if (M == 0) {
    return {weight, depth, rho3, rho2, local_uv, plane, valid};
  }
  const c10::cuda::CUDAGuard device_guard(ray_transforms.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(ray_transforms.scalar_type(), "ray_splat_forward", [&] {
    ray_splat_forward_kernel<scalar_t><<<blocks_for(M), 256, 0, stream>>>(
        ray_transforms.data_ptr<scalar_t>(),
        means2d.data_ptr<scalar_t>(),
        pixel_ids.data_ptr<int64_t>(),
        weight.data_ptr<scalar_t>(),
        depth.data_ptr<scalar_t>(),
        rho3.data_ptr<scalar_t>(),
        rho2.data_ptr<scalar_t>(),
        local_uv.data_ptr<scalar_t>(),
        plane.data_ptr<bool>(),
        valid.data_ptr<bool>(),
        M,
        image_width,
        static_cast<scalar_t>(hz_epsilon));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {weight, depth, rho3, rho2, local_uv, plane, valid};
}

std::vector<torch::Tensor> ray_splat_backward_cuda(
    torch::Tensor ray_transforms,
    torch::Tensor means2d,
    torch::Tensor pixel_ids,
    torch::Tensor local_uv,
    torch::Tensor weight,
    torch::Tensor valid,
    torch::Tensor plane,
    torch::Tensor grad_weight,
    torch::Tensor grad_depth,
    torch::Tensor grad_rho3,
    torch::Tensor grad_rho2,
    torch::Tensor grad_uv,
    int64_t image_width) {
  const auto M = ray_transforms.size(0);
  auto grad_ray = torch::zeros_like(ray_transforms);
  auto grad_means = torch::zeros_like(means2d);
  if (M == 0) {
    return {grad_ray, grad_means};
  }
  const c10::cuda::CUDAGuard device_guard(ray_transforms.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(ray_transforms.scalar_type(), "ray_splat_backward", [&] {
    ray_splat_backward_kernel<scalar_t><<<blocks_for(M), 256, 0, stream>>>(
        ray_transforms.data_ptr<scalar_t>(),
        means2d.data_ptr<scalar_t>(),
        pixel_ids.data_ptr<int64_t>(),
        local_uv.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        valid.data_ptr<bool>(),
        plane.data_ptr<bool>(),
        grad_weight.data_ptr<scalar_t>(),
        grad_depth.data_ptr<scalar_t>(),
        grad_rho3.data_ptr<scalar_t>(),
        grad_rho2.data_ptr<scalar_t>(),
        grad_uv.data_ptr<scalar_t>(),
        grad_ray.data_ptr<scalar_t>(),
        grad_means.data_ptr<scalar_t>(),
        M,
        image_width);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_ray, grad_means};
}
