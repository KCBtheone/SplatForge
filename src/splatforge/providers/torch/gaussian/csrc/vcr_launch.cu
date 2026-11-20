// Kernel launch wrappers for the SplatForge VCR-GauS rasterization provider.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "vcr_kernels.cuh"

namespace vcr {

void preprocess_launch(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& view,
    const torch::Tensor& proj,
    double focal_x,
    double focal_y,
    double tan_fov_x,
    double tan_fov_y,
    int64_t width,
    int64_t height,
    int64_t tiles_x,
    int64_t tiles_y,
    torch::Tensor& radii,
    torch::Tensor& tiles_touched,
    torch::Tensor& screen_x,
    torch::Tensor& screen_y,
    torch::Tensor& view_x,
    torch::Tensor& view_y,
    torch::Tensor& view_z,
    torch::Tensor& conic_a,
    torch::Tensor& conic_b,
    torch::Tensor& conic_c) {
  const int64_t count = means.size(0);
  const int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  preprocess_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (int)count, means.data_ptr<float>(), scales.data_ptr<float>(),
      rotations.data_ptr<float>(), opacities.data_ptr<float>(),
      view.data_ptr<float>(), proj.data_ptr<float>(), (float)focal_x,
      (float)focal_y, (float)tan_fov_x, (float)tan_fov_y, (int)width,
      (int)height, (int)tiles_x, (int)tiles_y, radii.data_ptr<int32_t>(),
      tiles_touched.data_ptr<int32_t>(), screen_x.data_ptr<float>(),
      screen_y.data_ptr<float>(), view_x.data_ptr<float>(),
      view_y.data_ptr<float>(), view_z.data_ptr<float>(),
      conic_a.data_ptr<float>(), conic_b.data_ptr<float>(),
      conic_c.data_ptr<float>());
}

void duplicate_with_keys_launch(
    const torch::Tensor& radii,
    const torch::Tensor& offsets,
    const torch::Tensor& screen_x,
    const torch::Tensor& screen_y,
    const torch::Tensor& view_z,
    int64_t tiles_x,
    int64_t tiles_y,
    torch::Tensor& keys,
    torch::Tensor& values) {
  const int64_t count = radii.size(0);
  const int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  duplicate_with_keys_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (int)count, (int)tiles_x, (int)tiles_y, radii.data_ptr<int32_t>(),
      offsets.data_ptr<int32_t>(), screen_x.data_ptr<float>(),
      screen_y.data_ptr<float>(), view_z.data_ptr<float>(),
      reinterpret_cast<long long*>(keys.data_ptr<int64_t>()),
      values.data_ptr<int32_t>());
}

void render_launch(
    const torch::Tensor& ranges,
    const torch::Tensor& point_list,
    const torch::Tensor& screen_x,
    const torch::Tensor& screen_y,
    const torch::Tensor& view_x,
    const torch::Tensor& view_y,
    const torch::Tensor& view_z,
    const torch::Tensor& conic_a,
    const torch::Tensor& conic_b,
    const torch::Tensor& conic_c,
    const torch::Tensor& opacities,
    const torch::Tensor& colours,
    const torch::Tensor& normals,
    const torch::Tensor& dirs,
    const torch::Tensor& background,
    int64_t width,
    int64_t height,
    int64_t tiles_x,
    int64_t tiles_y,
    torch::Tensor& out_colour,
    torch::Tensor& out_depth,
    torch::Tensor& out_normal,
    torch::Tensor& out_alpha,
    torch::Tensor& out_distortion,
    torch::Tensor& final_transmittance,
    torch::Tensor& moment_one,
    torch::Tensor& moment_two,
    torch::Tensor& contributors) {
  const dim3 grid((unsigned int)tiles_x, (unsigned int)tiles_y);
  const dim3 block(16, 16);
  render_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (int)tiles_x, (int)tiles_y, (int)width, (int)height,
      ranges.data_ptr<int32_t>(), point_list.data_ptr<int32_t>(),
      screen_x.data_ptr<float>(), screen_y.data_ptr<float>(),
      view_x.data_ptr<float>(), view_y.data_ptr<float>(),
      view_z.data_ptr<float>(), conic_a.data_ptr<float>(),
      conic_b.data_ptr<float>(), conic_c.data_ptr<float>(),
      opacities.data_ptr<float>(), colours.data_ptr<float>(),
      normals.data_ptr<float>(), dirs.data_ptr<float>(),
      background.data_ptr<float>(), out_colour.data_ptr<float>(),
      out_depth.data_ptr<float>(), out_normal.data_ptr<float>(),
      out_alpha.data_ptr<float>(), out_distortion.data_ptr<float>(),
      final_transmittance.data_ptr<float>(), moment_one.data_ptr<float>(),
      moment_two.data_ptr<float>(), contributors.data_ptr<int32_t>());
}

void backward_render_launch(
    const torch::Tensor& ranges,
    const torch::Tensor& point_list,
    const torch::Tensor& screen_x,
    const torch::Tensor& screen_y,
    const torch::Tensor& view_x,
    const torch::Tensor& view_y,
    const torch::Tensor& view_z,
    const torch::Tensor& conic_a,
    const torch::Tensor& conic_b,
    const torch::Tensor& conic_c,
    const torch::Tensor& opacities,
    const torch::Tensor& colours,
    const torch::Tensor& normals,
    const torch::Tensor& dirs,
    const torch::Tensor& background,
    const torch::Tensor& rendered_alpha,
    const torch::Tensor& rendered_depth,
    const torch::Tensor& final_transmittance,
    const torch::Tensor& moment_one,
    const torch::Tensor& moment_two,
    const torch::Tensor& contributors,
    const torch::Tensor& grad_colour,
    const torch::Tensor& grad_depth,
    const torch::Tensor& grad_normal,
    const torch::Tensor& grad_alpha,
    const torch::Tensor& grad_distortion,
    int64_t width,
    int64_t height,
    int64_t tiles_x,
    int64_t tiles_y,
    torch::Tensor& grad_mean2d,
    torch::Tensor& grad_mean2d_densify,
    torch::Tensor& grad_conic,
    torch::Tensor& grad_opacity,
    torch::Tensor& grad_colours,
    torch::Tensor& grad_normals,
    torch::Tensor& grad_view_xyz) {
  const dim3 grid((unsigned int)tiles_x, (unsigned int)tiles_y);
  const dim3 block(16, 16);
  backward_render_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (int)tiles_x, (int)tiles_y, (int)width, (int)height,
      ranges.data_ptr<int32_t>(), point_list.data_ptr<int32_t>(),
      screen_x.data_ptr<float>(), screen_y.data_ptr<float>(),
      view_x.data_ptr<float>(), view_y.data_ptr<float>(),
      view_z.data_ptr<float>(), conic_a.data_ptr<float>(),
      conic_b.data_ptr<float>(), conic_c.data_ptr<float>(),
      opacities.data_ptr<float>(), colours.data_ptr<float>(),
      normals.data_ptr<float>(), dirs.data_ptr<float>(),
      background.data_ptr<float>(), rendered_alpha.data_ptr<float>(),
      rendered_depth.data_ptr<float>(), final_transmittance.data_ptr<float>(),
      moment_one.data_ptr<float>(), moment_two.data_ptr<float>(),
      contributors.data_ptr<int32_t>(), grad_colour.data_ptr<float>(),
      grad_depth.data_ptr<float>(), grad_normal.data_ptr<float>(),
      grad_alpha.data_ptr<float>(), grad_distortion.data_ptr<float>(),
      grad_mean2d.data_ptr<float>(), grad_mean2d_densify.data_ptr<float>(),
      grad_conic.data_ptr<float>(), grad_opacity.data_ptr<float>(),
      grad_colours.data_ptr<float>(), grad_normals.data_ptr<float>(),
      grad_view_xyz.data_ptr<float>());
}

void backward_preprocess_launch(
    const torch::Tensor& means,
    const torch::Tensor& radii,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& view,
    const torch::Tensor& proj,
    double focal_x,
    double focal_y,
    double tan_fov_x,
    double tan_fov_y,
    const torch::Tensor& grad_mean2d,
    const torch::Tensor& grad_conic,
    const torch::Tensor& grad_view_xyz,
    torch::Tensor& grad_means,
    torch::Tensor& grad_scales,
    torch::Tensor& grad_rotations) {
  const int64_t count = means.size(0);
  const int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  backward_preprocess_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (int)count, means.data_ptr<float>(), radii.data_ptr<int32_t>(),
      scales.data_ptr<float>(), rotations.data_ptr<float>(),
      view.data_ptr<float>(), proj.data_ptr<float>(), (float)focal_x,
      (float)focal_y, (float)tan_fov_x, (float)tan_fov_y,
      grad_mean2d.data_ptr<float>(), grad_conic.data_ptr<float>(),
      grad_view_xyz.data_ptr<float>(), grad_means.data_ptr<float>(),
      grad_scales.data_ptr<float>(), grad_rotations.data_ptr<float>());
}

void count_launch(
    const torch::Tensor& ranges,
    const torch::Tensor& point_list,
    const torch::Tensor& screen_x,
    const torch::Tensor& screen_y,
    const torch::Tensor& conic_a,
    const torch::Tensor& conic_b,
    const torch::Tensor& conic_c,
    const torch::Tensor& opacities,
    int64_t width,
    int64_t height,
    torch::Tensor& counts) {
  const int64_t tiles_x = (width + 15) / 16;
  const int64_t tiles_y = (height + 15) / 16;
  const dim3 grid((unsigned int)tiles_x, (unsigned int)tiles_y);
  const dim3 block(16, 16);
  count_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (int)tiles_x, (int)tiles_y, (int)width, (int)height,
      ranges.data_ptr<int32_t>(), point_list.data_ptr<int32_t>(),
      screen_x.data_ptr<float>(), screen_y.data_ptr<float>(),
      conic_a.data_ptr<float>(), conic_b.data_ptr<float>(),
      conic_c.data_ptr<float>(), opacities.data_ptr<float>(),
      counts.data_ptr<int32_t>());
}

}  // namespace vcr
