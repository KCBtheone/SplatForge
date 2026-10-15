// Torch bindings for independently written fused kernels. Apache-2.0.
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> ray_splat_forward_cuda(
    torch::Tensor ray_transforms,
    torch::Tensor means2d,
    torch::Tensor pixel_ids,
    int64_t image_width,
    double hz_epsilon);

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
    int64_t image_width);

std::vector<torch::Tensor> corrected_depth_forward_cuda(
    torch::Tensor depths,
    torch::Tensor opacities,
    torch::Tensor weights,
    torch::Tensor segment_ids,
    torch::Tensor valid,
    int64_t num_segments,
    double opacity_epsilon,
    double threshold);

std::vector<torch::Tensor> convergence_pairs_cuda(
    torch::Tensor depths,
    torch::Tensor weights,
    torch::Tensor segment_ids,
    torch::Tensor valid,
    double scene_radius);

std::vector<torch::Tensor> zncc_forward_cuda(
    torch::Tensor reference_patches,
    torch::Tensor neighbor_patches,
    double variance_epsilon);

std::vector<torch::Tensor> depth_to_points_cuda(
    torch::Tensor depth,
    torch::Tensor valid,
    torch::Tensor camera_to_world,
    torch::Tensor intrinsics);

std::vector<torch::Tensor> zncc_backward_cuda(
    torch::Tensor reference_patches,
    torch::Tensor neighbor_patches,
    torch::Tensor grad_corr,
    torch::Tensor valid,
    double variance_epsilon);

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
    double far_k);

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
    double scene_radius);

std::vector<torch::Tensor> depth_to_points_backward_cuda(
    torch::Tensor depth,
    torch::Tensor point_valid,
    torch::Tensor camera_to_world,
    torch::Tensor intrinsics,
    torch::Tensor grad_points);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ray_splat_forward", &ray_splat_forward_cuda, "fused ray/surfel forward");
  m.def("ray_splat_backward", &ray_splat_backward_cuda, "fused ray/surfel backward");
  m.def("corrected_depth_forward", &corrected_depth_forward_cuda, "fused corrected depth");
  m.def("convergence_pairs", &convergence_pairs_cuda, "fused convergence pairs");
  m.def("zncc_forward", &zncc_forward_cuda, "fused ZNCC");
  m.def("depth_to_points", &depth_to_points_cuda, "fused depth to points");
  m.def("zncc_backward", &zncc_backward_cuda, "fused ZNCC backward");
  m.def("depth_to_points_backward", &depth_to_points_backward_cuda, "fused depth to points backward");
  m.def("rasterize_unbiased_aux", &rasterize_unbiased_aux_cuda, "fused tile raster unbiased auxiliaries");
  m.def(
      "rasterize_unbiased_aux_backward",
      &rasterize_unbiased_aux_backward_cuda,
      "fused tile raster unbiased auxiliaries backward");
}
