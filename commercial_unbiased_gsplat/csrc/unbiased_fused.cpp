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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ray_splat_forward", &ray_splat_forward_cuda, "fused ray/surfel forward");
  m.def("ray_splat_backward", &ray_splat_backward_cuda, "fused ray/surfel backward");
}
