// Host orchestration and python bindings for the SplatForge VCR-GauS kernels.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

#include <vector>

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
    torch::Tensor& conic_c);

void duplicate_with_keys_launch(
    const torch::Tensor& radii,
    const torch::Tensor& offsets,
    const torch::Tensor& screen_x,
    const torch::Tensor& screen_y,
    const torch::Tensor& view_z,
    int64_t tiles_x,
    int64_t tiles_y,
    torch::Tensor& keys,
    torch::Tensor& values);

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
    torch::Tensor& contributors);

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
    torch::Tensor& grad_view_xyz);

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
    torch::Tensor& counts);

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
    torch::Tensor& grad_rotations);

// Preprocess the Gaussians and bin them into per tile key/value lists.
std::vector<torch::Tensor> rasterize_geometry(
    torch::Tensor means,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor view,
    torch::Tensor proj,
    double focal_x,
    double focal_y,
    double tan_fov_x,
    double tan_fov_y,
    int64_t width,
    int64_t height) {
  const c10::cuda::CUDAGuard device_guard(means.device());
  const auto options = means.options();
  const int64_t count = means.size(0);
  const int64_t tiles_x = (width + 15) / 16;
  const int64_t tiles_y = (height + 15) / 16;
  const int64_t tile_count = tiles_x * tiles_y;

  auto radii = torch::zeros({count}, options.dtype(torch::kInt32));
  auto tiles_touched = torch::zeros({count}, options.dtype(torch::kInt32));
  auto screen_x = torch::zeros({count}, options);
  auto screen_y = torch::zeros({count}, options);
  auto view_x = torch::zeros({count}, options);
  auto view_y = torch::zeros({count}, options);
  auto view_z = torch::zeros({count}, options);
  auto conic_a = torch::zeros({count}, options);
  auto conic_b = torch::zeros({count}, options);
  auto conic_c = torch::zeros({count}, options);

  preprocess_launch(means, scales, rotations, opacities, view, proj, focal_x,
                    focal_y, tan_fov_x, tan_fov_y, width, height, tiles_x,
                    tiles_y, radii, tiles_touched, screen_x, screen_y, view_x,
                    view_y, view_z, conic_a, conic_b, conic_c);

  // The inclusive scan of the touched tiles gives the write offset of every
  // Gaussian in the unsorted key array.
  auto offsets = torch::cumsum(tiles_touched, 0).to(torch::kInt32).contiguous();
  int64_t total = 0;
  if (count > 0) {
    total = offsets[count - 1].item<int64_t>();
  }
  auto keys = torch::empty({total}, options.dtype(torch::kInt64));
  auto values = torch::empty({total}, options.dtype(torch::kInt32));
  torch::Tensor point_list;
  torch::Tensor ranges;
  if (total > 0) {
    duplicate_with_keys_launch(radii, offsets, screen_x, screen_y, view_z,
                               tiles_x, tiles_y, keys, values);
    // The stable argsort overload takes (self, stable, dim, descending).
    const auto order =
        torch::argsort(keys, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    point_list = values.index_select(0, order).to(torch::kInt32).contiguous();
    const auto sorted_keys = keys.index_select(0, order);
    const auto tile_of = torch::bitwise_right_shift(sorted_keys, 32);
    const auto tile_values =
        torch::arange(tile_count, options.dtype(torch::kInt64));
    const auto starts = torch::searchsorted(tile_of, tile_values);
    const auto ends = torch::searchsorted(tile_of, tile_values + 1);
    ranges = torch::stack({starts, ends}, 1).to(torch::kInt32).contiguous();
  } else {
    point_list = torch::zeros({0}, options.dtype(torch::kInt32));
    ranges = torch::zeros({tile_count, 2}, options.dtype(torch::kInt32));
  }

  return {radii,          screen_x,    screen_y,     view_x,      view_y,
          view_z,         conic_a,     conic_b,      conic_c,     ranges,
          point_list,     tiles_touched};
}

// Rasterize the binned scene into the surface channels.
std::vector<torch::Tensor> rasterize_render(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor screen_x,
    torch::Tensor screen_y,
    torch::Tensor view_x,
    torch::Tensor view_y,
    torch::Tensor view_z,
    torch::Tensor conic_a,
    torch::Tensor conic_b,
    torch::Tensor conic_c,
    torch::Tensor opacities,
    torch::Tensor colours,
    torch::Tensor normals,
    torch::Tensor dirs,
    torch::Tensor background,
    int64_t width,
    int64_t height) {
  const c10::cuda::CUDAGuard device_guard(opacities.device());
  const auto options = opacities.options();
  const int64_t tiles_x = (width + 15) / 16;
  const int64_t tiles_y = (height + 15) / 16;

  auto out_colour = torch::zeros({3, height, width}, options);
  auto out_depth = torch::zeros({1, height, width}, options);
  auto out_normal = torch::zeros({3, height, width}, options);
  auto out_alpha = torch::zeros({1, height, width}, options);
  auto out_distortion = torch::zeros({1, height, width}, options);
  auto final_transmittance = torch::zeros({height, width}, options);
  auto moment_one = torch::zeros({height, width}, options);
  auto moment_two = torch::zeros({height, width}, options);
  auto contributors = torch::zeros({height, width}, options.dtype(torch::kInt32));

  render_launch(ranges, point_list, screen_x, screen_y, view_x, view_y, view_z,
                conic_a, conic_b, conic_c, opacities, colours, normals, dirs,
                background, width, height, tiles_x, tiles_y, out_colour,
                out_depth, out_normal, out_alpha, out_distortion,
                final_transmittance, moment_one, moment_two, contributors);

  return {out_colour,     out_depth,           out_normal, out_alpha,
          out_distortion, final_transmittance, moment_one, moment_two,
          contributors};
}

std::vector<torch::Tensor> rasterize_forward(
    torch::Tensor means,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colours,
    torch::Tensor normals,
    torch::Tensor dirs,
    torch::Tensor view,
    torch::Tensor proj,
    torch::Tensor background,
    double focal_x,
    double focal_y,
    double tan_fov_x,
    double tan_fov_y,
    int64_t width,
    int64_t height) {
  const c10::cuda::CUDAGuard device_guard(means.device());
  auto geometry = rasterize_geometry(means, scales, rotations, opacities, view,
                                     proj, focal_x, focal_y, tan_fov_x,
                                     tan_fov_y, width, height);
  std::vector<torch::Tensor> render_inputs = {
      geometry[9],  geometry[10], geometry[1],  geometry[2],  geometry[3],
      geometry[4],  geometry[5],  geometry[6],  geometry[7],  geometry[8],
      opacities,    colours,      normals,      dirs,         background};
  auto rendered = rasterize_render(render_inputs[0], render_inputs[1],
                                    render_inputs[2], render_inputs[3],
                                    render_inputs[4], render_inputs[5],
                                    render_inputs[6], render_inputs[7],
                                    render_inputs[8], render_inputs[9],
                                    render_inputs[10], render_inputs[11],
                                    render_inputs[12], render_inputs[13],
                                    render_inputs[14], width, height);
  std::vector<torch::Tensor> output = rendered;
  for (const auto& tensor : geometry) {
    output.push_back(tensor);
  }
  return output;
}

torch::Tensor rasterize_counts(
    torch::Tensor means,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor view,
    torch::Tensor proj,
    double focal_x,
    double focal_y,
    double tan_fov_x,
    double tan_fov_y,
    int64_t width,
    int64_t height) {
  const c10::cuda::CUDAGuard device_guard(means.device());
  auto geometry = rasterize_geometry(means, scales, rotations, opacities, view,
                                     proj, focal_x, focal_y, tan_fov_x,
                                     tan_fov_y, width, height);
  const auto options = means.options();
  const int64_t count = means.size(0);
  auto counts = torch::zeros({count}, options.dtype(torch::kInt32));
  count_launch(geometry[9], geometry[10], geometry[1], geometry[2], geometry[6],
               geometry[7], geometry[8], opacities, width, height, counts);
  return counts;
}

std::vector<torch::Tensor> rasterize_backward(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor radii,
    torch::Tensor screen_x,
    torch::Tensor screen_y,
    torch::Tensor view_x,
    torch::Tensor view_y,
    torch::Tensor view_z,
    torch::Tensor conic_a,
    torch::Tensor conic_b,
    torch::Tensor conic_c,
    torch::Tensor means,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colours,
    torch::Tensor normals,
    torch::Tensor dirs,
    torch::Tensor background,
    torch::Tensor final_transmittance,
    torch::Tensor moment_one,
    torch::Tensor moment_two,
    torch::Tensor contributors,
    torch::Tensor rendered_alpha,
    torch::Tensor rendered_depth,
    torch::Tensor grad_colour,
    torch::Tensor grad_depth,
    torch::Tensor grad_normal,
    torch::Tensor grad_alpha,
    torch::Tensor grad_distortion,
    torch::Tensor view,
    torch::Tensor proj,
    double focal_x,
    double focal_y,
    double tan_fov_x,
    double tan_fov_y,
    int64_t width,
    int64_t height) {
  const c10::cuda::CUDAGuard device_guard(means.device());
  const auto options = means.options();
  const int64_t count = means.size(0);
  const int64_t tiles_x = (width + 15) / 16;
  const int64_t tiles_y = (height + 15) / 16;

  auto grad_mean2d = torch::zeros({count, 3}, options);
  auto grad_mean2d_densify = torch::zeros({count, 3}, options);
  auto grad_conic = torch::zeros({count, 3}, options);
  auto grad_opacity = torch::zeros({count, 1}, options);
  auto grad_colours = torch::zeros({count, 3}, options);
  auto grad_normals = torch::zeros({count, 3}, options);
  auto grad_view_xyz = torch::zeros({count, 3}, options);

  backward_render_launch(
      ranges, point_list, screen_x, screen_y, view_x, view_y, view_z, conic_a,
      conic_b, conic_c, opacities, colours, normals, dirs, background,
      rendered_alpha, rendered_depth, final_transmittance, moment_one,
      moment_two, contributors, grad_colour, grad_depth, grad_normal,
      grad_alpha, grad_distortion, width, height, tiles_x, tiles_y,
      grad_mean2d, grad_mean2d_densify, grad_conic, grad_opacity,
      grad_colours, grad_normals, grad_view_xyz);

  auto grad_means = torch::zeros({count, 3}, options);
  auto grad_scales = torch::zeros({count, 3}, options);
  auto grad_rotations = torch::zeros({count, 4}, options);
  backward_preprocess_launch(means, radii, scales, rotations, view, proj,
                             focal_x, focal_y, tan_fov_x, tan_fov_y,
                             grad_mean2d, grad_conic, grad_view_xyz,
                             grad_means, grad_scales, grad_rotations);
  return {grad_means, grad_scales, grad_rotations, grad_opacity, grad_colours,
          grad_normals, grad_mean2d, grad_mean2d_densify};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("rasterize_geometry", &rasterize_geometry,
             "VCR-GauS geometry and binning");
  module.def("rasterize_render", &rasterize_render,
             "VCR-GauS forward rasterization");
  module.def("rasterize_forward", &rasterize_forward,
             "VCR-GauS forward rasterization");
  module.def("rasterize_counts", &rasterize_counts,
             "VCR-GauS per Gaussian pixel activation counts");
  module.def("rasterize_backward", &rasterize_backward,
             "VCR-GauS backward rasterization");
}

}  // namespace vcr
