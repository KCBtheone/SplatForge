// Backward pass of the SplatForge VCR-GauS rasterization provider.
//
// The gradients produced per Gaussian are the screen space position (used by
// the densification statistic), the 2D conic, the opacity, the colour, the
// camera space normal and the view space position.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "vcr_kernels.cuh"

namespace vcr {

__global__ void backward_render_kernel(
    const int tiles_x,
    const int tiles_y,
    const int width,
    const int height,
    const int* ranges,
    const int* point_list,
    const float* screen_x,
    const float* screen_y,
    const float* view_x,
    const float* view_y,
    const float* view_z,
    const float* conic_a,
    const float* conic_b,
    const float* conic_c,
    const float* opacities,
    const float* colours,
    const float* normals,
    const float* dirs,
    const float* background,
    const float* rendered_alpha,
    const float* rendered_depth,
    const float* final_transmittance,
    const float* moment_one,
    const float* moment_two,
    const int* contributors,
    const float* grad_colour,
    const float* grad_depth,
    const float* grad_normal,
    const float* grad_alpha,
    const float* grad_distortion,
    float* grad_mean2d,
    float* grad_mean2d_densify,
    float* grad_conic,
    float* grad_opacity,
    float* grad_colours,
    float* grad_normals,
    float* grad_view_xyz) {
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int pixel_x = tile_x * kTile + threadIdx.x;
  const int pixel_y = tile_y * kTile + threadIdx.y;
  if (pixel_x >= width || pixel_y >= height) {
    return;
  }
  const int pixel_id = pixel_y * width + pixel_x;
  const int tile = tile_y * tiles_x + tile_x;
  const int range_start = ranges[2 * tile];
  const int range_end = ranges[2 * tile + 1];

  Vec3 ray{0.0f, 0.0f, 0.0f};
  if (dirs != nullptr) {
    ray = {dirs[pixel_id], dirs[height * width + pixel_id],
           dirs[2 * height * width + pixel_id]};
  }

  const float transmittance_final = final_transmittance[pixel_id];
  const float weight_final = rendered_alpha[pixel_id];
  const float depth_final = rendered_depth[pixel_id];
  const float final_d = moment_one[pixel_id];
  const float final_d2 = moment_two[pixel_id];
  const float final_a = 1.0f - transmittance_final;
  const float grad_distortion_value = grad_distortion[pixel_id];
  const float grad_depth_value = grad_depth[pixel_id];
  const float grad_alpha_value = grad_alpha[pixel_id];
  float grad_pixel_colour[kChannels];
  for (int channel = 0; channel < kChannels; ++channel) {
    grad_pixel_colour[channel] = grad_colour[channel * height * width + pixel_id];
  }
  float grad_pixel_normal[3];
  for (int channel = 0; channel < 3; ++channel) {
    grad_pixel_normal[channel] = grad_normal[channel * height * width + pixel_id];
  }

  float transmittance = transmittance_final;
  float last_dL_dT = 0.0f;
  float colour_record[kChannels] = {0.0f, 0.0f, 0.0f};
  float last_colour[kChannels] = {0.0f, 0.0f, 0.0f};
  float depth_record = 0.0f;
  float depth_record_count = 0.0f;
  float normal_record[3] = {0.0f, 0.0f, 0.0f};
  float normal_record_count[3] = {0.0f, 0.0f, 0.0f};
  float last_normal[3] = {0.0f, 0.0f, 0.0f};
  float alpha_record = 0.0f;
  float last_alpha = 0.0f;
  float last_depth = 0.0f;

  const float ddelx_dx = 0.5f * width;
  const float ddely_dy = 0.5f * height;

  int contributor = range_end - range_start;
  const int last_contributor = contributors[pixel_id];

  for (int slot = range_end - 1; slot >= range_start; --slot) {
    --contributor;
    if (contributor >= last_contributor) {
      continue;
    }
    const int id = point_list[slot];
    const float dx = screen_x[id] - (float)pixel_x;
    const float dy = screen_y[id] - (float)pixel_y;
    const float power = -0.5f * (conic_a[id] * dx * dx + conic_c[id] * dy * dy) -
                        conic_b[id] * dx * dy;
    if (power > 0.0f) {
      continue;
    }
    const float response = expf(power);
    const float alpha = fminf(0.99f, opacities[id] * response);
    if (alpha < 1.0f / 255.0f) {
      continue;
    }
    const Vec3 normal{normals[3 * id], normals[3 * id + 1], normals[3 * id + 2]};
    const Vec3 point{view_x[id], view_y[id], view_z[id]};
    const float depth = dirs != nullptr ? intersect_depth(point, normal, ray)
                                        : view_z[id];
    if (depth < kNearPlane) {
      continue;
    }

    transmittance /= (1.0f - alpha);
    const float channel_factor = alpha * transmittance;

    float grad_opacity_value = 0.0f;
    for (int channel = 0; channel < kChannels; ++channel) {
      const float colour = colours[3 * id + channel];
      colour_record[channel] = last_alpha * last_colour[channel] +
                               (1.0f - last_alpha) * colour_record[channel];
      last_colour[channel] = colour;
      const float grad_channel = grad_pixel_colour[channel];
      grad_opacity_value += (colour - colour_record[channel]) * grad_channel;
      atomicAdd(&grad_colours[3 * id + channel], channel_factor * grad_channel);
    }

    depth_record = last_alpha * last_depth + (1.0f - last_alpha) * depth_record;
    depth_record_count =
        last_alpha + (1.0f - last_alpha) * depth_record_count;
    grad_opacity_value += (((depth - depth_record) -
                       depth_final * (1.0f - depth_record_count)) /
                      weight_final) * grad_depth_value;
    float grad_depth_gaussian = (channel_factor / weight_final) * grad_depth_value;

    const float depth_ndc =
        (kFarPlane * depth - kFarPlane * kNearPlane) /
        ((kFarPlane - kNearPlane) * depth);
    const float depth_slope =
        (kFarPlane * kNearPlane) / ((kFarPlane - kNearPlane) * depth * depth);
    grad_depth_gaussian += 2.0f * (transmittance * alpha) *
                           (depth_ndc * final_a - final_d) *
                           grad_distortion_value * depth_slope;

    float grad_weight = 0.0f;
    grad_weight += (final_d2 + depth_ndc * depth_ndc * final_a -
                    2.0f * depth_ndc * final_d) * grad_distortion_value;
    grad_opacity_value += grad_weight - last_dL_dT;
    last_dL_dT = grad_weight * alpha + (1.0f - alpha) * last_dL_dT;

    // The tangent plane depth is depth = dot(normal, point) * ray.z / dot(normal, ray),
    // so the gradient with respect to the normal is
    // ray.z * (dot(normal, ray) * point - ray * dot(normal, point)) / dot(normal, ray)^2.
    const float projection = dot(normal, ray);
    const float normal_factor =
        grad_depth_gaussian * ray.z / (projection * projection);
    const float normal_dot_point = dot(normal, point);
    const Vec3 grad_normal_depth{
        normal_factor * (projection * point.x - ray.x * normal_dot_point),
        normal_factor * (projection * point.y - ray.y * normal_dot_point),
        normal_factor * (projection * point.z - ray.z * normal_dot_point)};
    const Vec3 grad_point{
        grad_depth_gaussian * ray.z * normal.x / projection,
        grad_depth_gaussian * ray.z * normal.y / projection,
        grad_depth_gaussian * ray.z * normal.z / projection};
    atomicAdd(&grad_view_xyz[3 * id], grad_point.x);
    atomicAdd(&grad_view_xyz[3 * id + 1], grad_point.y);
    atomicAdd(&grad_view_xyz[3 * id + 2], grad_point.z);

    for (int channel = 0; channel < 3; ++channel) {
      const float component = normals[3 * id + channel];
      normal_record[channel] = last_alpha * last_normal[channel] +
                               (1.0f - last_alpha) * normal_record[channel];
      normal_record_count[channel] =
          last_alpha + (1.0f - last_alpha) * normal_record_count[channel];
      last_normal[channel] = component;
      grad_opacity_value += (((component - normal_record[channel]) -
                        last_normal[channel] *
                            (1.0f - normal_record_count[channel])) /
                       weight_final) * grad_pixel_normal[channel];
      atomicAdd(&grad_normals[3 * id + channel],
                (channel_factor / weight_final) * grad_pixel_normal[channel]);
      atomicAdd(&grad_normals[3 * id + channel], index(grad_normal_depth, channel));
    }

    alpha_record = last_alpha + (1.0f - last_alpha) * alpha_record;
    grad_opacity_value += (1.0f - alpha_record) * grad_alpha_value;
    grad_opacity_value *= transmittance;
    last_alpha = alpha;
    last_depth = depth;

    float background_dot = 0.0f;
    for (int channel = 0; channel < kChannels; ++channel) {
      background_dot += background[channel] * grad_pixel_colour[channel];
    }
    grad_opacity_value +=
        (-transmittance_final / (1.0f - alpha)) * background_dot;

    const float grad_response = opacities[id] * grad_opacity_value;
    const float gdx = response * dx;
    const float gdy = response * dy;
    const float dG_ddelx = -gdx * conic_a[id] - gdy * conic_b[id];
    const float dG_ddely = -gdy * conic_c[id] - gdx * conic_b[id];
    atomicAdd(&grad_mean2d[3 * id], grad_response * dG_ddelx * ddelx_dx);
    atomicAdd(&grad_mean2d[3 * id + 1], grad_response * dG_ddely * ddely_dy);
    atomicAdd(&grad_mean2d_densify[3 * id],
              fabsf(grad_response * dG_ddelx * ddelx_dx));
    atomicAdd(&grad_mean2d_densify[3 * id + 1],
              fabsf(grad_response * dG_ddely * ddely_dy));
    atomicAdd(&grad_conic[3 * id], -0.5f * gdx * dx * grad_response);
    atomicAdd(&grad_conic[3 * id + 1], -0.5f * gdx * dy * grad_response);
    atomicAdd(&grad_conic[3 * id + 2], -0.5f * gdy * dy * grad_response);
    atomicAdd(&grad_opacity[id], response * grad_opacity_value);
  }
}

__global__ void backward_preprocess_kernel(
    const int count,
    const float* means,
    const int* radii,
    const float* scales,
    const float* rotations,
    const float* view,
    const float* proj,
    const float focal_x,
    const float focal_y,
    const float tan_fov_x,
    const float tan_fov_y,
    const float* grad_mean2d,
    const float* grad_conic,
    const float* grad_view_xyz,
    float* grad_means,
    float* grad_scales,
    float* grad_rotations) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || !(radii[index] > 0)) {
    return;
  }
  const Vec3 origin{means[3 * index], means[3 * index + 1], means[3 * index + 2]};

  // ------------------------------------------------ 2D covariance backward
  const Vec3 view_point = transform_point(view, origin);
  const float limit_x = 1.3f * tan_fov_x;
  const float limit_y = 1.3f * tan_fov_y;
  const float ratio_x = view_point.x / view_point.z;
  const float ratio_y = view_point.y / view_point.z;
  const float x_grad_mul = (ratio_x < -limit_x || ratio_x > limit_x) ? 0.0f : 1.0f;
  const float y_grad_mul = (ratio_y < -limit_y || ratio_y > limit_y) ? 0.0f : 1.0f;
  const float clamped_x = fminf(limit_x, fmaxf(-limit_x, ratio_x)) * view_point.z;
  const float clamped_y = fminf(limit_y, fmaxf(-limit_y, ratio_y)) * view_point.z;

  Mat3 scaling{};
  scaling.m[0] = scales[3 * index];
  scaling.m[4] = scales[3 * index + 1];
  scaling.m[8] = scales[3 * index + 2];
  const Mat3 rotation = rotation_from_quaternion(rotations, index);
  const Mat3 model = mat_mul(scaling, rotation);
  const Mat3 sigma = mat_mul(transpose(model), model);

  Mat3 jacobian{};
  jacobian.m[0] = focal_x / view_point.z;
  jacobian.m[2] = -(focal_x * clamped_x) / (view_point.z * view_point.z);
  jacobian.m[4] = focal_y / view_point.z;
  jacobian.m[5] = -(focal_y * clamped_y) / (view_point.z * view_point.z);
  Mat3 view_rotation{};
  view_rotation.m[0] = view[0];
  view_rotation.m[1] = view[4];
  view_rotation.m[2] = view[8];
  view_rotation.m[3] = view[1];
  view_rotation.m[4] = view[5];
  view_rotation.m[5] = view[9];
  view_rotation.m[6] = view[2];
  view_rotation.m[7] = view[6];
  view_rotation.m[8] = view[10];
  const Mat3 transform = mat_mul(view_rotation, jacobian);
  Mat3 projected = mat_mul(transpose(transform),
                           mat_mul(transpose(sigma), transform));
  projected.m[0] += 0.3f;
  projected.m[4] += 0.3f;

  const float a = projected.m[0];
  const float b = projected.m[1];
  const float c = projected.m[4];
  const float denom = a * c - b * b;
  const float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

  const float grad_conic_x = grad_conic[3 * index];
  const float grad_conic_y = grad_conic[3 * index + 1];
  const float grad_conic_z = grad_conic[3 * index + 2];

  float grad_a = 0.0f;
  float grad_b = 0.0f;
  float grad_c = 0.0f;
  float grad_cov[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (denom2inv != 0.0f) {
    grad_a = denom2inv * (-c * c * grad_conic_x + 2.0f * b * c * grad_conic_y +
                          (denom - a * c) * grad_conic_z);
    grad_c = denom2inv * (-a * a * grad_conic_z + 2.0f * a * b * grad_conic_y +
                          (denom - a * c) * grad_conic_x);
    grad_b = denom2inv * 2.0f * (b * c * grad_conic_x -
                                 (denom + 2.0f * b * b) * grad_conic_y +
                                 a * b * grad_conic_z);

    // cov2D = transpose(T) * V * T.  The gradient with respect to the packed
    // upper triangle of V follows from the chain rule; the off diagonal
    // entries appear twice in the symmetric matrix.
    grad_cov[0] = at(transform, 0, 0) * at(transform, 0, 0) * grad_a +
                   at(transform, 0, 0) * at(transform, 0, 1) * grad_b +
                   at(transform, 0, 1) * at(transform, 0, 1) * grad_c;
    grad_cov[3] = at(transform, 1, 0) * at(transform, 1, 0) * grad_a +
                   at(transform, 1, 0) * at(transform, 1, 1) * grad_b +
                   at(transform, 1, 1) * at(transform, 1, 1) * grad_c;
    grad_cov[5] = at(transform, 2, 0) * at(transform, 2, 0) * grad_a +
                   at(transform, 2, 0) * at(transform, 2, 1) * grad_b +
                   at(transform, 2, 1) * at(transform, 2, 1) * grad_c;
    grad_cov[1] = 2.0f * at(transform, 0, 0) * at(transform, 1, 0) * grad_a +
                   (at(transform, 0, 0) * at(transform, 1, 1) +
                    at(transform, 1, 0) * at(transform, 0, 1)) * grad_b +
                   2.0f * at(transform, 0, 1) * at(transform, 1, 1) * grad_c;
    grad_cov[2] = 2.0f * at(transform, 0, 0) * at(transform, 2, 0) * grad_a +
                   (at(transform, 0, 0) * at(transform, 2, 1) +
                    at(transform, 2, 0) * at(transform, 0, 1)) * grad_b +
                   2.0f * at(transform, 0, 1) * at(transform, 2, 1) * grad_c;
    grad_cov[4] = 2.0f * at(transform, 2, 0) * at(transform, 1, 0) * grad_a +
                   (at(transform, 1, 0) * at(transform, 2, 1) +
                    at(transform, 2, 0) * at(transform, 1, 1)) * grad_b +
                   2.0f * at(transform, 1, 1) * at(transform, 2, 1) * grad_c;
  }

  // Gradient of the upper 2x3 part of the transform matrix.
  const float grad_T00 =
      2.0f * (at(transform, 0, 0) * at(sigma, 0, 0) +
              at(transform, 1, 0) * at(sigma, 1, 0) +
              at(transform, 2, 0) * at(sigma, 2, 0)) * grad_a +
      (at(transform, 0, 1) * at(sigma, 0, 0) +
       at(transform, 1, 1) * at(sigma, 1, 0) +
       at(transform, 2, 1) * at(sigma, 2, 0)) * grad_b;
  const float grad_T01 =
      2.0f * (at(transform, 0, 0) * at(sigma, 0, 1) +
              at(transform, 1, 0) * at(sigma, 1, 1) +
              at(transform, 2, 0) * at(sigma, 2, 1)) * grad_a +
      (at(transform, 0, 1) * at(sigma, 0, 1) +
       at(transform, 1, 1) * at(sigma, 1, 1) +
       at(transform, 2, 1) * at(sigma, 2, 1)) * grad_b;
  const float grad_T02 =
      2.0f * (at(transform, 0, 0) * at(sigma, 0, 2) +
              at(transform, 1, 0) * at(sigma, 1, 2) +
              at(transform, 2, 0) * at(sigma, 2, 2)) * grad_a +
      (at(transform, 0, 1) * at(sigma, 0, 2) +
       at(transform, 1, 1) * at(sigma, 1, 2) +
       at(transform, 2, 1) * at(sigma, 2, 2)) * grad_b;
  const float grad_T10 =
      2.0f * (at(transform, 0, 1) * at(sigma, 0, 0) +
              at(transform, 1, 1) * at(sigma, 1, 0) +
              at(transform, 2, 1) * at(sigma, 2, 0)) * grad_c +
      (at(transform, 0, 0) * at(sigma, 0, 0) +
       at(transform, 1, 0) * at(sigma, 1, 0) +
       at(transform, 2, 0) * at(sigma, 2, 0)) * grad_b;
  const float grad_T11 =
      2.0f * (at(transform, 0, 1) * at(sigma, 0, 1) +
              at(transform, 1, 1) * at(sigma, 1, 1) +
              at(transform, 2, 1) * at(sigma, 2, 1)) * grad_c +
      (at(transform, 0, 0) * at(sigma, 0, 1) +
       at(transform, 1, 0) * at(sigma, 1, 1) +
       at(transform, 2, 0) * at(sigma, 2, 1)) * grad_b;
  const float grad_T12 =
      2.0f * (at(transform, 0, 1) * at(sigma, 0, 2) +
              at(transform, 1, 1) * at(sigma, 1, 2) +
              at(transform, 2, 1) * at(sigma, 2, 2)) * grad_c +
      (at(transform, 0, 0) * at(sigma, 0, 2) +
       at(transform, 1, 0) * at(sigma, 1, 2) +
       at(transform, 2, 0) * at(sigma, 2, 2)) * grad_b;

  // Gradient of the non zero entries of the Jacobian.
  const float grad_J00 = at(view_rotation, 0, 0) * grad_T00 +
                          at(view_rotation, 1, 0) * grad_T01 +
                          at(view_rotation, 2, 0) * grad_T02;
  const float grad_J02 = at(view_rotation, 0, 2) * grad_T00 +
                          at(view_rotation, 1, 2) * grad_T01 +
                          at(view_rotation, 2, 2) * grad_T02;
  const float grad_J11 = at(view_rotation, 0, 1) * grad_T10 +
                          at(view_rotation, 1, 1) * grad_T11 +
                          at(view_rotation, 2, 1) * grad_T12;
  const float grad_J12 = at(view_rotation, 0, 2) * grad_T10 +
                          at(view_rotation, 1, 2) * grad_T11 +
                          at(view_rotation, 2, 2) * grad_T12;

  const float inverse_z = 1.0f / view_point.z;
  const float inverse_z2 = inverse_z * inverse_z;
  const float inverse_z3 = inverse_z2 * inverse_z;
  const float grad_tx = x_grad_mul * -focal_x * inverse_z2 * grad_J02;
  const float grad_ty = y_grad_mul * -focal_y * inverse_z2 * grad_J12;
  const float grad_tz = -focal_x * inverse_z2 * grad_J00 -
                        focal_y * inverse_z2 * grad_J11 +
                        (2.0f * focal_x * clamped_x) * inverse_z3 * grad_J02 +
                        (2.0f * focal_y * clamped_y) * inverse_z3 * grad_J12;
  const Vec3 grad_from_covariance =
      transform_vector_transpose(view, Vec3{grad_tx, grad_ty, grad_tz});

  // ------------------------------------------- screen space mean backward
  const Vec4 clip = transform_point_h(proj, origin);
  const float inverse_w = 1.0f / (clip.w + 0.0000001f);
  const float mul_x = clip.x * inverse_w * inverse_w;
  const float mul_y = clip.y * inverse_w * inverse_w;
  const float grad_screen_x = grad_mean2d[3 * index];
  const float grad_screen_y = grad_mean2d[3 * index + 1];
  const Vec3 grad_from_screen{
      (proj[0] * inverse_w - proj[3] * mul_x) * grad_screen_x +
          (proj[1] * inverse_w - proj[3] * mul_y) * grad_screen_y,
      (proj[4] * inverse_w - proj[7] * mul_x) * grad_screen_x +
          (proj[5] * inverse_w - proj[7] * mul_y) * grad_screen_y,
      (proj[8] * inverse_w - proj[11] * mul_x) * grad_screen_x +
          (proj[9] * inverse_w - proj[11] * mul_y) * grad_screen_y};

  // -------------------------------------------- tangent plane depth backward
  const Vec3 grad_view{
      grad_view_xyz[3 * index], grad_view_xyz[3 * index + 1],
      grad_view_xyz[3 * index + 2]};
  const Vec3 grad_from_depth{view[0] * grad_view.x + view[1] * grad_view.y +
                                 view[2] * grad_view.z,
                             view[4] * grad_view.x + view[5] * grad_view.y +
                                 view[6] * grad_view.z,
                             view[8] * grad_view.x + view[9] * grad_view.y +
                                 view[10] * grad_view.z};

  grad_means[3 * index] = grad_from_covariance.x + grad_from_screen.x + grad_from_depth.x;
  grad_means[3 * index + 1] =
      grad_from_covariance.y + grad_from_screen.y + grad_from_depth.y;
  grad_means[3 * index + 2] =
      grad_from_covariance.z + grad_from_screen.z + grad_from_depth.z;

  // ------------------------------------------- scale and rotation backward
  Mat3 grad_sigma{};
  grad_sigma.m[0] = grad_cov[0];
  grad_sigma.m[1] = 0.5f * grad_cov[1];
  grad_sigma.m[2] = 0.5f * grad_cov[2];
  grad_sigma.m[3] = 0.5f * grad_cov[1];
  grad_sigma.m[4] = grad_cov[3];
  grad_sigma.m[5] = 0.5f * grad_cov[4];
  grad_sigma.m[6] = 0.5f * grad_cov[2];
  grad_sigma.m[7] = 0.5f * grad_cov[4];
  grad_sigma.m[8] = grad_cov[5];
  Mat3 grad_model = mat_mul(model, grad_sigma);
  for (int i = 0; i < 9; ++i) {
    grad_model.m[i] *= 2.0f;
  }
  const Mat3 grad_model_t = transpose(grad_model);
  const Mat3 rotation_t = transpose(rotation);

  grad_scales[3 * index] = at(rotation_t, 0, 0) * at(grad_model_t, 0, 0) +
                           at(rotation_t, 1, 0) * at(grad_model_t, 1, 0) +
                           at(rotation_t, 2, 0) * at(grad_model_t, 2, 0);
  grad_scales[3 * index + 1] = at(rotation_t, 0, 1) * at(grad_model_t, 0, 1) +
                               at(rotation_t, 1, 1) * at(grad_model_t, 1, 1) +
                               at(rotation_t, 2, 1) * at(grad_model_t, 2, 1);
  grad_scales[3 * index + 2] = at(rotation_t, 0, 2) * at(grad_model_t, 0, 2) +
                               at(rotation_t, 1, 2) * at(grad_model_t, 1, 2) +
                               at(rotation_t, 2, 2) * at(grad_model_t, 2, 2);

  // Scale the columns of the transposed model gradient.
  float scaled[9];
  for (int i = 0; i < 9; ++i) {
    scaled[i] = grad_model_t.m[i];
  }
  scaled[0] *= scales[3 * index];
  scaled[1] *= scales[3 * index];
  scaled[2] *= scales[3 * index];
  scaled[3] *= scales[3 * index + 1];
  scaled[4] *= scales[3 * index + 1];
  scaled[5] *= scales[3 * index + 1];
  scaled[6] *= scales[3 * index + 2];
  scaled[7] *= scales[3 * index + 2];
  scaled[8] *= scales[3 * index + 2];

  const float r = rotations[4 * index];
  const float x = rotations[4 * index + 1];
  const float y = rotations[4 * index + 2];
  const float z = rotations[4 * index + 3];
  const float s00 = scaled[0], s10 = scaled[1], s20 = scaled[2];
  const float s01 = scaled[3], s11 = scaled[4], s21 = scaled[5];
  const float s02 = scaled[6], s12 = scaled[7], s22 = scaled[8];

  const float grad_qw = 2.0f * z * (s10 - s01) + 2.0f * y * (s02 - s20) +
                       2.0f * x * (s21 - s12);
  const float grad_qx = 2.0f * y * (s01 + s10) + 2.0f * z * (s02 + s20) +
                       2.0f * r * (s21 - s12) - 4.0f * x * (s22 + s11);
  const float grad_qy = 2.0f * x * (s01 + s10) + 2.0f * r * (s02 - s20) +
                       2.0f * z * (s21 + s12) - 4.0f * y * (s22 + s00);
  const float grad_qz = 2.0f * r * (s10 - s01) + 2.0f * x * (s02 + s20) +
                       2.0f * y * (s21 + s12) - 4.0f * z * (s11 + s00);

  grad_rotations[4 * index] = grad_qw;
  grad_rotations[4 * index + 1] = grad_qx;
  grad_rotations[4 * index + 2] = grad_qy;
  grad_rotations[4 * index + 3] = grad_qz;
}

}  // namespace vcr
