// Forward pass of the SplatForge VCR-GauS rasterization provider.
//
// The forward pass produces the five channels the surface reconstruction
// recipe needs: the alpha composited colour, the accumulated alpha, the
// tangent plane (surfel) depth, the camera space surface normal and the two
// dimensional distortion measure.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "vcr_kernels.cuh"

namespace vcr {

__global__ void preprocess_kernel(
    const int count,
    const float* means,
    const float* scales,
    const float* rotations,
    const float* opacities,
    const float* view,
    const float* proj,
    const float focal_x,
    const float focal_y,
    const float tan_fov_x,
    const float tan_fov_y,
    const int width,
    const int height,
    const int tiles_x,
    const int tiles_y,
    int* radii,
    int* tiles_touched,
    float* screen_x,
    float* screen_y,
    float* view_x,
    float* view_y,
    float* view_z,
    float* conic_a,
    float* conic_b,
    float* conic_c) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  radii[index] = 0;
  tiles_touched[index] = 0;

  const Vec3 origin{means[3 * index], means[3 * index + 1], means[3 * index + 2]};
  const Vec3 view_point = transform_point(view, origin);
  if (view_point.z <= 0.2f) {
    return;
  }

  const Vec4 clip = transform_point_h(proj, origin);
  const float inverse_w = 1.0f / (clip.w + 0.0000001f);
  const float ndc_x = clip.x * inverse_w;
  const float ndc_y = clip.y * inverse_w;

  // 3D covariance of the Gaussian.
  Mat3 scaling{};
  scaling.m[0] = scales[3 * index];
  scaling.m[4] = scales[3 * index + 1];
  scaling.m[8] = scales[3 * index + 2];
  const Mat3 rotation = rotation_from_quaternion(rotations, index);
  const Mat3 model = mat_mul(scaling, rotation);
  const Mat3 sigma = mat_mul(transpose(model), model);

  const Cov2D cov = project_covariance(sigma, view_point, focal_x, focal_y,
                                        tan_fov_x, tan_fov_y, view);
  const float det = cov.a * cov.c - cov.b * cov.b;
  if (det == 0.0f) {
    return;
  }
  const float inverse_det = 1.0f / det;

  const float mid = 0.5f * (cov.a + cov.c);
  const float lambda1 = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
  const float lambda2 = mid - sqrtf(fmaxf(0.1f, mid * mid - det));
  const int radius = (int)ceilf(3.0f * sqrtf(fmaxf(lambda1, lambda2)));

  const float px = to_pixel(ndc_x, width);
  const float py = to_pixel(ndc_y, height);
  const Rect rect = tile_rect(px, py, radius, tiles_x, tiles_y);
  if ((rect.max_x - rect.min_x) * (rect.max_y - rect.min_y) == 0) {
    return;
  }

  radii[index] = radius;
  tiles_touched[index] = (rect.max_y - rect.min_y) * (rect.max_x - rect.min_x);
  screen_x[index] = px;
  screen_y[index] = py;
  view_x[index] = view_point.x;
  view_y[index] = view_point.y;
  view_z[index] = view_point.z;
  conic_a[index] = cov.c * inverse_det;
  conic_b[index] = -cov.b * inverse_det;
  conic_c[index] = cov.a * inverse_det;
}

__global__ void duplicate_with_keys_kernel(
    const int count,
    const int tiles_x,
    const int tiles_y,
    const int* radii,
    const int* offsets,
    const float* screen_x,
    const float* screen_y,
    const float* view_z,
    long long* keys,
    int* values) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  if (radii[index] <= 0) {
    return;
  }
  int slot = (index == 0) ? 0 : offsets[index - 1];
  const int radius = radii[index];
  const Rect rect = tile_rect(screen_x[index], screen_y[index], radius, tiles_x,
                              tiles_y);
  const unsigned int depth_bits = __float_as_uint(view_z[index]);
  for (int y = rect.min_y; y < rect.max_y; ++y) {
    for (int x = rect.min_x; x < rect.max_x; ++x) {
      const unsigned long long tile = (unsigned long long)(y * tiles_x + x);
      const unsigned long long key = (tile << 32) | (unsigned long long)depth_bits;
      keys[slot] = (long long)key;
      values[slot] = index;
      ++slot;
    }
  }
}

__global__ void render_kernel(
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
    float* out_colour,
    float* out_depth,
    float* out_normal,
    float* out_alpha,
    float* out_distortion,
    float* final_transmittance,
    float* moment_one,
    float* moment_two,
    int* contributors) {
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

  float transmittance = 1.0f;
  int contributor = 0;
  int last_contributor = 0;
  float colour[kChannels] = {0.0f, 0.0f, 0.0f};
  float weight = 1e-10f;
  float depth_sum = 0.0f;
  float normal_sum[3] = {0.0f, 0.0f, 0.0f};
  float moment_1 = 0.0f;
  float moment_2 = 0.0f;
  float distortion = 0.0f;

  for (int slot = range_start; slot < range_end; ++slot) {
    ++contributor;
    const int id = point_list[slot];
    const float dx = screen_x[id] - (float)pixel_x;
    const float dy = screen_y[id] - (float)pixel_y;
    const float power = -0.5f * (conic_a[id] * dx * dx + conic_c[id] * dy * dy) -
                        conic_b[id] * dx * dy;
    if (power > 0.0f) {
      continue;
    }
    const Vec3 normal{normals[3 * id], normals[3 * id + 1], normals[3 * id + 2]};
    const Vec3 point{view_x[id], view_y[id], view_z[id]};
    const float depth = dirs != nullptr ? intersect_depth(point, normal, ray)
                                        : view_z[id];
    if (depth <= kNearPlane) {
      continue;
    }
    const float alpha = fminf(0.99f, opacities[id] * expf(power));
    if (alpha < 1.0f / 255.0f) {
      continue;
    }
    const float test_transmittance = transmittance * (1.0f - alpha);
    if (test_transmittance < 0.0001f) {
      break;
    }
    const float contribution = alpha * transmittance;

    const float depth_ndc =
        (kFarPlane * depth - kFarPlane * kNearPlane) /
        ((kFarPlane - kNearPlane) * depth);
    const float coverage = 1.0f - transmittance;
    const float error = depth_ndc * depth_ndc * coverage + moment_2 -
                        2.0f * depth_ndc * moment_1;
    distortion += error * contribution;
    moment_1 += depth_ndc * contribution;
    moment_2 += depth_ndc * depth_ndc * contribution;

    for (int channel = 0; channel < kChannels; ++channel) {
      colour[channel] += colours[3 * id + channel] * contribution;
    }
    weight += contribution;
    depth_sum += depth * contribution;
    for (int channel = 0; channel < 3; ++channel) {
      normal_sum[channel] += index(normal, channel) * contribution;
    }
    transmittance = test_transmittance;
    last_contributor = contributor;
  }

  final_transmittance[pixel_id] = transmittance;
  contributors[pixel_id] = last_contributor;
  for (int channel = 0; channel < kChannels; ++channel) {
    out_colour[channel * height * width + pixel_id] =
        colour[channel] + transmittance * background[channel];
    out_normal[channel * height * width + pixel_id] =
        normal_sum[channel] / weight;
  }
  out_depth[pixel_id] = depth_sum / weight;
  out_alpha[pixel_id] = weight;
  out_distortion[pixel_id] =
      distortion / ((1.0f - transmittance) * (1.0f - transmittance) + 1e-7f);
  moment_one[pixel_id] = moment_1;
  moment_two[pixel_id] = moment_2;
}

__global__ void count_kernel(
    const int tiles_x,
    const int tiles_y,
    const int width,
    const int height,
    const int* ranges,
    const int* point_list,
    const float* screen_x,
    const float* screen_y,
    const float* conic_a,
    const float* conic_b,
    const float* conic_c,
    const float* opacities,
    int* counts) {
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int pixel_x = tile_x * kTile + threadIdx.x;
  const int pixel_y = tile_y * kTile + threadIdx.y;
  if (pixel_x >= width || pixel_y >= height) {
    return;
  }
  const int tile = tile_y * tiles_x + tile_x;
  const int range_start = ranges[2 * tile];
  const int range_end = ranges[2 * tile + 1];
  float transmittance = 1.0f;
  for (int slot = range_start; slot < range_end; ++slot) {
    const int id = point_list[slot];
    const float dx = screen_x[id] - (float)pixel_x;
    const float dy = screen_y[id] - (float)pixel_y;
    const float power = -0.5f * (conic_a[id] * dx * dx + conic_c[id] * dy * dy) -
                        conic_b[id] * dx * dy;
    if (power > 0.0f) {
      continue;
    }
    const float alpha = fminf(0.99f, opacities[id] * expf(power));
    if (alpha < 1.0f / 255.0f) {
      continue;
    }
    const float test_transmittance = transmittance * (1.0f - alpha);
    if (test_transmittance < 0.0001f) {
      break;
    }
    atomicAdd(&counts[id], 1);
    transmittance = test_transmittance;
  }
}

}  // namespace vcr
