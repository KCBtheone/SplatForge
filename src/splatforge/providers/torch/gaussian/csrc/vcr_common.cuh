// Shared math for the SplatForge VCR-GauS rasterization kernels.
//
// Matrices use a column major layout (m[column * 3 + row]) so that the
// index expressions of the mathematical contract can be transcribed without
// ambiguity.  The 4x4 camera matrices are row major and act on row vectors,
// matching the convention of the reference derivation.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace vcr {

constexpr int kTile = 16;
constexpr int kBlock = 256;
constexpr int kChannels = 3;
constexpr float kNearPlane = 0.2f;
constexpr float kFarPlane = 100.0f;

struct Vec3 {
  float x, y, z;
};

__device__ __forceinline__ float index(const Vec3& v, int i) {
  return i == 0 ? v.x : (i == 1 ? v.y : v.z);
}

struct Vec4 {
  float x, y, z, w;
};

// Column major 3x3 matrix: m[column * 3 + row].
struct Mat3 {
  float m[9];
};

// Element (row, column) of a column major matrix.
__device__ __forceinline__ float at(const Mat3& a, int row, int column) {
  return a.m[column * 3 + row];
}

__device__ __forceinline__ Mat3 mat_mul(const Mat3& a, const Mat3& b) {
  Mat3 out{};
  for (int column = 0; column < 3; ++column) {
    for (int row = 0; row < 3; ++row) {
      float sum = 0.0f;
      for (int k = 0; k < 3; ++k) {
        sum += at(a, row, k) * at(b, k, column);
      }
      out.m[column * 3 + row] = sum;
    }
  }
  return out;
}

__device__ __forceinline__ Mat3 transpose(const Mat3& a) {
  Mat3 out{};
  for (int column = 0; column < 3; ++column) {
    for (int row = 0; row < 3; ++row) {
      out.m[column * 3 + row] = at(a, column, row);
    }
  }
  return out;
}

// Row vector transform with a row major 4x4 matrix.
__device__ __forceinline__ Vec3 transform_point(const float* matrix, const Vec3& p) {
  return {matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
          matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
          matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14]};
}

__device__ __forceinline__ Vec4 transform_point_h(const float* matrix, const Vec3& p) {
  return {matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
          matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
          matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
          matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
}

// Row vector transform of a direction with the transposed rotation.
__device__ __forceinline__ Vec3 transform_vector_transpose(const float* matrix,
                                                            const Vec3& v) {
  return {matrix[0] * v.x + matrix[1] * v.y + matrix[2] * v.z,
          matrix[4] * v.x + matrix[5] * v.y + matrix[6] * v.z,
          matrix[8] * v.x + matrix[9] * v.y + matrix[10] * v.z};
}

__device__ __forceinline__ float dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Pixel centre convention of the reference implementation.
__device__ __forceinline__ float to_pixel(float value, int size) {
  return ((value + 1.0f) * size - 1.0f) * 0.5f;
}

// Tangent plane intersection depth along the pixel ray.
__device__ __forceinline__ float intersect_depth(const Vec3& point,
                                                 const Vec3& normal,
                                                 const Vec3& ray) {
  const float t = dot(normal, point) / dot(normal, ray);
  return t * ray.z;
}

// Rotation matrix of an unnormalised quaternion stored as (w, x, y, z).
__device__ __forceinline__ Mat3 rotation_from_quaternion(const float* quaternion,
                                                         int index) {
  const float r = quaternion[4 * index + 0];
  const float x = quaternion[4 * index + 1];
  const float y = quaternion[4 * index + 2];
  const float z = quaternion[4 * index + 3];
  Mat3 out{};
  out.m[0] = 1.0f - 2.0f * (y * y + z * z);
  out.m[1] = 2.0f * (x * y - r * z);
  out.m[2] = 2.0f * (x * z + r * y);
  out.m[3] = 2.0f * (x * y + r * z);
  out.m[4] = 1.0f - 2.0f * (x * x + z * z);
  out.m[5] = 2.0f * (y * z - r * x);
  out.m[6] = 2.0f * (x * z - r * y);
  out.m[7] = 2.0f * (y * z + r * x);
  out.m[8] = 1.0f - 2.0f * (x * x + y * y);
  return out;
}

// The 2D covariance (a, b, c) of the projected Gaussian.
struct Cov2D {
  float a;
  float b;
  float c;
};

// EWA projection of the 3D covariance into screen space.  The result is the
// symmetric 2x2 covariance with the one pixel low pass applied.
__device__ __forceinline__ Cov2D project_covariance(
    const Mat3& sigma,
    const Vec3& view_point,
    const float focal_x,
    const float focal_y,
    const float tan_fov_x,
    const float tan_fov_y,
    const float* view) {
  const float limit_x = 1.3f * tan_fov_x;
  const float limit_y = 1.3f * tan_fov_y;
  const float clamped_x =
      fminf(limit_x, fmaxf(-limit_x, view_point.x / view_point.z)) * view_point.z;
  const float clamped_y =
      fminf(limit_y, fmaxf(-limit_y, view_point.y / view_point.z)) * view_point.z;

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

  Cov2D out;
  out.a = projected.m[0];
  out.b = projected.m[1];
  out.c = projected.m[4];
  return out;
}

// The tile rectangle covered by a Gaussian.
struct Rect {
  int min_x;
  int min_y;
  int max_x;
  int max_y;
};

__device__ __forceinline__ Rect tile_rect(float px, float py, int radius,
                                           int tiles_x, int tiles_y) {
  Rect rect;
  rect.min_x = min(tiles_x, max(0, (int)((px - radius) / kTile)));
  rect.min_y = min(tiles_y, max(0, (int)((py - radius) / kTile)));
  rect.max_x = min(tiles_x, max(0, (int)((px + radius + kTile - 1) / kTile)));
  rect.max_y = min(tiles_y, max(0, (int)((py + radius + kTile - 1) / kTile)));
  return rect;
}

}  // namespace vcr
