#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void squared_distance_kernel(
    const float* __restrict__ batched_points_xyz_1,
    const float* __restrict__ batched_points_xyz_2,
    int n_batch, int n_points_1, int n_points_2,
    float* __restrict__ batched_dists) {
  constexpr int kMaxBlockSize2D = 32;
  __shared__ float as[kMaxBlockSize2D][3], bs[kMaxBlockSize2D][3];

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = by * blockDim.y + ty;
  const int col = bx * blockDim.x + tx;

  for (int b = 0; b < n_batch; b++) {
    const float* __restrict__ points_xyz_1 =
        batched_points_xyz_1 + b * n_points_1 * 3;
    const float* __restrict__ points_xyz_2 =
        batched_points_xyz_2 + b * n_points_2 * 3;
    float* __restrict__ dists = batched_dists + b * n_points_1 * n_points_2;

    if (tx < 3 && row < n_points_1) {
      as[ty][tx] = points_xyz_1[row * 3 + tx];
    }
    if (ty < 3 && col < n_points_2) {
      bs[tx][ty] = points_xyz_2[col * 3 + ty];
    }
    __syncthreads();

    float dist = 0;
    for (int k = 0; k < 3; k++) {
      float diff = as[ty][k] - bs[tx][k];
      dist += diff * diff;
    }
    if (row < n_points_1 && col < n_points_2) {
        dists[row * n_points_2 + col] = dist;
    }
    __syncthreads();
  }
}

void squared_distance_kernel_wrapper(
    const float* batched_points_xyz_1,
    const float* batched_points_xyz_2,
    int n_batch, int n_points_1, int n_points_2,
    float* batched_dists) {
  constexpr int kBlockSize = 32;
  dim3 number_of_blocks(
      (n_points_2 + kBlockSize - 1) / kBlockSize,
      (n_points_1 + kBlockSize - 1) / kBlockSize);
  dim3 threads_per_block(kBlockSize, kBlockSize);
  squared_distance_kernel<<<number_of_blocks, threads_per_block>>>(
      batched_points_xyz_1,
      batched_points_xyz_2,
      n_batch, n_points_1, n_points_2,
      batched_dists);
}