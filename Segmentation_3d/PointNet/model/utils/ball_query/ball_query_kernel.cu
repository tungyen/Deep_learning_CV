#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void ball_query_kernel(
    const float* __restrict__ batched_points_xyz,
    const float* __restrict__ batched_centroids_xyz,
    int n_points, int n_centroids, float radius, int n_points_per_group,
    int64_t* __restrict__ batched_indexes) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  const float* __restrict__ points_xyz = batched_points_xyz + bid * n_points * 3;
  const float* __restrict__ centroids_xyz = batched_centroids_xyz + bid * n_centroids * 3;
  int64_t* __restrict__ all_indexes = batched_indexes + bid * n_centroids * n_points_per_group;

  float radius_square = radius * radius;
  for (int i = tid; i < n_centroids; i += block_size) {
    int64_t* __restrict__ indexes = all_indexes + i * n_points_per_group;

    float cx = centroids_xyz[i * 3 + 0];
    float cy = centroids_xyz[i * 3 + 1];
    float cz = centroids_xyz[i * 3 + 2];

    int count = 0;
    for (int j = 0; j < n_points && count < n_points_per_group; j++) {
        float px = points_xyz[j * 3 + 0];
        float py = points_xyz[j * 3 + 1];
        float pz = points_xyz[j * 3 + 2];
        float d = (cx - px) * (cx - px) + (cy - py) * (cy - py) + (cz - pz) * (cz - pz);
        if (d < radius_square) {
            indexes[count++] = j;
        }
    }
    for (int j = count; j < n_points_per_group; j++) {
        indexes[j] = indexes[0];
    }
  }
}

void ball_query_kernel_wrapper(
    const float* batched_points_xyz, 
    const float* batched_centroids_xyz,
    int n_batches, int n_points, int n_centroids,
    float radius, int n_points_per_group,
    int64_t* batched_indexes) {
  ball_query_kernel<<<n_batches, 1024>>>(
      batched_points_xyz,
      batched_centroids_xyz, n_points,
      n_centroids, radius,
      n_points_per_group, batched_indexes);
}
