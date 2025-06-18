#include "ball_query.h"

void ball_query_kernel_wrapper(const float* batched_points_xyz, 
                               const float* batched_centroids_xyz,
                               int n_batches, int n_points, int n_centroids,
                               float radius, int n_points_per_group,
                               int64_t* batched_index);

torch::Tensor ball_query_cuda(torch::Tensor points_xyz, 
                              torch::Tensor centroids_xyz,
                              float radius, int n_points_per_group) {
  TORCH_CHECK(points_xyz.is_contiguous(), "points_xyz must be a contiguous tensor");
  TORCH_CHECK(points_xyz.is_cuda(), "points_xyz must be a CUDA tensor");
  TORCH_CHECK(points_xyz.scalar_type() == torch::ScalarType::Float, "points_xyz must be a float tensor");

  TORCH_CHECK(centroids_xyz.is_contiguous(), "centroids_xyz must be a contiguous tensor");
  TORCH_CHECK(centroids_xyz.is_cuda(), "centroids_xyz must be a CUDA tensor");
  TORCH_CHECK(centroids_xyz.scalar_type() == torch::ScalarType::Float, "centroids_xyz must be a float tensor");

  int64_t batch_size  = points_xyz.size(0);
  int64_t n_points = points_xyz.size(1);
  int64_t n_centroids = centroids_xyz.size(1);
  torch::Tensor index = torch::ones(
    { batch_size, n_centroids, n_points_per_group },
    torch::device(points_xyz.device()).dtype(torch::kInt64)) * n_points;
  ball_query_kernel_wrapper(points_xyz.data_ptr<float>(),
                            centroids_xyz.data_ptr<float>(), batch_size,
                            n_points, n_centroids,
                            radius, n_points_per_group,
                            index.data_ptr<int64_t>());
}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_ball_query_cuda", &ball_query_cuda, "CUDA implement of ball-query")
}