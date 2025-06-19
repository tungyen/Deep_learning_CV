#pragma once

#include <torch/extension.h>

torch::Tensor ball_query_cuda(
    torch::Tensor points_xyz,
    torch::Tensor centroids_xyz,
    float radius,
    int n_points_per_group);
