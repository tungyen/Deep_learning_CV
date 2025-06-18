import torch

try:
    from ._ball_query_cuda import _ball_query_cuda
    from ._furthest_point_sampling_cuda import _furthest_point_sampling_cuda
    from ._squared_distance_cuda import _squared_distance_cuda
except ImportError:
    raise ImportError('Failed to load one or more extensions')

def squared_distance(points_xyz_1, points_xyz_2, cpp_impl=True):
    @torch.cuda.amp.autocast(enabled=False)
    def _squared_distance_py(points_xyz_1, points_xyz_2):
        assert points_xyz_1.shape[0] == points_xyz_2.shape[0]
        
        batch_size, n_points_1, n_points_2 = points_xyz_1.shape[0], points_xyz_1.shape[1], points_xyz_2.shape[1]
        dist = -2 * torch.matmul(points_xyz_1, points_xyz_2.permute(0, 2, 1))
        dist += torch.sum(points_xyz_1 ** 2, -1).view(batch_size, n_points_1, 1)
        dist += torch.sum(points_xyz_2 ** 2, -1).view(batch_size, n_points_2, 1)
        return dist
    
    if cpp_impl:
        return _squared_distance_cuda(points_xyz_1, points_xyz_2)
    else:
        return _squared_distance_py(points_xyz_1, points_xyz_2)


def ball_query(points_xyz, centroid_xyz, radius, n_points_per_group, cpp_impl=True):
    def _ball_query_py(points_xyz, centroid_xyz, radius, n_points_per_group):
        batch_size, n_points, n_centroids = points_xyz.shape[0], points_xyz.shape[1], centroid_xyz.shape[1]
        idx = torch.arange(n_points, dtype=torch.long, device=points_xyz.device)
        grouped_idx = idx.view(1, 1, n_points).repeat([batch_size, n_centroids, 1])
        dists = squared_distance(centroid_xyz, points_xyz, cpp_impl=False)
        grouped_idx[dists >= radius ** 2] = n_points # This is for out of range points
        grouped_idx = grouped_idx.sort(dim=-1)[0][:, :, :n_points_per_group]
        grouped_idx_first = grouped_idx[:, :, :1].expand(grouped_idx.shape)
        mask = grouped_idx == n_points
        grouped_idx[mask] = grouped_idx_first[mask]
        return grouped_idx
    
    if cpp_impl:
        grouped_idx = _ball_query_cuda(points_xyz, centroid_xyz, radius, n_points_per_group)
    else:
        grouped_idx = _ball_query_py(points_xyz, centroid_xyz, radius, n_points_per_group)
        
    return grouped_idx.to(torch.long)


def furthest_point_sampling(points_xyz, n_samples, cpp_impl=True):
    def _furthest_point_sampling_py(points_xyz, n_samples):
        batch_size, n_points, _ = points_xyz.shape
        farthest_idx = torch.zeros(batch_size, n_samples, dtype=torch.long, device=points_xyz.device)
        distances = torch.ones(batch_size, n_points, device=points_xyz.device) * 1e10
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=points_xyz.device)
        current_farthest_idx = torch.zeros(batch_size, dtype=torch.long, device=points_xyz.device)
        
        for i in range(n_samples):
            farthest_idx[:, i] = current_farthest_idx
            current_farthest = points_xyz[batch_idx, current_farthest_idx, :].view(batch_size, 1, 3)
            new_distances = torch.sum((points_xyz - current_farthest) ** 2, -1)
            mask = new_distances < distances
            distances[mask] = new_distances[mask]
            current_farthest_idx = torch.max(distances, -1)[1]
            
        if cpp_impl:
            return _furthest_point_sampling_cuda(points_xyz, n_samples).to(torch.long)
        else:
            return _furthest_point_sampling_py(points_xyz, n_samples).to(torch.long)
        
def k_nearest_neighbor(points_xyz, centroids_xyz, k, cpp_impl=True):
    dists = squared_distance(centroids_xyz, points_xyz, cpp_impl)
    top_k = dists.topk(k, dim=-1, largest=False)
    return top_k.values, top_k.index

def batch_indexing(batched_data, batched_index):
    assert batched_data.shape[0] == batched_index.shape[0]
    batch_size = batched_data.shape[0]
    view_shape = [batch_size] + [1] * (len(batched_index.shape) - 1)
    expand_shape = [batch_size] + list(batched_index.shape)[1:]
    index_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
    index_of_batch = index_of_batch.view(view_shape).expand(expand_shape)
    return batched_data[index_of_batch, batched_index, :]