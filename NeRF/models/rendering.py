import torch
from torchsearchsorted import searchsorted

def pdfSampling(bins, weights, n_importance, det=False, eps=1e-5):
    # This function samples n_importance samples from bins with distribution defined by weights
    # Input:
    #     bins - (nRays, n_sample+1) where n_samples is the number of coarse samples per ray - 2
    #     weights - Weights in shape of (nRays, n_sample)
    #     n_importance - The number of samples to draw from the distribution
    #     det - Deterministic or not
    #     eps - A small number to prevent division by zero
    # Output:
    #     samples - The sampled samples
    nRays, n_sample = weights.shape
    weights = weights + eps
    pdf = weights / torch.sum(weights, -1, keepdim=True) # Shape of (nRays, n_sample)
    cdf = torch.cumsum(pdf, -1) # (nRays, n_sample)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1) # (nRays, n_sample+1)
    
    if det:
        u = torch.linspace(0, 1, n_importance, device=bins.device)
        u = u.expand(nRays, n_importance)
    else:
        u = torch.rand(nRays, n_importance, device=bins.device)
    u = u.contiguous
    
    inds = searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, n_sample)
    
    inds_sampled = torch.stack([below, above], -1).view(nRays, 2*n_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(nRays, n_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(nRays, n_importance, 2)
    
    denom  =cdf_g[...,1]-cdf_g[...,0]
    denom[denom < eps] = 1
    
    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def renderRays(models, embeddings, rays, n_sample=64, useDisp=False, perturb=0, noiseStd=1, n_importance=0
               chunk=1024*32, white_back=False, test_time=False):
    # This function render rays by computing the output of model applied on rays
    # Input:
    #     models - list of NeRF models (coarse and fine) defined in nerf.py
    #     embeddings - list of embedding models of origin and direction defined in nerf.py
    #     rays - (nRays, 3+3+2), ray origins, directions and near, far depth bounds
    #     n_sample - number of coarse samples per ray
    #     use_disp - whether to sample in disparity space (inverse depth)
    #     perturb - factor to perturb the sampling position on the ray (for coarse model only)
    #     noise_std - factor to perturb the model's prediction of sigma
    #     n_importance - number of fine samples per ray
    #     chunk - the chunk size in batched inference
    #     white_back - whether the background is white (dataset dependent)
    #     test_time - whether it is test (inference only) or not. If True, it will not do inference
    #                on coarse rgb to save time
    # Output:
    #     res - dictionary containing final rgb and depth maps for coarse and fine models
    def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):
        # This function perform model inference of NeRF
        # Inputs:
        #     model - NeRF model (coarse or fine)
        #     embedding_xyz - embedding module for xyz
        #     xyz_ - (nRays, n_sample, 3) sampled positions
        #           n_sample is the number of sampled points in each ray;
        #                      = n_sample for coarse model
        #                      = n_sample+n_importance for fine model
        #     dir_: (nRays, 3) ray directions
        #     dir_embedded: (nRays, embed_dir_channels) embedded directions
        #     z_vals: (nRays, n_sample) depths of the sampled positions
        #     weights_only: do inference on sigma only or not

        # Outputs:
        #     if weights_only:
        #         weights: (nRays, n_sample: weights of each sample
        #     else:
        #         rgb_final: (nRays, 3) the final rgb image
        #         depth_final: (nRays) depth map
        #         weights: (nRays, n_sample): weights of each sample
            