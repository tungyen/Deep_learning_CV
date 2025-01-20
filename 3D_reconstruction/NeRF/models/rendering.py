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


def renderRays(models, embeddings, rays, n_sample=64, useDisp=False, perturb=0, noiseStd=1, n_importance=0, chunk=1024*32, white_back=False, test_time=False):
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
        n_sample = xyz_.shape[1]
        nRays = xyz_.shape[0]
        # Embedding directions
        xyz_ = xyz_.view(-1, 3) # (nRays * n_sample, 3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=n_sample, dim=0) # (nRays * n_sample, embed_dir_channels)
            
        # Now perform the model inference to get rgb and raw sigmas
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded[i:i+chunk]], 1)
            else:
                xyzdir_embedded = xyzdir_embedded
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]
            
        out = torch.cat(out_chunks, 0)
        if weights_only:
            sigmas = out.view(nRays, n_sample)
        else:
            rgbsigma = out.view(nRays, n_sample, 4)
            rgbs = rgbsigma[..., :3] # (nRays, n_sample, 3)
            sigmas = rgbsigma[..., 3] # (nRays, n_sample)
            
        # Conversion of values by Volume rendering
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (nRays, n_sample-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (nRays, 1), which indicates that the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1) # (nRays, n_sample)
        
        # Multiply each distance by the norm of its corresponding direction ray for real world distance
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)
        noise = torch.randn(sigmas.shape, device=sigmas.device) * noiseStd
        
        # Compute alpha by the formula
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (nRays, n_sample)
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1)
        weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (nRays, n_sample)
        weights_sum = weights.sum(1) # (nRays) -> equal to 1 - (1-a1)(1-a2)(1-a3)...(1-an)
        
        if weights_only:
            return weights
        
        # Compute final weight output
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (nRays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (nRays)
        
        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)
        return rgb_final, depth_final, weights
    
    # Extract model from lists
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]
    
    # Decomposing the input
    nRays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # (nRays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # (nRays, 1)
    
    # Embedding direction
    dir_embedded = embedding_dir(rays_d) # (nRays, embed_dir_channels)
    
    # Sampling depth points
    z_steps = torch.linspace(0, 1, n_sample, device=rays.device) # (n_sample)
    if not useDisp: # using linear sampling in depth space
        z_vals = near * (1-z_steps) + 1/far * z_steps
    else: # using linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)
        
    z_vals = z_vals.expand(nRays, n_sample)
    
    if perturb > 0: # purturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1]+z_vals[:, 1:]) # (nRays, n_sample-1)->interval mid point
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper-lower) * perturb_rand
        
    xyz_coarse_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (nRays, n_sample, 3)
    
    if test_time:
        weights_coarse = inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d, dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse':weights_coarse.sum(1)}
    else:
        rgb_coarse, depth_coarse, weights_coarse = inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d, dir_embedded, z_vals, weights_only=False)
        result = {'rgb_coarse':rgb_coarse, 'depth_coarse':depth_coarse, 'weights_coarse':weights_coarse.sum(1)}
        
    if n_importance > 0: # sampling points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (nRays, n_sample-1)
        z_vals_ = pdfSampling(z_vals_mid, weights_coarse[:, 1:-1], n_importance, det=(perturb==0).detach()) # Detach to make gradient not propogate to weights_coarse
        z_vals_ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)
        xyz_fine_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_.unsqueeze(2) # (nRays, n_sample+n_importance, 3)
        
        model_fine = models[1]
        rgb_fine, depth_fine, weights_fine = inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d, dir_embedded, z_vals, weights_only=False)
        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)
        
    return result