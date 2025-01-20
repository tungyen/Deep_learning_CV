import torch
from kornia import create_meshgrid

def getRaysDirection(H: int, W: int, f: int):
    # This function return all rays direction of pixels in an image
    # Inputs:
    #     H - The height of the image
    #     W - The width of the image
    #     f - The focal length of the camera
    # Outputs:
    #     raysDir - (H, W, 3) in camera coordinates
    
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    u, v = grid.unbind(-1)
    raysDir = torch.stack([(u-W/2)/f, -(v-H/2)/f, -torch.ones_like(u)], -1) # (H, W, 3)
    return raysDir

def getRays(raysDir, P):
    # This function return origin and normalized rays in world coordinates
    # Inputs:
    #     raysDir - (H, W, 3) in camera coordinates
    #     P - (3, 4) transformation matrix from camera coordinates to world coordinates
    # Outputs:
    #     raysOri - (H * W, 3) origin of rays in world coordinates
    #     raysNorm - (H * W, 3) normalized direction of rays in world coordinates
    raysNorm = raysDir @ P[:, :3].T #(H, W, 3)
    raysNorm = raysNorm / torch.norm(raysNorm, dim=-1, keepdim=True)
    raysOri = P[:, 3].expand(raysNorm.shape) #(H, W, 3)
    
    raysOri.view(-1, 3)
    raysNorm.view(-1, 3)
    return raysOri, raysNorm

def getRaysNDC(H, W, f, near, raysOri, raysNorm):
    # This function transforms rays from world coordinates to NDC
    # Inputs:
    #     H - The height of the image
    #     W - The width of the image
    #     f - The focal length of the camera
    #     near - (nRays) or float, the depths of the near plane
    #     raysOri - (nRays, 3) origin of rays in world coordinates
    #     raysNorm - (nRays, 3) normalized direction of rays in world coordinates
    # Outputs:
    #     raysOri - (nRays, 3) origin of rays in NDC
    #     raysNorm - (nRays, 3) normalized direction of rays in NDC
    t = -(near + raysOri[...,2]) / raysNorm[...,2]
    raysOri = raysOri + t[..., None] * raysNorm
    
    # Store some intermediate homogeneous results
    ox_oz = raysOri[...,0] / raysOri[...,2]
    oy_oz = raysOri[...,1] / raysOri[...,2]
    
    # Projection
    o0 = -1./(W/(2.*f)) * ox_oz
    o1 = -1./(H/(2.*f)) * oy_oz
    o2 = 1. + 2. * near / raysOri[...,2]

    d0 = -1./(W/(2.*f)) * (raysNorm[...,0]/raysNorm[...,2] - ox_oz)
    d1 = -1./(H/(2.*f)) * (raysNorm[...,1]/raysNorm[...,2] - oy_oz)
    d2 = 1 - o2
    
    raysOri = torch.stack([o0, o1, o2], -1) # (B, 3)
    raysNorm = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return raysOri, raysNorm
    