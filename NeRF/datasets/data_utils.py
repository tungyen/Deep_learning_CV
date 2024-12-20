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
    
    