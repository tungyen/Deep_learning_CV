import numpy as np

def _gather_feat(feat, ind, mask=None):
    dim = feat.shape[2]
    ind = ind.unsqueeze(2).expand(ind.shape[0], ind.shape[1], dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand(mask.shape[0], mask.shape[1], dim)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.shape[0], -1, feat.shape[3])
    feat = _gather_feat(feat, ind)
    return feat