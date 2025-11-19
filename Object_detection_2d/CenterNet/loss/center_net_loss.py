import torch
import torch.nn as nn

from Object_detection_2d.CenterNet.utils.feats_utils import _transpose_and_gather_feat

class RegressionL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask, ind, gt):
        pred = _transpose_and_gather_feat(pred, ind)

        mask = mask.unsqueeze(2).expand_as(pred).float()

        loss = nn.functional.l1_loss(pred * mask, gt * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        positive_inds = gt.eq(1).float()
        negative_inds = gt.lt(1).float()

        negative_weights = torch.pow(1 - gt, 4)
        loss = 0

        positive_loss = torch.log(pred) * torch.pow(1-pred, 2) * positive_inds
        negative_loss = torch.log(1-pred) * torch.pow(pred, 2) * negative_weights * negative_inds

        num_positive = positive_inds.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss = loss - negative_loss
        else:
            loss = loss - (positive_loss + negative_loss) / num_positive
        return loss

class CenterDetectionLoss(nn.Module):
    def __init__(self, wh_loss_weight=0.1, offset_loss_weight=1.0):
        super().__init__()
        self.heatmap_loss = HeatmapLoss()
        self.regression_loss = RegressionL1Loss()
        self.wh_loss_weight = wh_loss_weight
        self.offset_loss_weight = offset_loss_weight

    def forward(self, pred, gt):
        pred['hm'] = torch.clamp(pred['hm'], min=1e-4, max=1-1e-4)
        hm_loss = self.heatmap_loss(pred['hm'], gt['hm'])
        wh_loss = self.regression_loss(pred['wh'], gt['offsets_mask'], gt['ind'], gt['wh'])
        offset_loss = self.regression_loss(pred['offsets'], gt['offsets_mask'], gt['ind'], gt['offsets'])

        total_loss = hm_loss + self.wh_loss_weight * wh_loss + self.offset_loss_weight * offset_loss
        loss_dict = {
            'loss': total_loss,
            'hm_loss': hm_loss,
            'wh_loss': wh_loss,
            'offset_loss': offset_loss
        }
        return loss_dict