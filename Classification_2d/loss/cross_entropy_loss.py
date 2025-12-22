import torch.nn as nn
import torch
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        if ignore_index is None:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        loss_dict = {}
        ce_loss = self.ce(logits, labels)
        total_loss = ce_loss
        loss_dict['ce_loss'] = ce_loss
        loss_dict['loss'] = total_loss
        return loss_dict