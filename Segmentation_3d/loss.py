import torch
import torch.nn as nn
import torch.nn.functional as F

def transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, mat_diff_loss_scale=0.001, lovasz_weight=None, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.lovasz_weight = lovasz_weight
        self.ignore_index = ignore_index
        
    def forward(self, logits, labels, trans_feats=None):
        loss_dict = {}
        if self.ignore_index is not None:
            ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        else:
            ce = nn.CrossEntropyLoss()
        ce_loss = ce(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt) ** self.gamma * ce_loss
        total_loss = focal_loss
        loss_dict['focal_loss'] = focal_loss
        if self.lovasz_weight is not None:
            lovasz_softmax = LovaszSoftmaxLoss(ignore_index=self.ignore_index)
            lovasz_softmax_loss = self.lovasz_weight * lovasz_softmax(logits, labels)
            loss_dict['lovasz_softmax_loss'] = lovasz_softmax_loss
            total_loss += lovasz_softmax_loss
        if trans_feats is not None:
            mat_diff_loss = transform_reguliarzer(trans_feats)
            loss_dict['transform_loss'] = mat_diff_loss * self.mat_diff_loss_scale
            total_loss += loss_dict['transform_loss']
        loss_dict['loss'] = total_loss
        return loss_dict
    
class CrossEntropyLoss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, lovasz_weight=None, ignore_index=None):
        super().__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.lovasz_weight = lovasz_weight
        self.ignore_index = ignore_index

    def forward(self, logits, labels, trans_feats=None):
        loss_dict = {}
        if self.ignore_index is not None:
            ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        else:
            ce = nn.CrossEntropyLoss()
        ce_loss = ce(logits, labels)
        total_loss = ce_loss
        loss_dict['ce_loss'] = ce_loss
        if self.lovasz_weight is not None:
            lovasz_softmax = LovaszSoftmaxLoss(ignore_index=self.ignore_index)
            lovasz_softmax_loss = self.lovasz_weight * lovasz_softmax(logits, labels)
            loss_dict['lovasz_softmax_loss'] = lovasz_softmax_loss
            total_loss += lovasz_softmax_loss
        if trans_feats is not None:
            mat_diff_loss = transform_reguliarzer(trans_feats)
            loss_dict['transform_loss'] = mat_diff_loss * self.mat_diff_loss_scale
            total_loss += loss_dict['transform_loss']
        loss_dict['loss'] = total_loss
        return loss_dict
    
class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index=None, classes='present'):
        self.ignore_index = ignore_index
        self.classes = classes
        
    def forward(self, logits, labels):
        batch_size, cls_num, _ = logits.shape
        probs = F.softmax(logits, dim=1)
        losses = []
        
        for b in range(batch_size):
            prob = probs[b]
            label = labels[b]
            if self.ignore_index is not None:
                valid = (label != self.ignore_index)
                if valid.sum() == 0:
                    continue
                prob = prob[:, valid]
                label = label[valid]
            
            for c in range(cls_num):
                fg = (label == c).float()
                if (self.classes is 'present' and fg.sum() == 0):
                    continue
                predict_class = prob[c]
                errors = (fg - predict_class).abs()
                errors_sorted, perm = torch.sort(errors, 0, descending=True)
                fg_sorted = fg[perm]
                grad = self.lovasz_grad(fg_sorted)
                loss = torch.dot(errors_sorted, grad)
                losses.append(loss)
        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=probs.device)
    
    def lovasz_grad(self, gt_sorted):
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1-gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        if len(gt_sorted) > 1:
            tmp = jaccard.clone()
            jaccard[1:] -= tmp[:-1]
        return jaccard

def get_loss(args):
    loss_func = args.loss_func
    if loss_func == "ce":
        return CrossEntropyLoss()
    elif loss_func == "focal":
        return FocalLoss()
    elif loss_func == "ce_lovasz":
        return CrossEntropyLoss(lovasz_weight=args.lovasz_weight)
    elif loss_func == "focal_lovasz":
        return FocalLoss(lovasz_weight=args.lovasz_weight)
    else:
        raise ValueError(f'Unknown loss function {loss_func}.')