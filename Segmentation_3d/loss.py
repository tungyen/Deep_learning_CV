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
    def __init__(self, alpha=0.25, gamma=2, mat_diff_loss_scale=0.001, lovasz_alpha=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.lovasz_alpha = lovasz_alpha
        
    def forward(self, logits, labels, trans_feats=None):
        loss = nn.CrossEntropyLoss()
        CE_loss = loss(logits, labels)
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * (1-pt) ** self.gamma * CE_loss
        if self.lovasz_alpha is not None:
            lovasz_loss = LovaszSoftmaxLoss(logits, labels)
            # print("Lovasz Softmax Loss: ", lovasz_loss)
            # print("Focal Loss: ", focal_loss)
            focal_loss = (1.0 - self.lovasz_alpha) * focal_loss + self.lovasz_alpha * lovasz_loss
        if trans_feats is not None:
            mat_diff_loss = transform_reguliarzer(trans_feats)
            focal_loss += mat_diff_loss * self.mat_diff_loss_scale
        return focal_loss
    
class CrossEntropyLoss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, lovasz_alpha=None):
        super().__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.lovasz_alpha = lovasz_alpha

    def forward(self, logits, labels, trans_feats=None):
        loss = nn.CrossEntropyLoss()
        ce_loss = loss(logits, labels)
        if self.lovasz_alpha is not None:
            lovasz_loss = LovaszSoftmaxLoss(logits, labels)
            # print("Lovasz Softmax Loss: ", lovasz_loss)
            # print("Cross Entropy Loss: ", ce_loss)
            ce_loss = (1.0 - self.lovasz_alpha) * ce_loss + self.lovasz_alpha * LovaszSoftmaxLoss(logits, labels)
        if trans_feats is not None:
            mat_diff_loss = transform_reguliarzer(trans_feats)
            ce_loss += mat_diff_loss
        return ce_loss
    
def LovaszSoftmaxLoss(logits, labels, ignore_index=None, classes='present'):
    batch_size, cls_num, _ = logits.shape
    probs = F.softmax(logits, dim=1)
    losses = []
    
    for b in range(batch_size):
        prob = probs[b]
        label = labels[b]
        if ignore_index is not None:
            valid = (label != ignore_index)
            if valid.sum() == 0:
                continue
            prob = prob[:, valid]
            label = label[valid]
        
        for c in range(cls_num):
            fg = (label == c).float()
            if (classes is 'present' and fg.sum() == 0):
                continue
            predict_class = prob[c]
            errors = (fg - predict_class).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm]
            grad = lovasz_grad(fg_sorted)
            loss = torch.dot(errors_sorted, grad)
            losses.append(loss)
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=probs.device)
    
def lovasz_grad(gt_sorted):
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
        return CrossEntropyLoss(lovasz_alpha=args.lovasz_alpha)
    elif loss_func == "focal_lovasz":
        return FocalLoss(lovasz_alpha=args.lovasz_alpha)
    else:
        raise ValueError(f'Unknown loss function {loss_func}.')