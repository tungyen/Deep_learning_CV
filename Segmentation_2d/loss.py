import torch.nn as nn
import torch
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, lovasz_alpha=None, ignore_index=None):
        super().__init__()
        self.lovasz_alpha = lovasz_alpha
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        ce_loss = loss(logits, labels)
        if self.lovasz_alpha is not None:
            ce_loss = (1.0 - self.lovasz_alpha) * ce_loss + self.lovasz_alpha * LovaszSoftmaxLoss(logits, labels, ignore_index=self.ignore_index)
        return ce_loss

def LovaszSoftmaxLoss(logits, labels, classes='present', per_image=False, ignore_index=None):
    probs = F.softmax(logits, dim=1)
    if per_image:
        losses = [lovasz_softmax_flat(*flatten_probs(
            prob.unsqueeze(), label.unsqueeze(0), ignore_index), classes=classes) for prob, label in zip(probs, labels)]
        return torch.mean(torch.stack(losses))
    else:
        return lovasz_softmax_flat(*flatten_probs(probs, labels, ignore_index), classes=classes)

def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1-gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if len(gt_sorted) > 1:
        tmp = jaccard.clone()
        jaccard[1:] -= tmp[:-1]
    return jaccard
    
def lovasz_softmax_flat(probs, labels, classes='present'):
    cls_num = probs.shape[1]
    losses = []
    
    for c in range(cls_num):
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        predict_classes = probs[:, c]
        errors = (fg - predict_classes).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        loss = torch.dot(errors_sorted, grad)
        losses.append(loss)
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=probs.device)

def flatten_probs(probs, labels, ignore_index=None):
    cls_num = probs.shape[1]
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, cls_num)
    labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels
    
    valid = (labels != ignore_index)
    return probs[valid], labels[valid]
    
def get_loss(args):
    ignore_idx = args.ignore_idx
    if args.loss_func == 'ce':
        return CrossEntropyLoss(ignore_index=ignore_idx)
    elif args.loss_func == 'ce_lovasz':
        return CrossEntropyLoss(ignore_index=ignore_idx, lovasz_alpha=args.lovasz_alpha)
    else:
        raise ValueError(f'Unknown loss function {args.loss_func}.')