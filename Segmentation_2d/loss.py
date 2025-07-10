import torch.nn as nn
import torch
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, lovasz_weight=None, boundary_weight=None, ignore_index=None):
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        ce_loss = loss(logits, labels)

        if self.lovasz_weight is not None:
            lovasz_loss = LovaszSoftmaxLoss(ignore_index=self.ignore_index)
            ce_loss += self.lovasz_weight * lovasz_loss(logits, labels)
            
        if self.boundary_weight is not None:
            boundary_loss = BoundaryLoss(ignore_index=self.ignore_index)
            ce_loss += self.boundary_weight * boundary_loss(logits, labels) 
        return ce_loss

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=None):
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
        
    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        if self.per_image:
            losses = [self.lovasz_softmax_flat(*self.flatten_probs(
                prob.unsqueeze(), label.unsqueeze(0))) for prob, label in zip(probs, labels)]
            return torch.mean(torch.stack(losses))
        else:
            return self.lovasz_softmax_flat(*self.flatten_probs(probs, labels))

    def lovasz_grad(self, gt_sorted):
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1-gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        if len(gt_sorted) > 1:
            tmp = jaccard.clone()
            jaccard[1:] -= tmp[:-1]
        return jaccard
    
    def lovasz_softmax_flat(self, probs, labels):
        cls_num = probs.shape[1]
        losses = []
        
        for c in range(cls_num):
            fg = (labels == c).float()
            if self.classes == 'present' and fg.sum() == 0:
                continue
            predict_classes = probs[:, c]
            errors = (fg - predict_classes).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self.lovasz_grad(fg_sorted)
            loss = torch.dot(errors_sorted, grad)
            losses.append(loss)
        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=probs.device)

    def flatten_probs(self, probs, labels):
        cls_num = probs.shape[1]
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, cls_num)
        labels = labels.view(-1)
        if self.ignore_index is None:
            return probs, labels
        
        valid = (labels != self.ignore_index)
        return probs[valid], labels[valid]

class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size=3, ignore_index=None):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.ignore_index = ignore_index
    
    def get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        pool_result = self.pool(mask)
        boundary = pool_result - mask
        return boundary.clamp(min=0, max=1)
    
    def forward(self, logits, labels):
        cls_num = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        if self.ignore_index is not None:
            valid_mask = (labels != self.ignore_index).unsqueeze(1).long()
            labels = labels.clone()
            labels *= valid_mask.squeeze()
        else:
            valid_mask = torch.ones_like(labels, dtype=torch.long).unsqueeze(1)
            
        labels_onehot = F.one_hot(labels.clamp(min=0), num_classes=cls_num).permute(0, 3, 1, 2).float()
        labels_onehot *= valid_mask
        probs *= valid_mask
            
        pred_boundary = self.get_boundary(probs)
        labels_boundary = self.get_boundary(labels_onehot)
        
        smooth = 1.0
        intersection = (pred_boundary * labels_boundary).sum(dim=(2, 3))
        union = pred_boundary.sum(dim=(2, 3)) + labels_boundary.sum(dim=(2, 3))
        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
        return dice_loss.mean()
    
def get_loss(args):
    ignore_idx = args.ignore_idx
    if args.loss_func == 'ce':
        return CrossEntropyLoss(ignore_index=ignore_idx, lovasz_weight=args.lovasz_weight, boundary_weight=args.boundary_weight)
    else:
        raise ValueError(f'Unknown loss function {args.loss_func}.')