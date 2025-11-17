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
        loss_dict = {}
        CE = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        ce_loss = CE(logits, labels)
        total_loss = ce_loss
        loss_dict['ce_loss'] = ce_loss

        if self.lovasz_weight is not None:
            lovasz_softmax = LovaszSoftmaxLoss(ignore_index=self.ignore_index)
            lovasz_softmax_loss = self.lovasz_weight * lovasz_softmax(logits, labels)
            loss_dict['lovasz_softmax_loss'] = lovasz_softmax_loss
            total_loss += lovasz_softmax_loss
            
        if self.boundary_weight is not None:
            boundary = BoundaryLoss(ignore_index=self.ignore_index)
            boundary_loss = self.boundary_weight * boundary(logits, labels)
            loss_dict['boundary_loss'] = boundary_loss
            total_loss += boundary_loss
        loss_dict['loss'] = total_loss
        return loss_dict

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
    def __init__(self, kernel_size=3, ignore_index=None, eps=1e-7):
        super().__init__()
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index
        self.eps = eps
    
    def get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        boundary_map = F.max_pool2d(1 - mask, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size-1) // 2)
        boundary_map = boundary_map - (1 - mask)
        return boundary_map
    
    def forward(self, logits, labels):
        cls_num = logits.shape[1]
        
        valid_mask = (labels != self.ignore_index)
        labels_clamped = labels.clone()
        labels_clamped[~valid_mask] = 0
        labels_onehot = F.one_hot(labels_clamped, cls_num).permute(0, 3, 1, 2).float()
        labels_onehot = labels_onehot * valid_mask.unsqueeze(1).float()
            
        logits_boundary_map = self.get_boundary(logits)
        labels_boundary_map = self.get_boundary(labels_onehot)
        true_positive = torch.sum(labels_boundary_map * logits_boundary_map, dim=[2, 3])
        precision = true_positive / (torch.sum(logits_boundary_map, dim=[2, 3]) + self.eps)
        recall = true_positive / (torch.sum(labels_boundary_map, dim=[2, 3]) + self.eps)
        
        boundary_f1_score = 2 * precision * recall / (precision + recall + self.eps)
        if self.ignore_index is not None and 0 <= self.ignore_index < cls_num:
            boundary_f1_score[:, self.ignore_index] = 0
            boundary_f1_score = torch.sum(boundary_f1_score, dim=1) / (cls_num - 1)
        else:
            boundary_f1_score = torch.mean(boundary_f1_score, dim=1)
            
        return torch.mean(1 - boundary_f1_score)