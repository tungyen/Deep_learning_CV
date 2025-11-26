import torch

from Object_detection_2d.data.container import Container
from Object_detection_2d.CenterNet.utils import _gather_feat, _transpose_and_gather_feat

def NMS(hm):
    hm_max = torch.nn.functional.max_pool2d(hm, (3, 3), stride=1, padding=1)
    keep = (hm_max == hm).float()
    return hm * keep


def _topk(scores, K):
    bs, c, h, w = scores.shape

    topk_scores, topk_inds = torch.topk(scores.view(bs, c, -1), K)
    topk_inds = topk_inds % (h * w)
    topk_ys = (topk_inds / w).int().float()
    topk_xs = (topk_inds % w).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(bs, -1), K)
    topk_clses = (topk_ind / K).int()

    topk_inds = _gather_feat(topk_inds.view(bs, -1, 1), topk_ind).view(bs, K)
    topk_ys = _gather_feat(topk_ys.view(bs, -1, 1), topk_ind).view(bs, K)
    topk_xs = _gather_feat(topk_xs.view(bs, -1, 1), topk_ind).view(bs, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

class PostProcessor:
    def __init__(self, post_process_score_thres, img_size=512, topk=50):
        self.post_process_score_thres = post_process_score_thres
        self.img_width = img_size
        self.img_height = img_size
        self.topk = topk

    def __call__(self, detection_results):
        hms, whs, offsets = detection_results['hm'], detection_results['wh'], detection_results['offsets']
        batch_size, class_num, height, width = hms.shape
        hms = NMS(hms)

        scores, inds, clses, ys, xs = _topk(hms, K=self.topk)
        offsets = _transpose_and_gather_feat(offsets, inds).view(batch_size, self.topk, 2)
        xs = xs.view(batch_size, self.topk) + offsets[:, :, 0]
        ys = xs.view(batch_size, self.topk) + offsets[:, :, 1]
        

        whs = _transpose_and_gather_feat(whs, inds).view(batch_size, self.topk, 2)
        clses = clses.view(batch_size, self.topk, 1).float()
        scores = scores.view(batch_size, self.topk, 1)

        xs = xs.view(batch_size, self.topk, 1)
        ys = ys.view(batch_size, self.topk, 1)
        width = whs[..., 0].view(batch_size, self.topk, 1)
        height = whs[..., 1].view(batch_size, self.topk, 1)

        bboxes = torch.cat([
            xs - width / 2,
            ys - height / 2,
            xs + width / 2,
            ys + height / 2,
        ], dim=2)

        bboxes[:, :, [0, 2]] *= self.img_width / width
        bboxes[:, :, [1, 3]] *= self.img_height / height

        detections = []
        for i in range(batch_size):
            bbox = bboxes[i]
            score = scores[i].reshape(-1)
            labels = clses[i].reshape(-1)
            container = Container(boxes=bbox, labels=labels, scores=score)
            container.img_width = self.img_width
            container.img_height = self.img_height
            detections.append(container)
        return detections