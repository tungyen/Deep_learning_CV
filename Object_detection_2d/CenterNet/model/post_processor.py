import torch

from Object_detection_2d.data.container import Container

def NMS(hm):
    hm_max = torch.nn.functional.max_pool2d(hm, (3, 3), stride=1, padding=1)
    keep = (hm_max == hm).float()
    return hm * keep

class PostProcessor:
    def __init__(self, post_process_score_thres, img_size=512):
        self.post_process_score_thres = post_process_score_thres
        self.img_width = img_size
        self.img_height = img_size

    def __call__(self, detection_results):
        hms, whs, offsets = detection_results['hm'], detection_results['wh'], detection_results['offsets']
        batch_size, class_num, height, width = hms.shape
        hms = NMS(hms)
        device = hms.device

        hms = hms.permute(0, 2, 3, 1)
        whs = whs.permute(0, 2, 3, 1)
        offsets = offsets.permute(0, 2, 3, 1)

        detections = []
        for b in range(batch_size):
            hm = hms[b].view(-1, class_num)
            wh = whs[b].view(-1, 2)
            offset = offsets[b].view(-1, 2)

            yv, xv = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
            xv, yv = xv.flatten().float().to(device), yv.flatten().float().to(device)

            score, label = torch.max(hm, dim=-1)
            # print("Max of score: ", torch.max(score))
            # print("Min of score: ", torch.min(score))
            # mask = score > self.post_process_score_thres
            # print("Total objects: ", mask.sum())

            # wh_mask = wh[mask]
            # offset_mask = offset[mask]
            # label_mask = label[mask]
            # score_mask = score[mask]

            # if len(wh_mask) == 0:
            #     continue

            # xv_mask = torch.unsqueeze(xv[mask] + offset_mask[..., 0], -1)
            # yv_mask = torch.unsqueeze(yv[mask] + offset_mask[..., 1], -1)
            # half_w, half_h = wh_mask[..., 0] / 2, wh_mask[..., 1] / 2
            # bboxes = torch.cat([xv_mask-half_w, yv_mask-half_h, xv_mask+half_w, yv_mask+half_h], dim=1)

            xv = torch.unsqueeze(xv + offset[..., 0], -1)
            yv = torch.unsqueeze(yv + offset[..., 1], -1)
            half_w, half_h = wh[..., 0] / 2, wh[..., 1] / 2
            bboxes = torch.cat([xv_mask-half_w, yv_mask-half_h, xv_mask+half_w, yv_mask+half_h], dim=1)
            
            bboxes[:, [0, 2]] *= self.img_width / width
            bboxes[:, [1, 3]] *= self.img_height / height
            container = Container(boxes=bboxes, labels=label_mask, scores=score_mask)
            container.img_width = self.img_width
            container.img_height = self.img_height
            detections.append(container)
        return detections