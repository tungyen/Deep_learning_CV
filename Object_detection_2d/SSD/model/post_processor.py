import torch
from Object_detection_2d.data.container import Container
from Object_detection_2d.SSD.utils import batched_nms
import time
class PostProcessor:
    def __init__(self, img_size=300, confidence_thres=0.01,
                 nms_thres=0.45, topk=100):
        self.width = img_size
        self.height = img_size
        self.confidence_thres = confidence_thres
        self.nms_thres = nms_thres
        self.topk = topk

    def __call__(self, detection_results):
        pred_boxes, pred_scores = detection_results
        device = pred_boxes.device
        batch_size = pred_boxes.shape[0]
        results = []

        for i in range(batch_size):
            batch_start_time = time.time()
            boxes, scores = pred_boxes[i], pred_scores[i]
            boxes_number = boxes.shape[0]
            class_num = scores.shape[1]

            boxes = boxes.view(boxes_number, 1, 4).expand(boxes_number, class_num, 4)
            labels = torch.arange(class_num, device=device)
            labels = labels.view(1, class_num).expand_as(scores)

            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            indexes = torch.nonzero(scores > self.confidence_thres).squeeze(1)
            boxes, scores, labels = boxes[indexes], scores[indexes], labels[indexes]

            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height
            nms_start_time = time.time()
            keep = batched_nms(boxes, scores, labels, self.nms_thres)
            keep = keep[:self.topk]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            container = Container(boxes=boxes, labels=labels, scores=scores)
            container.img_width = self.width
            container.img_height = self.height
            results.append(container)
        return results