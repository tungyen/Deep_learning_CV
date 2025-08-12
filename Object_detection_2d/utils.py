import torch
import torch.nn.functional as F
import torch.distributed as dist

from Object_detection_2d.SSD.model import SSD

def get_model(args):
    model_name = args.model
    class_num = args.class_num
    
    if model_name == "ssd":
        model = SSD(class_num=class_num)
        return model
    else:
        raise ValueError(f'Unknown model {model_name}')
            
def setup_args_with_dataset(dataset_type, args):
    if dataset_type == 'voc':
        args.class_num = 21
        args.train_batch_size = 8
        args.eval_batch_size = 8
        args.test_batch_size = 4
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    return args

def xy_to_cxcy(xy):
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2, xy[:, 2:] - xy[:, :2]], 1)

def cxcy_to_xy(cxcy):
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)

def cxcy_to_offset(cxcy, prior_cxcy):
    return torch.cat([(cxcy[:, :2] - prior_cxcy[:, :2]) / (prior_cxcy[:, 2:] / 10),
                      torch.log(cxcy[:, 2:] / prior_cxcy[:, 2:]) * 5], dim=1)

def offset_to_cxcy(offset, prior_cxcy):
    return torch.cat([offset[:, :2] * prior_cxcy[:, 2:] / 10 + prior_cxcy[:, :2],
                      torch.exp(offset[:, 2:] / 5) * prior_cxcy[:, 2:]], dim=1)

def find_boxes_intersection(boxes1, boxes2):
    lower_bound = torch.max(boxes1[:, :2].unsqueeze(1), boxes2[:, :2].unsqueeze(0))
    upper_bound = torch.min(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:].unsqueeze(0))
    intersection_size = torch.clamp(upper_bound - lower_bound, min=0)
    return intersection_size[:, :, 0] * intersection_size[:, :, 1]

def find_boxes_overlap(boxes1, boxes2):
    intersection = find_boxes_intersection(boxes1, boxes2)
    
    boxes_area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes_area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    unions = boxes_area1.unsqueeze(1) - boxes_area2.unsqueeze(0) - intersection
    return intersection / unions

def decode_boxes(args, pred_boxes, pred_scores, prior_cxcy):
    min_scores = args.min_scores
    max_overlap = args.max_overlap
    top_k = args.top_k
    
    batch_size = pred_boxes.size(0)
    device = pred_boxes.device
    prior_cxcy = prior_cxcy.to(device)
    n_priors = prior_cxcy.size(0)
    class_num = pred_scores.size(1)
    pred_boxes = pred_boxes.permute(0, 2, 1).contiguous()
    pred_scores = F.softmax(pred_scores, dim=1)

    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    assert n_priors == pred_boxes.size(1) == pred_scores.size(2)
    for i in range(batch_size):
        decoded_boxes = cxcy_to_xy(offset_to_cxcy(pred_boxes[i], prior_cxcy))

        image_boxes = list()
        image_labels = list()
        image_scores = list()
        max_scores, best_labels = pred_scores[i].max(dim=0)

        for c in range(1, class_num):
            class_scores = pred_scores[i][c, :]
            score_above_min_score = class_scores > min_scores

            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue

            class_scores = class_scores[score_above_min_score]
            class_decoded_boxes = decoded_boxes[score_above_min_score]

            class_scores, idx = class_scores.sort(dim=0, descending=True)
            class_decoded_boxes = class_decoded_boxes[idx]

            overlap = find_boxes_overlap(class_decoded_boxes, class_decoded_boxes)

            # NMS
            suppressed = torch.zeros((n_above_min_score, ), dtype=torch.bool, device=device)
            for box in range(class_decoded_boxes.size(0)):
                if suppressed[box] == 1:
                    continue
                suppressed = torch.max(suppressed, overlap[box] > max_overlap)
                suppressed[box] = 0
            image_boxes.append(class_decoded_boxes[~suppressed])
            image_labels.append(torch.tensor((~suppressed).sum().item() * [c], device=device))
            image_scores.append(class_scores[~suppressed])
        if len(image_boxes) == 0:
            images_boxes.append(torch.tensor([0., 0., 1., 1.], dtype=torch.float32, device=device))
            images_labels.append(torch.tensor([0], dtype=torch.long, device=device))
            imagew_scores.append(torch.tensor([0.], dtype=torch.float32, device=device))

        images_boxes = torch.cat(image_boxes, dim=0)
        images_labels = torch.cat(image_labels, dim=0)
        images_scores = torch.cat(image_scores, dim=0)
        n_objects = images_boxes.size(0)

        if n_objects > top_k:
            image_scores, ind = images_scores.sort(dim=0, descending=True)
            images_boxes = images_boxes[ind[:top_k]]
            images_labels = images_labels[ind[:top_k]]
            images_scores = images_scores[ind[:top_k]]
        
        all_images_boxes.append(images_boxes)
        all_images_labels.append(images_labels)
        all_images_scores.append(images_scores)

    return all_images_boxes, all_images_labels, all_images_scores

def gather_list_ddp(data):
    world_size = dist.get_world_size()
    all_data = [None for _ in range(world_size)]
    dist.all_gather_object(all_data, data)

    merged_data = list()
    for d in all_data:
        merged_data.extend(d)
    return merged_data
