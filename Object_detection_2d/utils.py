import torch

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
        args.train_batch_size = 12
        args.eval_batch_size = 16
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