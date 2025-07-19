import torch
from tqdm import tqdm
import argparse
import os

from Segmentation_2d.dataset.utils import get_dataset
from Segmentation_2d.utils import get_model, setup_args_with_dataset
from Segmentation_2d.metrics import compute_image_seg_metrics


def eval_model(args):
    model_name = args.model
    dataset_type = args.dataset
    root = os.path.dirname(os.path.abspath(__file__))
    args = setup_args_with_dataset(dataset_type, args)
    ckpts_path = args.experiment
    
    if dataset_type == 'cityscapes':
        weight_path = os.path.join(root, ckpts_path, "{}_{}.pth".format(model_name, dataset_type))
    elif dataset_type == 'voc':
        weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, args.voc_year))
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    
    device = args.device
    _, val_dataloader, _, class_dict, _, _ = get_dataset(args)
    model = get_model(args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluation"):
            output = model(imgs.to(device))
            pred_class = torch.argmax(output, dim=1)
            
            all_preds.append(pred_class.cpu())
            all_labels.append(labels)
        
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    class_ious_dict, miou = compute_image_seg_metrics(args, all_preds, all_labels)
    print("Validation mIoU===>{:.4f}".format(miou))
    for cls, iou in class_ious_dict.items():
        print("{} IoU: {:.4f}".format(class_dict[cls], iou))
    
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cityscapes")
    parse.add_argument('--crop_size', type=int, default=513)
    parse.add_argument('--voc_data_root', type=str, default="Dataset/VOC")
    parse.add_argument('--voc_year', type=str, default="2012_aug")
    parse.add_argument('--voc_download', type=bool, default=False)
    parse.add_argument('--voc_crop_val', type=bool, default=True)
    parse.add_argument('--cityscapes_crop_val', type=bool, default=True)
    
    # Model
    parse.add_argument('--model', type=str, default="deeplabv3")
    parse.add_argument('--backbone', type=str, default="resnet101")
    parse.add_argument('--bn_momentum', type=float, default=0.1)
    
    # Validation
    parse.add_argument('--experiment', type=str, required=True)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    eval_model(args)