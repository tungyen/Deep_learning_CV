import torch
from tqdm import tqdm
import os
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Segmentation_3d.dataset.utils import get_dataset
from Segmentation_3d.utils import get_model, setup_args_with_dataset
from Segmentation_3d.metrics import compute_pcloud_partseg_metrics
from Segmentation_3d.metrics import ConfusionMatrix

def eval_model(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    root = os.path.dirname(os.path.abspath(__file__))
    model_name = args.model
    dataset_type = args.dataset
    args = setup_args_with_dataset(dataset_type, args)
    task = args.task
    ckpts_path = args.experiment
    weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, task))
    
    if dist.get_rank() == 0:
        print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    _, val_dataloader, _, class_dict = get_dataset(args)
    model = get_model(args).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()
    
    if args.task[-3:] == "seg":
        confusion_matrix = ConfusionMatrix(class_num=args.seg_class_num)
    else:
        confusion_matrix = ConfusionMatrix(class_num=args.cls_class_num)
    
    with torch.no_grad():
        for pclouds, *labels in tqdm(val_dataloader, desc="Evaluation", disable=dist.get_rank() != 0):
            # Semantic Segmentation or Classification
            if len(labels) == 1:
                labels = labels[0]
                outputs, _ = model(pclouds.to(local_rank))
                pred_classes = torch.argmax(outputs, dim=1).cpu()
            # Part Segmentation
            elif len(labels) == 2:
                cls_labels, labels = labels
                instance2parts, _, label2class = class_dict
                outputs, _ = model(pclouds.to(local_rank), cls_labels.to(local_rank))
                outputs = outputs.cpu()
                pred_classes = torch.zeros((outputs.shape[0], outputs.shape[2])).astype(torch.int64)
                for i in range(outputs.shape[0]):
                    instance = label2class[cls_labels[i].item()]
                    logits = outputs[i, :, :]
                    pred_classes[i, :] = torch.argmax(logits[instance2parts[instance], :], 0) + instance2parts[instance][0]
            else:
                raise ValueError(f'Too much input data.')
            confusion_matrix.update(pred_classes.cpu(), labels)

    if dist.get_rank() == 0:
        metrics = confusion_matrix.compute_metrics()
        if task == "cls":
            precision = metrics['mean_precision']
            recall = metrics['mean_recall']
            print("Validation Precision of {} on {} ===> {:.4f}".format(model_name, dataset_type, precision))
            print("Validation Recall of {} on {} ===> {:.4f}".format(model_name, dataset_type, recall))

        elif task == "semseg":
            ious = metrics['ious']
            mious = metrics['mious']
            print("Validation mIoU of {} on {} ===> {:.4f}".format(model_name, dataset_type, mious))
            for cls in class_dict:
                print("{} IoU: {:.4f}".format(class_dict[cls], ious[cls]))

        elif task == 'partseg':
            pass
        else:
            raise ValueError(f'Unknown segmentation task {task}.')  
    dist.destroy_process_group()

def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="shapenet")
    
    # S3DIS
    parse.add_argument('--test_area', type=int, default=5)
    parse.add_argument('--max_dropout', type=float, default=0.95)
    parse.add_argument('--block_type', type=str, default='static')
    parse.add_argument('--block_size', type=float, default=1.0)
    
    # ShapeNet
    parse.add_argument('--normal_channel', type=bool, default=True)
    parse.add_argument('--class_choice', type=list, default=None)
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet")
    
    # Eval
    parse.add_argument('--experiment', type=str, required=True)
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)