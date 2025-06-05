import torch
from tqdm import tqdm
import os
import argparse
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)


from utils import get_model
from Classification_2d.dataset import get_dataset
from Classification_2d.metrics import compute_image_cls_metrics

def eval_model(args):
    ckpts_path = "ckpts"
    device = args.device
    model_name = args.model
    dataset_type = args.dataset
    weight_path = os.path.join(ckpts_path, '{}_{}.pth'.format(model_name, dataset_type))
    
    print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    
    _, val_dataloader, _, _ = get_dataset(args)
    model = get_model(args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img, label in tqdm(val_dataloader):
            output = model(img.to(device))
            pred_class = torch.argmax(output, dim=1)
            
            all_preds.append(pred_class.cpu())
            all_labels.append(label)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
            
    accuracy, precision, recall = compute_image_cls_metrics(args, all_preds, all_labels)
    print("Validation Accuracy===>{:.4f}".format(accuracy))
    print("Validation Precision===>{:.4f}".format(precision))
    print("Validation Recall===>{:.4f}".format(recall))

            
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cifar100")
    parse.add_argument('--data_path', type=str, default="../../Dataset/flower_data")
    
    # Model
    parse.add_argument('--model', type=str, default="vit_rope")
    parse.add_argument('--img_size', type=int, default=32)
    parse.add_argument('--patch_size', type=int, default=4)
    parse.add_argument('--class_num', type=int, default=100)
    
    # evaluating
    parse.add_argument('--batch_size', type=int, default=128)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)