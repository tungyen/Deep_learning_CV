import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
import os

from Segmentation_3d.dataset.utils import get_dataset
from Segmentation_3d.utils import get_model, get_loss
from Segmentation_3d.metrics import compute_pcloud_seg_metrics, compute_pcloud_cls_metrics

def train_model(args):
    ckpts_path = "ckpts"
    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(root, ckpts_path), exist_ok=True)
    model_name = args.model
    task = model_name[-3:]
    dataset_type = args.dataset
    
    if dataset_type == 'chair':
        args.cls_class_num = 4
        args.seg_class_num = 4
        args.n_points = 1600
        args.n_feats = 0
    elif dataset_type == 'modelnet40':
        args.cls_class_num = 40
        args.seg_class_num = 40
        args.n_points = 2048
        args.n_feats = 0
    elif dataset_type == 's3dis':
        args.cls_class_num = 14
        args.seg_class_num = 14
        args.n_points = 4096
        args.n_feats = 6
    elif dataset_type == "shapenet_cls" or dataset_type == "shapenet_seg":
        args.cls_class_num = 16
        args.seg_class_num = 50
        args.n_points = 2048
        if args.normal_channel:
            args.n_feats = 3
        else:
            args.n_feats = 0
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    
    weight_path = os.path.join(root, ckpts_path, "{}_{}.pth".format(model_name, dataset_type))
    device = args.device
    lr = args.lr
    beta1= args.beta1
    beta2 = args.beta2
    eps = args.eps
    epochs = args.epochs
    weight_decay = args.weight_decay
    
    print("Start training model {} on {} dataset!".format(model_name, dataset_type))
    train_dataloader, val_dataloader, _, class_dict = get_dataset(args)
    model = get_model(args)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)
    criterion = get_loss(args)
    best_metric = 0.0
    
    for epoch in range(epochs):
        print("Epoch {} start now!".format(epoch+1))
        for pclouds, labels in tqdm(train_dataloader):
            pclouds = pclouds.to(device).float()
            labels = labels.to(device)
            outputs = model(pclouds)

            loss = criterion(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()     
        print("Epoch {}-training loss===>{:.4f}".format(epoch+1, loss.item()))
        
        # Validation
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for pclouds, labels in tqdm(val_dataloader):
                outputs = model(pclouds.to(device))
                pred_classes = torch.argmax(outputs, dim=1)
                
                all_preds.append(pred_classes.cpu())
                all_labels.append(labels)
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        if task == "cls":
            accuracy, precision, recall = compute_pcloud_cls_metrics(args, all_preds, all_labels)
            print("Epoch {}-validation Accuracy===>{:.4f}".format(epoch+1, accuracy))
            print("Epoch {}-validation Precision===>{:.4f}".format(epoch+1, precision))
            print("Epoch {}-validation Recall===>{:.4f}".format(epoch+1, recall))
            
            if precision > best_metric:
                best_metric = precision
                torch.save(model.state_dict(), weight_path)
                
        elif task == "seg":
            class_ious, miou = compute_pcloud_seg_metrics(args, all_preds, all_labels)
            print("Validation mIoU===>{:.4f}".format(miou))
            for cls in class_dict:
                print("{} IoU: {:.4f}".format(class_dict[cls], class_ious[cls]))
                
            if miou > best_metric:
                best_metric = miou
                torch.save(model.state_dict(), weight_path)   
        else:
            raise ValueError(f'unknown task {task}')
        
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="shapenet_seg")
    
    # S3DIS
    parse.add_argument('--test_area', type=int, default=5)
    parse.add_argument('--max_dropout', type=float, default=0.95)
    parse.add_argument('--block_type', type=str, default='static')
    parse.add_argument('--block_size', type=float, default=1.0)
    
    # ShapeNet
    parse.add_argument('--normal_channel', type=bool, default=True)
    parse.add_argument('--class_choice', type=list, default=None)
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet_partseg")
    
    # training
    parse.add_argument('--epochs', type=int, default=200)
    parse.add_argument('--batch_size', type=int, default=64)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--beta1', type=float, default=0.9)
    parse.add_argument('--beta2', type=float, default=0.999)
    parse.add_argument('--eps', type=float, default=1e-8)
    parse.add_argument('--weight_decay', type=float, default=1e-4)
    parse.add_argument('--loss_func', type=str, default="focal")
    args = parse.parse_args()
    return args
     
if __name__ =='__main__':
    args = parse_args()
    train_model(args)
    
    
    
    