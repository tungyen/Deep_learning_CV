import torch
import torch.optim as optim
from tqdm import tqdm
import argparse

from dataset import *
from model import *
from utils import *


def train_model(args):
    os.makedirs("ckpts", exist_ok=True)
    model_name = args.model
    task = model_name[-3:]
    dataset_type = args.dataset
    weight_path = "ckpts/{}_{}.pth".format(model_name, dataset_type)
    device = args.device
    lr = args.lr
    beta1= args.beta1
    beta2 = args.beta2
    eps = args.eps
    epochs = args.epochs
    weight_decay = args.weight_decay
    
    train_dataloader, val_dataloader, _, _ = get_dataset(args)
    val_num = len(val_dataloader.dataset)
    model = get_model(args)
    
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)
    criterion = get_loss(args)
    
    best_metric = 0.0
    
    for epoch in range(epochs):
        print("Epoch {} start now!".format(epoch+1))
        for pcloud, label in tqdm(train_dataloader):
            pcloud = pcloud.to(device).float()
            label = label.to(device)
            output = model(pcloud)

            loss = criterion(output, label)
            opt.zero_grad()
            loss.backward()
            opt.step()      
        print("Epoch {}-training loss===>{}".format(epoch, loss.item()))
        
        # Validation
        if task == "cls":
            acc = 0.0
            with torch.no_grad():
                for pcloud, label in tqdm(val_dataloader):
                    output = model(pcloud.to(device))
                    pred_class = torch.argmax(output, dim=1)
                    correct = torch.eq(pred_class, label.to(device)).sum().item()
                    acc += correct
            acc = acc / val_num
            print("Epoch {}-validation Acc===>{}".format(epoch+1, acc))
            if acc > best_metric:
                best_metric = acc
                torch.save(model.state_dict(), weight_path)
        elif task == "seg":
            torch.save(model.state_dict(), weight_path)
        else:
            raise ValueError(f'unknown task {task}')
        
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="modelnet40")
    parse.add_argument('--n_points', type=int, default=1500)
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet_cls")
    parse.add_argument('--class_num', type=int, default=40)
    
    # training
    parse.add_argument('--epochs', type=int, default=200)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--beta1', type=float, default=0.9)
    parse.add_argument('--beta2', type=float, default=0.999)
    parse.add_argument('--eps', type=float, default=1e-8)
    parse.add_argument('--weight_decay', type=float, default=1e-4)
    args = parse.parse_args()
    return args
                
                
if __name__ =='__main__':
    args = parse_args()
    train_model(args)
    
    
    
    