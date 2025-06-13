import torch
import argparse

from dataset.utils import get_dataset
from utils import get_model
from vis_utils import *


def test_model(args):
    device = args.device
    model_name = args.model
    dataset_type = args.dataset

    model = get_model(args)
    
    print("Start testing model {} on {} dataset!".format(model_name, dataset_type))
    
    weight_path = "ckpts/{}_{}.pth".format(model_name, dataset_type)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    _, _, test_dataloader, class_dict = get_dataset(args)
    
    if dataset_type == "cityscapes":
        for imgs, _ in test_dataloader:
            with torch.no_grad():
                outputs = model(imgs.to(device).float())
                predict_class = torch.argmax(outputs, dim=1).numpy()
                
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')


def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cityscapes")
    
    # Model
    parse.add_argument('--model', type=str, default="deeplabv3")
    parse.add_argument('--class_num', type=int, default=19)
    
    # testing
    parse.add_argument('--batch_size', type=int, default=4)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
                
                
if __name__ =='__main__':
    args = parse_args()
    test_model(args)