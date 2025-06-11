import os
import torch
import matplotlib.pyplot as plt
import argparse
import math
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

from utils import get_model
from Classification_2d.dataset import get_dataset, get_dataset_stat


def test_model(args):
    ckpts_path = "ckpts"
    device = args.device
    dataset_type = args.dataset
    batch_size = args.batch_size
    model_name = args.model
    weight_path = os.path.join(ckpts_path, '{}_{}.pth'.format(model_name, dataset_type))
    print("Start test model {} on {} dataset!".format(model_name, dataset_type))
    
    rows = cols = int(math.sqrt(batch_size))
    mean, std = get_dataset_stat(args)
    _, _, test_dataloader, class_dict = get_dataset(args)
    model = get_model(args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    plt.figure(figsize=(4, 4))
    for imgs, _ in test_dataloader:
        imgs_denorm = imgs * std + mean
        with torch.no_grad():
            outputs = torch.squeeze(model(imgs.to(device))).cpu()
            predicts = torch.softmax(outputs, dim=1)
            predict_classes = torch.argmax(predicts, dim=1).numpy()
            
            for i in range(batch_size):
                plt.subplot(rows, cols, i+1)
                plt.imshow(imgs_denorm[i].permute(1, 2, 0).numpy())
                plt.axis('off')
                
                class_name = class_dict[str(predict_classes[i])]
                score = predicts[i, predict_classes[i]].numpy()
                title = "{}, score: {:.2f}".format(class_name, score)
                plt.title(title, fontsize=10)
            
            plt.tight_layout()
            plt.show()
            plt.savefig('img/{}_{}.png'.format(model_name, dataset_type), bbox_inches='tight')
            break
    
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="flower")
    parse.add_argument('--data_path', type=str, default="../../Dataset/flower_data")
    
    # Model
    parse.add_argument('--model', type=str, default="resnet34")
    parse.add_argument('--img_size', type=int, default=224)
    parse.add_argument('--class_num', type=int, default=5)
    
    # evaluating
    parse.add_argument('--batch_size', type=int, default=4)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    test_model(args)
