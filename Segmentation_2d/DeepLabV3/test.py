import torch
import argparse
import os
import numpy as np

from Segmentation_2d.dataset.utils import get_dataset
from Segmentation_2d.utils import get_model, setup_args_with_dataset
from Segmentation_2d.vis_utils import visualize_image_seg


def test_model(args):
    root = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root, "imgs")
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    model_name = args.model
    dataset_type = args.dataset
    
    args = setup_args_with_dataset(dataset_type, args)
    
    if dataset_type == 'cityscapes':
        weight_path = os.path.join(root, "ckpts", "{}_{}.pth".format(model_name, dataset_type))
    elif dataset_type == 'voc':
        weight_path = os.path.join(root, "ckpts", "{}_{}_{}.pth".format(model_name, dataset_type, args.voc_year))
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')

    print("Start testing model {} on {} dataset!".format(model_name, dataset_type))
    model = get_model(args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    _, _, test_dataloader, _, mean, std = get_dataset(args)
    imgs, _ = next(iter(test_dataloader))
    imgs_denorm = imgs * std + mean
    imgs_denorm = imgs_denorm.permute(0, 2, 3, 1).numpy()
    imgs_denorm = (imgs_denorm * 255).astype(np.uint8)
    with torch.no_grad():
        outputs = model(imgs.to(device))
        predict_class = torch.argmax(outputs, dim=1).cpu().numpy()
        visualize_image_seg(args, predict_class, imgs_denorm, save_path)


def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cityscapes")
    parse.add_argument('--crop_size', type=int, default=513)
    parse.add_argument('--voc_data_root', type=str, default="../../Dataset/VOC")
    parse.add_argument('--voc_year', type=str, default="2012")
    parse.add_argument('--voc_download', type=bool, default=False)
    parse.add_argument('--voc_crop_val', type=bool, default=True)
    parse.add_argument('--cityscapes_crop_val', type=bool, default=False)
    
    # Model
    parse.add_argument('--model', type=str, default="deeplabv3")
    parse.add_argument('--backbone', type=str, default="resnet101")
    parse.add_argument('--bn_momentum', type=float, default=0.1)
    
    # Training
    parse.add_argument('--batch_size', type=int, default=4)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    test_model(args)