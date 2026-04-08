import argparse
import os
import torch

from collections import OrderedDict
from transformers import SegformerForSemanticSegmentation

def remove_attention_self(old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        new_k = k
        new_k = new_k.replace(".attention.self.", ".attention.")
        new_k = new_k.replace("attention.output.dense", "attention.proj")
        new_state_dict[new_k] = v
    return new_state_dict

backbone_dict = {
    "mit_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "mit_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "mit_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "mit_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "mit_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "mit_b5": "nvidia/segformer-b5-finetuned-ade-640-640",
}

def main(args):
    backbone_name = args.backbone_name
    model_name = backbone_dict.get(backbone_name, None)
    if model_name is None:
        raise ValueError(f"Invalid backbone name: {backbone_name}. Valid options are: {list(backbone_dict.keys())}")

    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    weight_name = backbone_name + "_backbone.pth"
    os.makedirs("pretrained_weights", exist_ok=True)
    os.makedirs("pretrained_weights/SegFormer", exist_ok=True)

    backbone = model.segformer.encoder
    torch.save(backbone.state_dict(), "pretrained_weights/SegFormer/" + weight_name)

    old_state_dict = torch.load("pretrained_weights/SegFormer/" + weight_name, map_location="cpu")
    new_state_dict = remove_attention_self(old_state_dict)

    torch.save(new_state_dict, "pretrained_weights/SegFormer2/" + weight_name)

def parse_args():
    parser = argparse.ArgumentParser(description="Download pretrained weights for SegFormer backbone")
    parser.add_argument("--backbone_name", type=str, default="mit_b0", help="Name of the backbone (e.g., mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)