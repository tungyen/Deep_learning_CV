from collections import OrderedDict
import torch

def remove_attention_self(old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        new_k = k
        # Remove '.self' from attention keys
        new_k = new_k.replace(".attention.self.", ".attention.")
        # Rename output.dense → proj
        new_k = new_k.replace("attention.output.dense", "attention.proj")
        new_state_dict[new_k] = v
    return new_state_dict

# Usage
old_state_dict = torch.load("pretrained_weights/SegFormer/mit_b5_backbone.pth", map_location="cpu")
new_state_dict = remove_attention_self(old_state_dict)

torch.save(new_state_dict, "pretrained_weights/SegFormer2/mit_b5_backbone.pth")