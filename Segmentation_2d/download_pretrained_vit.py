import argparse
import os
import torch
import timm

from collections import OrderedDict

backbone_dict = {
    "vit_tiny_patch16_384": "vit_tiny_patch16_384",
    "vit_small_patch16_384": "vit_small_patch16_384",
    "vit_base_patch16_384": "vit_base_patch16_384",
    "vit_large_patch16_384": "vit_large_patch16_384",
}

def remove_head(state_dict):
    return OrderedDict({k: v for k, v in state_dict.items() if not k.startswith("head")})

def main(args):
    backbone_name = args.backbone_name
    timm_name = backbone_dict.get(backbone_name, None)
    if timm_name is None:
        raise ValueError(
            f"Invalid backbone: '{backbone_name}'. "
            f"Valid options: {list(backbone_dict.keys())}"
        )

    print(f"Downloading '{timm_name}' from timm...")
    model = timm.create_model(timm_name, pretrained=True)
    model.eval()

    save_dir = os.path.join("pretrained_weights", "Segmenter")
    os.makedirs(save_dir, exist_ok=True)

    full_path = os.path.join(save_dir, f"{backbone_name}_full.pth")
    torch.save(model.state_dict(), full_path)
    print(f"Full weights saved to: {full_path}")

    # Save backbone-only weights (head removed) — what Segmenter actually loads
    backbone_state = remove_head(model.state_dict())
    backbone_path = os.path.join(save_dir, f"{backbone_name}_backbone.pth")
    torch.save(backbone_state, backbone_path)
    print(f"Backbone-only weights saved to: {backbone_path}")

    for key in backbone_state.keys():
        print(key)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Download pretrained ViT weights from timm for Segmenter"
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="vit_small_patch16_384",
        help=(
            "Backbone name. Options: "
            "vit_tiny_patch16_384, vit_small_patch16_384, "
            "vit_base_patch16_384, vit_large_patch16_384"
        ),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)