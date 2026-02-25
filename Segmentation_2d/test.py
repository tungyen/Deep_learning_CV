import torch
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)

backbone = model.segformer.encoder

# torch.save(backbone.state_dict(), "pretrained_weights/SegFormer/mit_b2_backbone.pth")

for k in backbone.state_dict().keys():
    print(k)