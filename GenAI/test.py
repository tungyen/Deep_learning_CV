from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

def get_transform():
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    def transform(samples):
        images = [preprocess(img.convert("RGB")) for img in samples["image"]]
        return dict(images=images)
    return transform

dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
dataset.set_transform(get_transform())
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

sample = dataset[0]["images"]
print(sample.shape)