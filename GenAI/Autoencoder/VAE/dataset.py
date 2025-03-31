import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class CelebA_dataset(Dataset):
    def __init__(self, root, img_size=64):
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.files = sorted(os.listdir(root))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        img = Image.open(img_path).convert('RGB')
        t = transforms.Compose([
            transforms.Resize(self.img_size, antialias=True),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor()
        ])
        
        return t(img)
    
    
if __name__ == '__main__':
    dataset = CelebA_dataset(root="/project/Dataset/img_align_celeba")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for img in dataloader:
        print(img.shape)
        break