import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class PI_dataset(Dataset):
    def __init__(self, xs_path, ys_path, img_path):
        xs = np.load(xs_path)
        ys = np.load(ys_path)
        img = np.array(Image.open(img_path))
        rgb_values = img[xs, ys]

        h, w, _ = img.shape
        self.h = h
        self.w = w
        
        data = np.column_stack((xs/(w-1), ys/(h-1), rgb_values/255))
        data = torch.tensor(data, dtype=torch.float32)
        self.data = data
        self.size = data.shape[0]
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        d = self.data[index]
        return d
        
        
        
if __name__ == '__main__':
    dataset = PI_dataset("pi_xs.npy", "pi_ys.npy", "sparse_pi_colored.jpg")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for d in dataloader:
        print(d)
        break