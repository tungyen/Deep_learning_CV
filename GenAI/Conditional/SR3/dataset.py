import random
from PIL import Image
from os import listdir
from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class DIV2K_Flickr2K_Dataset(Dataset):
    def __init__(self, root, data_name="DIV2K", crop_size=64, mode="train"):
        super().__init__()
        if data_name == "DIV2K" and root[-5:] == data_name:
            img_path = join(root, "DIV2K_"+mode+"_LR_bicubic", "X4")
            gt_path = join(root, "DIV2K_"+mode+"_HR")
        elif data_name == "Flickr2K" and root[-8:] == data_name:
            if mode == "train":
                img_path = join(root, "Flickr2K_LR_bicubic", "X4_2650")
                gt_path = join(root, "Flickr2K_HR_2650")
            else:
                img_path = join(root, "Flickr2K_LR_bicubic", "X4_1000")
                gt_path = join(root, "Flickr2K_HR_1000")
        else:
            raise ValueError(f'unknown dataset {data_name}')
        
        self.img_names = sorted([name for name in listdir(img_path)])
        self.gt_names = sorted([name for name in listdir(gt_path)])
        self.crop_size = crop_size
        self.mode = mode
        self.img_path = img_path
        self.gt_path = gt_path
        
        self.upsample_fac = 4
        self.tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img = self.tensor(Image.open(join(self.img_path, self.img_names[idx])))
        gt = self.tensor(Image.open(join(self.gt_path, self.gt_names[idx])))
        
        if self.mode == "train":
            params = transforms.RandomCrop(self.crop_size).get_params(img, (self.crop_size, self.crop_size))
            img = transforms.functional.crop(img, *params)
            gt = transforms.functional.crop(gt, *[self.upsample_fac*p for p in params])
            
            if random.random() < 0.5:
                img = torch.flip(img, [2])
                gt = torch.flip(gt, [2])
                
            angle = float(90 * random.randint(0, 3))
            img = transforms.functional.rotate(img, angle)
            gt = transforms.functional.rotate(gt, angle)
            
        elif self.mode == "valid" and self.crop_size is not None:
            img = transforms.CenterCrop(self.crop_size)(img)
            gt = transforms.CenterCrop(self.upsample_fac*self.crop_size)(gt)
        
        return img, gt
    
    
class DF2KDataset(Dataset):
    def __init__(self, div_root, flickr_root, crop_size=64):
        super().__init__()
        self.div2k = DIV2K_Flickr2K_Dataset(root=div_root, data_name="DIV2K", crop_size=crop_size, mode="train")
        self.flickr2k = DIV2K_Flickr2K_Dataset(root=flickr_root, data_name="Flickr2K", crop_size=crop_size)
        self.div_len = len(self.div2k)
        self.flickr_len = len(self.flickr2k)
        self.total_len = self.div_len + self.flickr_len
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < self.div_len:
            return self.div2k.__getitem__(idx)
        else:
           return self.flickr2k.__getitem__(idx-self.div_len)
       
       
if __name__ == '__main__':
    div_root = "../../../Dataset/DIV2K"
    flickr_root = "../../../Dataset/Flickr2K"
    dataset = DF2KDataset(div_root, flickr_root)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(len(dataloader))
    for img, gt in dataloader:
        print(img.shape)
        print(gt.shape)
        break
        
        