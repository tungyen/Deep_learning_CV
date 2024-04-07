from Unet import *
import os
from data import *
from torchvision.utils import save_image
from PIL import Image
import numpy as np

def unet_test():
    unet = UNET().cuda()
    weightPath = 'unet.pth'


    if os.path.exists(weightPath):
            unet.load_state_dict(torch.load(weightPath))
            print("Successfully loading weight!")
    else:
        print("Not successfully loading weight!")


    _input = "plane.jpg"
    img = imageResize(_input)
    img_data = transform(img).cuda()

    img_data = torch.unsqueeze(img_data, dim=0)
    out = torch.squeeze(unet(img_data))
    out = torch.softmax(out, dim=0).cpu()
    predict_cla = torch.argmax(out, dim=0).numpy()
    print(np.sum(predict_cla))
#     res = Image.fromarray(predict_cla).convert('RGB')
#     res.show()
    
if __name__ == '__main__':
    unet_test()
    