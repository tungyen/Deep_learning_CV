from Unet import *
import os
from data import *
from torchvision.utils import save_image

unet = UNET.cuda()
weightPath = 'unet.pth'


if os.path.exists(weightPath):
        unet.load_state_dict(torch.load(weightPath))
        print("Successfully loading weight!")
else:
    print("Not successfully loading weight!")


_input = input('please input image path:')
img = imageResize(_input)
img_data = transform(img).cuda()
print(img_data.shape)

img_data = torch.unsqueeze(img_data, dim=0)
out = net(img_data)
save_image(out, 'result/result.jpg')
print(out)
    