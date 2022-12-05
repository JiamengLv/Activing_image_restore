import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
#from utils import is_image_file
import os
from PIL import Image
import random
import re
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

def save_jpg(img,name,path):
    img = torchvision.transforms.ToPILImage()(img.float())
    save_path = path + name + ".jpg"
    img.save(save_path)

def ToTensor(pic):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backard compability
        return img.float().div(255)
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


# You should build custom dataset as below.

class DATASET(data.Dataset):
    def __init__(self, dataPath, scale_dict, left=0, right=10, start=1, gap=0.5):
        super(DATASET, self).__init__()
        # list all images into a list
        train_data = []
        test_data = []
        split_val=0.8
        
        for iter in range(left, right):

            cur_path = os.path.join(dataPath, "fwhm_%s/" % str(iter*gap+start))
            data_list = [x for x in os.listdir(cur_path) if is_image_file(x)]
            data_list = [cur_path + x for x in data_list]
            test_data.extend(data_list[:10])
            length = len(data_list) * (right - left) * split_val

            cur_count = int(scale_dict[iter] * length)
            train_data.extend(data_list[10:cur_count+10])
            test_data.extend(data_list[10+cur_count:])

        print("The number of train_data is %d"%len(train_data))
        print("The number of test_data is %d"%len(test_data))
        
        self.train_data = train_data
        self.test_data = test_data
        self.dataPath = dataPath
        self.flag = True

 
    def __getitem__(self, index):

        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        if self.flag:
            path = self.train_data[index]
        else:
            path = self.test_data[index]

        # 2. load img and label   
        label = re.split(r"_", path)[2] 
        label = re.split(r"/", label)[0] 
        print(label)
        img = default_loader(path) 
        img = ToTensor(img) # 3 x 256 x 256

        # 3. Return a data pair (e.g. image and label).
        print(label)
        return img , label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        if self.flag:
            list = self.train_data
        else:
            list = self.test_data
        return len(list)
