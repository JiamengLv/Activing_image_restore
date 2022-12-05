import os
import random
import re

import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def default_loader(path):
    return Image.open(path).convert('RGB')


def save_jpg(img, name, path):
    img = T.ToPILImage()(img.float())
    save_path = path + name + ".jpg"
    img.save(save_path)


class DATASET(data.Dataset):
    def __init__(self, dataPath, scale_dict, per_class_count, left=0, right=10):

        """
        :param dataPath:  数据加载的路径
        :param dataset_count:  总数据集的数目
        :param scale_dict:  每类数据所占总训练集的数目
        :param left: 起始类别
        :param right: 截止类别
        """
        super(DATASET, self).__init__()
        train_data = []
        val_data = []

        for iter in range(left, right):

            cur_path = os.path.join(dataPath, "%s/" % str(iter))
            data_list = random.sample([cur_path + x for x in os.listdir(cur_path) if is_image_file(x)], per_class_count)

            # 训练集以及验证集的数据选择
            cur_count = int(scale_dict[str(iter)] * per_class_count)

            train_data.extend(data_list[1:cur_count + 1])
            val_data.extend(data_list[:1])
            val_data.extend(data_list[1 + cur_count:])

        print("The number of train_data is %d" % len(train_data))
        print("The number of val_data is %d" % len(val_data))

        self.train_data = train_data
        self.val_data = val_data
        self.dataPath = dataPath
        self.flag = True

    def __getitem__(self, index):

        if self.flag:
            path = self.train_data[index]
        else:
            path = self.val_data[index]

        # 类别标签
        label = re.split(r"/", path)[-2]

        # 数据
        img = default_loader(path)
        img = T.ToTensor()(img)

        return img, label

    def __len__(self):

        if self.flag:
            list = self.train_data
        else:
            list = self.val_data

        return len(list)
