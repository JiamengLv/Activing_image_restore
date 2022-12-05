import os

import torch
import torchvision
from PIL import Image
from torchvision import transforms

from jpg_Generator import Generator


def read_image(data_path):
    """
    load image and totensor
    :param data_path:
    :return:
    """
    data = Image.open(data_path).convert('RGB')
    data = transforms.ToTensor()(data).unsqueeze(0)

    return data


def save_jpg(img, name, path):
    """
    save jpg
    :param img:
    :param name:
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    img = (img - img.min()) / (img.max() - img.min())
    img = torchvision.transforms.ToPILImage()(img.float())
    save_path = path + name
    img.save(save_path)


def restore_jpg(data, output_nc=3, input_nc=3, ngf=16, G_BA_pth="./checkpoints/jpg/G_BA_140000.pth"):

    # model_set
    output_nc, input_nc, ngf = output_nc, input_nc, ngf
    G_BA_pth = G_BA_pth

    Restore = Generator(output_nc, input_nc, ngf)
    Restore.load_state_dict(torch.load(G_BA_pth, map_location="cpu"))

    output_img = Restore(data)

    return output_img


if __name__ == "__main__":

    data_path = "./figure/jpg_sun/"
    out_path = "./out/jpg/"

    data_path_list = os.listdir(data_path)

    for data_path_name in data_path_list:
        data_path_name1 = os.path.join(data_path, data_path_name)
        data = read_image(data_path_name1)
        out_data = restore_jpg(data)

        save_jpg(out_data[0], "re" + data_path_name, out_path)

        pass
