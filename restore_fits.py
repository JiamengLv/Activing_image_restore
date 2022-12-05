import os

import torch
import torchvision
from PIL import Image
from torchvision import transforms

from model.fits_Generator import Generator
from astropy.io import fits
import cv2
import numpy as np



class LoadSaveFits:

    def __init__(self, path, img, name):
        self.path = path
        self.img = img
        self.name = name

    def norm(img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # normalization
        # img -= np.mean(img)  # take the mean
        # img /= np.std(img)  # standardization
        img = np.array(img, dtype='float32')
        return img

    def read_fits(path):
        hdu = fits.open(path)
        img = hdu[0].data
        img = np.array(img, dtype=np.float32)
        hdu.close()
        return img

    def save_fit(img, name, path):

        if not os.path.exists(path):
            os.makedirs(path)

        if os.path.exists(path+name):
            os.remove(path+name)

        grey = fits.PrimaryHDU(img)
        greyHDU = fits.HDUList([grey])
        greyHDU.writeto(path + name)

    def fitstojpg(fits_data,  savepath):

        MAX = 4
        MIN = -0.1

        # normalize figures
        fits_data = (fits_data - MIN) / (MAX - MIN)
        # asinh scaling
        fits_data = np.arcsinh(10 * fits_data) / 3
        fits_data = fits_data * 200

        cv2.imwrite(savepath, fits_data.transpose(1, 2, 0)[:, :, :3])


def read_fits(path):

    data = LoadSaveFits.read_fits(path)
    # data = np.expand_dims(data,0).repeat(5,axis=0)
    data = torch.tensor(LoadSaveFits.norm(data)).unsqueeze(0)

    return data

def restore_fits(data, output_nc=5, input_nc=5, ngf=64, G_BA_pth="./checkpoints/fits/G_BA_900.pth"):

    # model_set
    output_nc, input_nc, ngf = output_nc, input_nc, ngf
    G_BA_pth = G_BA_pth

    Restore = Generator(output_nc, input_nc, ngf)
    Restore.load_state_dict(torch.load(G_BA_pth, map_location="cpu"))

    output_img = Restore(data)

    return output_img


if __name__ == "__main__":

    data_path = "./figure/fits_figure/blur/"
    label_path = "./figure/fits_figure/clear/"

    out_path = "./out/fits/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    data_path_list = os.listdir(data_path)

    for data_path_name in data_path_list:

        data_path_name1 = os.path.join(data_path, data_path_name)
        data = read_fits(data_path_name1)
        out_data = restore_fits(data)

        data_path_name12 = os.path.join(label_path, data_path_name)
        label = LoadSaveFits.read_fits(data_path_name12)
        input = LoadSaveFits.read_fits(data_path_name1)

        # LoadSaveFits.save_fit(data[0].detach().numpy(), data_path_name, out_path)
        # LoadSaveFits.save_fit(out_data[0].detach().numpy(), "re" + data_path_name, out_path)

        LoadSaveFits.fitstojpg(out_data[0].detach().numpy(), out_path + data_path_name + "out.jpg")
        LoadSaveFits.fitstojpg(input, out_path + data_path_name + "input.jpg")
        LoadSaveFits.fitstojpg(label,out_path + data_path_name + "label.jpg")

