#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import glob
import os
import random

import cv2
import numpy as np
from astropy.convolution import Gaussian2DKernel, Moffat2DKernel
from astropy.io import fits

# 加载参数
parser = argparse.ArgumentParser()


# 读fits文件
def read_fits(path):
    hdu = fits.open(path)
    img = hdu[0].data
    img = np.array(img, dtype=np.float32)
    hdu.close()
    return img


def Gauss_psf(fwhm):
    """
    高斯(Gaussian) psf  ----  仅考虑设备的影响
    Gaussian2DKernel(x_stddev, y_stddev=None, theta=0.0, **kwargs)
    x_stddev 高斯函数的标准差.
    theta 旋转角度，以弧度为单位
    FWHM = 2.355*sigma
    """
    sigma = fwhm / 2.355
    gaussian_2D_kernel = Gaussian2DKernel(sigma)

    return gaussian_2D_kernel.array


def Moffat_psf(fwhm):
    """
    Moffat psf  ----  考虑设备和大气湍流的综合影响
    Moffat2DKernel(gamma, alpha, **kwargs)
    α(gamma)和β(alpha)为尺度因子，可以控制函数的弥散程度
    β(alpha) 默认为4.765，可以很好地模拟大气扰动
    FWHM=2.0 * np.abs(gamma) * np.sqrt(2.0 ** (1.0 / alpha) - 1.0)
    """
    alpha = 4.765
    gamma = (fwhm / 2.0) ** 2 / 2.0 ** ((1.0 / alpha) - 1.0)
    moffat_2D_kernel = Moffat2DKernel(gamma, alpha)

    return moffat_2D_kernel.array


def adjust(origin):
    img = origin.copy()
    img[img > 4] = 4
    img[img < -0.1] = -0.1
    MIN = np.min(img)
    MAX = np.max(img)
    img = np.arcsinh(10 * (img - MIN) / (MAX - MIN)) / 3
    return img


def roou():
    parser.add_argument("--fwhm", default=5)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument("--sig", default="3")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--number", default=100)  # 每个epoch 训练的数据 从1000张 数据中 提取出来
    parser.add_argument("--input", default="/fitsdata/")  # ./fits_0.01_0.02
    parser.add_argument("--figure", default="figures")  # output_path  ==> figures01_0.2
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--model", default="models")
    args = parser.parse_args()

    fwhm = float(args.fwhm)
    sig = float(args.sig)

    # 输入地址与输出地址
    if args.mode == "train":

        input = args.input + "/fits_train/"
        args.figure = args.figure + "/get_%d/fwhm_%s/" % (args.epoch, args.fwhm)
        # 加载数据
        fits = '%s/*/*-g.fits' % (input)
        # glob.glob 返回list ，glob.iglob 返回一个迭代器
        files = glob.glob(fits)
        # 从生成的路径list中随机选取 number的数据
        files_path = random.sample(files, args.number)

    elif args.mode == "test":

        input = args.input + "/fits_test/"
        args.figure = args.figure + "/fwhm_%s/" % (args.fwhm)
        fits = '%s/*/*-g.fits' % (input)
        # glob.glob 返回list ，glob.iglob 返回一个迭代器
        files_path = glob.glob(fits)

    figure = args.figure
    if not os.path.exists('./' + args.figure):
        os.makedirs("./" + args.figure)

    for i, iter in enumerate(files_path):
        # os.path.basename() 返回path最后的文件名。如果path以／或\结尾，那么就会返回空值。
        file_name = os.path.basename(iter)

        filename = file_name.replace("-g.fits", '')
        filename_g = '%s/%s/%s-g.fits' % (input, filename, filename)
        filename_r = '%s/%s/%s-r.fits' % (input, filename, filename)
        filename_i = '%s/%s/%s-i.fits' % (input, filename, filename)

        gfits = read_fits(filename_g)
        rfits = read_fits(filename_r)
        ifits = read_fits(filename_i)
        data_g = gfits.data
        data_r = rfits.data
        data_i = ifits.data

        MAX = 4
        MIN = -0.1

        # 原始数据
        figure_original = np.ones((data_g.shape[0], data_g.shape[1], 3))

        figure_original[:, :, 0] = data_g
        figure_original[:, :, 1] = data_r
        figure_original[:, :, 2] = data_i

        # 1. gaussian 模糊 
        # gaussian_2D_kernel = Gauss_psf(fwhm)
        # figure_blurred = cv2.filter2D(figure_original,-1,gaussian_2D_kernel)
        # 2. moffat 模糊 
        moffat_2D_kernel = Moffat_psf(fwhm)
        # cv2.filter2D(original,depth,kernel) 原始图像，目标图像的深度，卷积核
        # 当ddepth=-1时，表示输出图像与原图像有相同的深度
        figure_blurred = cv2.filter2D(figure_original, -1, moffat_2D_kernel)

        # figure_blurred = np.ones((data_g.shape[0],data_g.shape[1],3))
        # figure_blurred[:,:,0] = figure_original[:,:,0]
        # figure_blurred[:,:,1] = figure_original[:,:,1]
        # figure_blurred[:,:,2] = figure_original[:,:,2]

        # # add white noise
        # figure_original_nz =  figure_original[figure_original<0.1]
        # figure_original_nearzero = figure_original_nz[figure_original_nz>-0.1]
        # figure_blurred_nz = figure_blurred[figure_blurred<0.1]
        # figure_blurred_nearzero = figure_blurred_nz[figure_blurred_nz>-0.1]
        # [m,s] = norm.fit(figure_original_nearzero)
        # [m2,s2] = norm.fit(figure_blurred_nearzero)

        # whitenoise_var = (sig*s)**2-s2**2
        # print(whitenoise_var)

        # if whitenoise_var < 0:
        #     whitenoise_var = 0.00000001

        # whitenoise =  np.random.normal(0, np.sqrt(whitenoise_var),figure_original[:,:,0].shape)

        # # add possionnoise

        # poissonnoise = numpy.random.poisson(lam=0.05,size=(figure_original[:,:,0].shape)).astype(float)

        # noise = whitenoise
        # figure_blurred[:,:,0] = figure_original[:,:,0] + noise
        # figure_blurred[:,:,1] = figure_original[:,:,1] + noise
        # figure_blurred[:,:,2] = figure_original[:,:,2] + noise

        # normalize figures
        figure_original = (figure_original - MIN) / (MAX - MIN)
        figure_blurred = (figure_blurred - MIN) / (MAX - MIN)
        # asinh scaling
        figure_blurred = np.arcsinh(10 * figure_blurred) / 3
        figure_original = np.arcsinh(10 * figure_original) / 3

        # 将模糊数据和清晰数据拼接在一起
        figure_combined = np.zeros((data_g.shape[0], data_g.shape[1] * 2, 3))
        figure_combined[:, : data_g.shape[1], :] = figure_original[:, :, :]
        figure_combined[:, data_g.shape[1]:2 * data_g.shape[1], :] = figure_blurred[:, :, :]

        # 保存图片
        jpg_path = './%sfwhm_%s_%d.jpg' % (figure, fwhm, i)
        image = (figure_combined * 256).astype(np.int)
        cv2.imwrite(jpg_path, image)


roou()
