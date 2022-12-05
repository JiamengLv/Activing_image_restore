import PIL.Image as Image
from astropy.io import fits
import torchvision.transforms as transforms
import numpy as np
import torch
import os
import torch.utils.data as Data
import random
import re

#定义加载和存fits文件的类
class LoadSaveFits:
    def __init__(self,path,img,name):
        self.path = path
        self.img = img
        self.name = name
    #img是一维矩阵的标准化    
    def norm(img):
        img = (img - np.min(img))/(np.max(img) - np.min(img)) #normalization
        img -= np.mean(img)  # take the mean
        img /= np.std(img)  #standardization
        img = np.array(img,dtype='float32')
        return img
    #img是多维矩阵的标准化
    def norm2(img,z):
        for i in range(z):
            img[i] = (img[i] - np.min(img[i]))/(np.max(img[i]) - np.min(img[i])) #normalization
            img[i] -= np.mean(img[i])  # take the mean
            img[i] /= np.std(img[i])  #standardization
            img[i] = np.array(img[i],dtype='float32')
        return img
    #读fits文件  
    def read_fits(path):
        hdu = fits.open(path)
        img = hdu[0].data
        img = np.array(img,dtype = np.float32)
        hdu.close()
        return img
    #使用cpu时的存fits文件
    def save_fit_cpu(img,name,path):
        if os.path.exists(path + name+'.fits'):
            os.remove(path + name+'.fits')
        grey=fits.PrimaryHDU(img)
        greyHDU=fits.HDUList([grey])
        greyHDU.writeto(path + name+'.fits')
    #使用Gpu的存fits文件    
    def save_fit(img,name,path):
        if torch.cuda.is_available(): 
            img = torch.Tensor.cpu(img)
            img = img.data.numpy()
            IMG = img[0,0,:,:]
        else:
            img = np.array(img)
        if os.path.exists(path + name+'.fits'):
            os.remove(path + name+'.fits')
        grey=fits.PrimaryHDU(IMG)
        greyHDU=fits.HDUList([grey])
        greyHDU.writeto(path + name+'.fits')


# 加载一张fits数据（大小512）如你加载的数据大小不是512 则需要修改
class DATASET_fits():
    def __init__(self, dataPath, scale_dict, fineSize=512, left=11, right=17):
        super(DATASET_fits, self).__init__()
        train_data = []
        test_data = []
        split_val=0.8

        for r in range(left, right):

            cur_path = os.path.join(dataPath, "r_%d/" % r)
            data_list = os.listdir(cur_path)
            data_list = [cur_path + x for x in data_list]
            test_data.extend(data_list[:10])
            length = len(data_list) * (right - left) * split_val
            
            cur_count = int(scale_dict[r] * length)
            train_data.extend(data_list[10:cur_count+10])
            test_data.extend(data_list[10+cur_count:])
        
        print("The number of train_data is %d"%len(train_data))
        print("The number of test_data is %d"%len(test_data))
        
        self.train_data = train_data
        self.test_data = test_data
        self.dataPath = dataPath
        self.fineSize = fineSize
        self.flag = True

    def __getitem__(self, index):

        if self.flag:
            path = self.train_data[index]
        else:
            path = self.test_data[index]
        
        #从文件名中提取label   
        label = re.split(r"_", path)[1] 
        label = re.split(r"/", label)[0] 
                       
        #读fits文件
        img = LoadSaveFits.read_fits(path) 
        z,h,w = img.shape
        #产生随机数0-3，让图像旋转n个90度，并且改变其大小
        number_rot = random.randint(0,3)
        for zz in range(z):
            img[zz,:,:] = np.rot90(img[zz,:,:],number_rot) 
        
        img = img[:,int((h/2-self.fineSize/2)):int((h/2+self.fineSize/2)),int((w/2-self.fineSize/2)):int((w/2+self.fineSize/2))]
        img = LoadSaveFits.norm2(img,z)
        img = torch.from_numpy(img)
        
        #返回值
        return img, label
       
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        if self.flag:
            return len(self.train_data)
        else:
            return len(self.test_data)



class DATASET():

    def __init__(self, dataPath,  fineSize=512, left=5, right=20):
        super(DATASET, self).__init__()
        
        path_list = []

        for r in range(left, right):

            cur_path = os.path.join(dataPath, "r_%d/" % r)
            pa_list = [cur_path + x for x in os.listdir(cur_path)]
            path_list.extend(pa_list)

        self.path_list = path_list
        self.fineSize = fineSize

    def __getitem__(self, index):

        path = self.path_list[index]
        
        #从文件名中提取label   
        label = re.split(r"_", path)[1] 
        label = re.split(r"/", label)[0] 
                       
        #读fits文件
        img = LoadSaveFits.read_fits(path) 
        z,h,w = img.shape

        #产生随机数0-3，让图像旋转n个90度，并且改变其大小
        number_rot = random.randint(0,3)
        for zz in range(z):
            img[zz,:,:] = np.rot90(img[zz,:,:],number_rot) 

        img = img[:,int((h/2-self.fineSize/2)):int((h/2+self.fineSize/2)),int((w/2-self.fineSize/2)):int((w/2+self.fineSize/2))]
        img = LoadSaveFits.norm2(img,z)
        img = torch.from_numpy(img)
      
        #返回值
        return img, label
    
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
       return len(self.path_list)



class SUNDATASET():

    def __init__(self,sundatapath,fineSize=2048, left=5, right=20):
        super(SUNDATASET, self).__init__()

        path_list = [sundatapath + x for x in os.listdir(sundatapath)]
        self.path_list = path_list
        self.fineSize = fineSize

    def __getitem__(self, index):

        path = self.path_list[index]
 
        #读fits文件
        img = LoadSaveFits.read_fits(path) 
        z,h,w = img.shape

        #产生随机数0-3，让图像旋转n个90度，并且改变其大小
        number_rot = random.randint(0,3)
        for zz in range(z):
            img[zz,:,:] = np.rot90(img[zz,:,:],number_rot) 

        img = img[:,int((h/2-self.fineSize/2)):int((h/2+self.fineSize/2)),int((w/2-self.fineSize/2)):int((w/2+self.fineSize/2))]
        img = LoadSaveFits.norm2(img,z)
        img = torch.from_numpy(img)
      
        #返回值
        return img
    
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
       return len(self.path_list)



        

    




    
