# Activing_image_restore_method  

## 算法介绍
一个创新结合深度学习技术与望远镜模拟技术的星系图像复原算法。通过训练深度学习模型，能够智能处理受噪声和模糊影响的天体图像。  
![算法框图](images/method.png]


## 算法结果
### 模拟数据的测试
为了验证算法的有效性，对模拟数据进行了测试，结果显示算法能够很好地处理不同程度的噪声和模糊。
![模拟数据的复原效果示例](images/simi.png)  
### 真实数据的测试
#### 低表面亮度星系
在真实数据测试中，使用斯隆数字巡天项目（SDSS）的低表面亮度星系数据。结果表明，该算法不仅显著提升了图像的峰值信噪比（PSNR），还原了更多星系的细节，而且处理速度比传统的理查德-露西方法快100倍。
![低表面亮度星系的复原效果示例](images/low.png)  

#### SDSS R-Band 
进一步我们使用SDSS R-band 的 1024*1024 的数据，将处理前的数据和处理后的数据都用sextractor极性测光定位，结果显示经过处理后的数据拥有更高的测光和定位精度。
![测光和定位的效果示例](images/sextractor.png) 

## 安装  
  
### 依赖项  
  
* Python 3.8
* PyTorch 
* NumPy  
* SciPy  
* OpenCV  
* Astropy
* Mpi4py
  
### 安装步骤  
  
1. 克隆仓库：`git clone https://github.com/your-username/galaxy-image-restoration.git`  
2. 进入项目目录：`cd galaxy-image-restoration`  
3. 安装依赖项：`pip install -r requirements.txt`  
  
## 使用指南  
## 不使用MPI
   主要应用于长曝光数据，即可变参数的某个值对应生成的数据都一样并且生成数据不会浪费过多时间。采用的方法 一次生成所有数据集，每个epoch调整每个等级所用于训练的数据
     step 1. 准备数据集：确保你有适当格式和标注的星系图像数据集。    
        dataset: 
                不同等级的数据放在不同的文件夹中，组织形式如下所示：
                     ./data/  ----->/0/ ---->   image0.jpg
                             |         |
                             |          --->  image1.jpg
                             |
                              --> /1/   ---->   image0.jpg
                                       |
                                        --->  image1.jpg
      
      step 2. 训练模型：运'active_learning_dataset/train.py'脚本开始训练深度学习模型。  
      step 3. 复原图像：使用'active_learning_dataset/restore.py'脚本和预训练的模型来复原新的星系图像。  
## 使用MPI
      step 0.配置MPI环境。(https://python-parallel-programmning-cookbook.readthedocs.io/zh_CN/latest/)
      step 1.修改'active_learning_MPi/data_dispatch.py' 分布式生成数据的程序
      step 2. 训练模型：运行'active_learning_MPi/train.py'脚本开始训练深度学习模型。  
      step 3. 复原图像：使用'active_learning_MPi/restore.py'脚本和预训练的模型来复原新的星系图像。
    
  
