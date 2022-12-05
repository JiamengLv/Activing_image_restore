import argparse
import os
import random
import shutil
from collections import defaultdict
from itertools import chain

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from model.Generator import Generator
from dataset import DATASET, save_jpg


"""
active_learning_dataset：
     主要应用于长曝光数据，即可变参数的某个值对应生成的数据都一样并且生成数据不会浪费过多时间。
     采用的方法 一次生成所有数据集，每个epoch调整每个等级所用于训练的数据
     
1.dataset: 
        不同等级的数据放在不同的文件夹中，组织形式如下所示：
             ./data/  ----->/0/ ---->   image0.jpg
                     |         |
                     |          --->  image1.jpg
                     |
                      --> /1/   ---->   image0.jpg
                               |
                                --->  image1.jpg
                                 
2.一个epoch数据的数目 ：all_data = per_class_count*class_conut
3.该方法应用于jpg数据 ：image --》 [size:2*size] 清晰，模糊数据拼接在一起

"""
parser = argparse.ArgumentParser(description='train pix2pix model')

parser.add_argument('--left', type=int, default=0, help='start class')
parser.add_argument('--right', type=int, default=3, help='end class')
parser.add_argument("--per_class_count", type=int, default=10, help="class_dataset_size/epoch")
parser.add_argument("--data_path", type=str, default="./data/")

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--test_batch', type=int, default=1)
parser.add_argument('--train_batch', type=int, default=1)
parser.add_argument('--G_AB', default='', help='path to pre-trained G_AB')
parser.add_argument('--G_BA', default='', help='path to pre-trained G_BA')
parser.add_argument('--input_nc', type=int, default=3, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=3, help='channel number of output image')
parser.add_argument('--fineSize', type=int, default=424, help='random crop image to this size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=False, action='store_true', help='enables cuda')
parser.add_argument('--lambda_ABA', type=float, default=10.0, help='weight of cycle loss ABA')
parser.add_argument('--lambda_BAB', type=float, default=10.0, help='weight of cycle loss BAB')
parser.add_argument('--lambda_idt', type=float, default=5.0, help='weight of cycle loss idt')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

parser.add_argument('--outf', default='checkpoints/', help='folder to output images and model checkpoints')
parser.add_argument('--save_path', default='./out_picture/', help='folder to output images and model checkpoints')
parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--test_step', type=int, default=5)

opt = parser.parse_args()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# 创建随机树种子 random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 判断是否使用 cuda
if opt.cuda:
    print("=> use  gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
torch.manual_seed(opt.seed)
cudnn.benchmark = True


# ###########   MODEL   ###########
# 初始化网络的参数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 定义网络
G_AB = Generator(opt.input_nc, opt.output_nc, opt.ngf)
G_BA = Generator(opt.output_nc, opt.input_nc, opt.ndf)

if (opt.G_AB != ''):
    print('Warning! Loading pre-trained weights.')
    G_AB.load_state_dict(torch.load(opt.G_AB))
    G_BA.load_state_dict(torch.load(opt.G_BA))
else:
    G_AB.apply(weights_init)
    G_BA.apply(weights_init)
if (opt.cuda):
    G_AB.cuda()
    G_BA.cuda()

# ##########   LOSS & OPTIMIZER   ##########
criterionMSE = nn.L1Loss()
# chain is used to update two generators simultaneously
optimizerG = torch.optim.Adam(chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

# ###########   GLOBAL VARIABLES   ###########
real_A = torch.FloatTensor(opt.train_batch, opt.input_nc, opt.fineSize, opt.fineSize)
AB = torch.FloatTensor(opt.train_batch, opt.input_nc, opt.fineSize, opt.fineSize)
real_B = torch.FloatTensor(opt.train_batch, opt.output_nc, opt.fineSize, opt.fineSize)
BA = torch.FloatTensor(opt.train_batch, opt.output_nc, opt.fineSize, opt.fineSize)
real_A = Variable(real_A)
real_B = Variable(real_B)
AB = Variable(AB)
BA = Variable(BA)
if (opt.cuda):
    real_A = real_A.cuda()
    real_B = real_B.cuda()
    AB = AB.cuda()
    BA = BA.cuda()
    criterionMSE.cuda()


# ############   val  ############
def val(dataset, epoch):
    dataset.flag = False
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.test_batch, shuffle=True)
    loader = iter(loader)

    loss_dict = defaultdict(float)  # 损失字典：记录每个退化等级的损失
    num_dict = defaultdict(int)  # 计数字典：记录每个模糊等级的数目
    scale_dict = defaultdict(float)  # 比例字典:记录每个模糊等级所占的损失比例

    per_iter = 0

    while True:
        try:
            imgA, label = next(loader)
        except StopIteration:
            break

        real_A.resize_(imgA[:, :, 0:opt.fineSize, 0:opt.fineSize].size()).copy_(
            imgA[:, :, 0:opt.fineSize, 0:opt.fineSize])
        real_B.resize_(imgA[:, :, 0:opt.fineSize, opt.fineSize:].size()).copy_(
            imgA[:, :, 0:opt.fineSize, opt.fineSize:])

        AB = G_AB(real_A)
        BA = G_BA(real_B)

        # 计算重建损失
        loss = ((real_A - BA) ** 2)
        loss = loss.view((loss.shape[0], -1))
        loss_list = [torch.sum(x) for x in loss]

        # 将一个batch中的损失，对应的放在 r 标签中
        for i in range(opt.test_batch):
            r = label[i]
            loss = loss_list[i].data
            loss_dict[r] += loss
            num_dict[r] += 1

        # 保存测试结果
        per_iter += 1
        if epoch % 50 == 0:
            savepath = opt.save_path + '/epoch%d/out_image_train/' % epoch
            if os.path.exists(savepath):
                shutil.rmtree(savepath)
            os.makedirs(savepath)

            save_jpg(AB.data[0], 'AB_%03d_' % epoch, savepath)
            save_jpg(BA.data[0], 'BA_%03d_' % epoch, savepath)
            save_jpg(real_A.data[0], 'real_A_%03d_' % epoch, savepath)
            save_jpg(real_B.data[0], 'real_B_%03d_' % epoch, savepath)

    # 计算每个等级的平均损失，更新比例字典

    for r in loss_dict:
        loss_dict[r] /= num_dict[r]  # 计算每个模糊等级的平均损失

    total_loss = sum([loss_dict[r] for r in loss_dict])  # 每个模糊等级的平均损失的总损失

    for r in loss_dict:  # 根据平均损失和总损失更新比例字典
        scale = loss_dict[r] / total_loss
        scale_dict[r] = scale

    print("-------loss-----\n", loss_dict)
    print("-------scale_dict------ \n", scale_dict)

    filepath = './num_val20.txt'  # 记录验证集中不同等级的数据的数目
    with open(filepath, 'a+') as f:
        f.write("       -------------------epoch%d------------------        \n" % epoch)
        f.write(str(num_dict) + "\n")

    filepath = './loss_val20.txt'  # 记录验证集中不同等级的损失
    with open(filepath, 'a+') as f:
        f.write("       -------------------epoch%d------------------        \n" % epoch)
        f.write(str(loss_dict) + "\n")

    filepath = './scale_val20.txt'  # 记录验证集中不同等级的比例字典
    with open(filepath, 'a+') as f:
        f.write("       -------------------epoch%d------------------        \n" % epoch)
        f.write(str(scale_dict) + "\n")

    print(scale_dict)

    return scale_dict


# ##########   Training   ###########
G_AB.train()
G_BA.train()

# 初始字典的设置
scale_dict = {}
for i in range(opt.left, opt.right):
    scale_dict[str(i)] = 1 / (opt.right - opt.left)

for epoch in range(opt.epoch):

    dataset = DATASET(opt.data_path, scale_dict, opt.per_class_count, opt.left, opt.right)
    loader_temp = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.train_batch, shuffle=True)
    loader = iter(loader_temp)

    while True:

        try:
            imgA, label = next(loader)
        except StopIteration:
            break

        real_A.resize_(imgA[:, :, 0:opt.fineSize, 0:opt.fineSize].size()).copy_(
            imgA[:, :, 0:opt.fineSize, 0:opt.fineSize])
        real_B.resize_(imgA[:, :, 0:opt.fineSize, opt.fineSize:].size()).copy_(
            imgA[:, :, 0:opt.fineSize, opt.fineSize:])

        G_AB.zero_grad()
        G_BA.zero_grad()

        AB = G_AB(real_A)
        ABA = G_BA(AB)

        BA = G_BA(real_B)
        BAB = G_AB(BA)

        # 重建损失
        l_idt = (criterionMSE(AB, real_B) + criterionMSE(BA, real_A)) * opt.lambda_idt

        # 循环损失
        l_rec_ABA = criterionMSE(ABA, real_A) * opt.lambda_ABA
        l_rec_BAB = criterionMSE(BAB, real_B) * opt.lambda_BAB
        errMSE = l_rec_ABA + l_rec_BAB
        errG = errMSE + l_idt

        errG.backward()
        optimizerG.step()

    # 保存网络的参数
    if epoch % opt.log_step:
        torch.save(G_AB.state_dict(), '{}G_AB_{}.pth'.format(opt.outf, epoch))
        torch.save(G_BA.state_dict(), '{}G_BA_{}.pth'.format(opt.outf, epoch))

    scale_dict = val(dataset, epoch)
