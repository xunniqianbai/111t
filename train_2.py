import torch
from torch import nn
from torchvision import datasets, transforms #导入Mnist数据集
from torch.nn import functional
import torch.utils.data as data
BATCH_SIZE = 1
import os
from PIL import Image, ImageFilter
import numpy as np
import torch.nn.functional as F
from network2.my_resnet import resnet50

from network2.model_geng2 import Model_geng2
from model.cv_model import CV_chan

# 设置超参数
K = 20
dataset_path = r"./dataset/train.csv"


# 加载权重文件
def load_weight(net,pth_path):
    # 方式3，根据tensor的形状相同加载权重
    # print(net.conv1.state_dict())
    orginal_dict = net.state_dict() #当前网络的权重字典。
    weight_dict = torch.load(pth_path) #读取的网络权重字典
    # 通过形状相同，把orignal_dict对应的tensor 换成 weight_dict的tensor。


    for key,value in orginal_dict.items():

        for key2,value2 in weight_dict.items():
            if value2.size() == value.size():
                # print("形状相同")
                orginal_dict[key] = weight_dict[key2] # 将orginal换成weight_dict
                weight_dict[key2] = torch.randn(1,1,1,1,1)  #？

    net.load_state_dict(orginal_dict)


# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
import torch
import math
import torch.nn as nn

# 进行高斯模糊
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def read_data_interface(file_path):
    return read_data_from_implement_csv(file_path)

def read_data_from_implement_excel(file_path):
    pass

def read_data_from_implement_csv(file_path):
    data_ = []

    with open(file_path) as f:
        for line_ in f.readlines():
            # print(line_)
            line_ = line_.strip()
            x_ , y_ = float(line_.split(",")[0]),float(line_.split(",")[1])
            data_.append([x_,y_])

    while len(data_) < K:
        data_.append([0,0])

    data_ = np.array(data_)
    return data_

class MyGaussianBlur(ImageFilter.Filter):
  name = "GaussianBlur"

  def __init__(self, radius=2, bounds=None):
    self.radius = radius
    self.bounds = bounds

  def filter(self, image):
    if self.bounds:
      clips = image.crop(self.bounds).gaussian_blur(self.radius)
      image.paste(clips, self.bounds)
      return image
    else:
      return image.gaussian_blur(self.radius)


class MyDataSet(data.Dataset):

    def __init__(self, root_path , input_size = 256):
        image_name_list = []
        label_name_list = []
        seg_name_list = []

        with open(root_path,encoding="utf-8") as f:
            for line in f.readlines():
                image_path , seg_path = line.strip().split(',')
                image_name_list.append(image_path)
                label_name_list.append(seg_path)
                seg_name_list.append(seg_path)

        self.image_name_list = image_name_list
        self.label_name_list = label_name_list
        self.seg_name_list = seg_name_list

        self.input_size = input_size

    def __getitem__(self, index):
        img_path   = self.image_name_list[index]
        img = Image.open(img_path).convert('L')
        img = img.resize((self.input_size,self.input_size))

        img = np.array(img)  #当使用PIL.Image.open()打开图片后，如果要使用img.shape函数，需要先将image形式转换成array数组
        img = np.expand_dims(img, 2) #(m, n, c)->(m, n, 1, c)
        # 对于256 * 256的三通道彩色图像，cv2以及PIL.Image等读出的numpy格式图像，通道数都在最后(256, 256, 3)
        # 而有时候会需要channelfirst，(3, 256, 256)就可以使用
        img = np.transpose(img, (2, 0, 1)) #图片矩阵变换
        img = img/255  #
        img = torch.tensor(img,dtype=torch.float32)

        # target_path = self.label_name_list[index]
        # target = read_data_interface(target_path)
        # target = target.reshape(-1,1)
        # target = target.squeeze()
        # # target = target.reshape(-1,2) #恢复
        # target1 = torch.tensor(target,dtype=torch.float32)
        target1 = torch.randn(1) #历史遗留问题

        # 读取分割的标签
        target_path_seg = self.seg_name_list[index]
        img_label_origion = Image.open(target_path_seg).convert('L')


        img_label = img_label_origion.resize((self.input_size, self.input_size))
        img_label = np.array(img_label)
        img_label = np.expand_dims(img_label, 2)
        img_label = np.transpose(img_label, (2, 0, 1))
        img_label = (img_label) / 255
        img_label = torch.tensor(img_label, dtype=torch.float32)
        # img_label[0,1] => [-1,1]
        target2 = img_label * 2 - 1 #GT2




        # img = torch.randn([1, 3, 64, 64])#.cuda()

        img_label_origion = Image.open(target_path_seg).convert('L')
        img_label_2 = img_label_origion.resize(( 32, 32 ))  #缩放
        img_label_2 = np.array(img_label_2)
        img_label_2 = np.expand_dims(img_label_2, 2)
        img_label_2 = np.transpose(img_label_2, (2, 0, 1))
        img_label_2 = (img_label_2) / 255
        img_label_2 = torch.tensor(img_label_2, dtype=torch.float32)
        target3 = img_label_2

        target3 = target3.unsqueeze(0)
        blur_layer = get_gaussian_kernel()#.cuda()
        blured_img = blur_layer(target3)
        target3 = blured_img.squeeze(0) * 2 -1  #GT3



        return img, target1 , target2 , target3

    def __len__(self):
        return len(self.image_name_list)

# 自定义损失函数，交叉熵损失函数
class MyEntropyLoss(nn.Module):

    def forward(self,output,target):
        batch_size_ = output.size()[0]
        num_class = output[0].size()[0] #获得类别数量
        label_one_hot = functional.one_hot(target, num_classes=num_class) #转换为独热吗

        loss = (output-label_one_hot)**2/num_class
        return torch.sum(loss)/batch_size_

class MyKLLos(nn.Module):
    def forward(self,x,y):
        kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
        return kl


def train_one_epoch(train_loader,epoch_num,level_model,device):
    loss_list = []
    # each_data 代表输入图片 256*256*1
    # each_label2  256*256的标签
    # each_label3  32*32的标签
    for i, (each_data, each_label1, each_label2,each_label3) in enumerate(train_loader):
        # 梯度清零，这一步必须要操作，因为不操作则会保留上一次训练的信息
        optimizer.zero_grad()

        img = each_data.to(device)
        each_label2 = each_label2.to(device)
        each_label3 = each_label3.to(device)
        # each_data # 获取数据
        # each_label # 获取标签

        # each_data = torch.randn(8, 3, 256, 256)
        # step4.进行前向传播，获取预测值
        # pred_x 表示CV模型的3个参数
        # pred_y 表示初始轮廓 32*32的大小
        # pred_y_256 表示初始轮廓 256*256的大小
        # pred_x, pred_y,pred_y_256 = net_gen(each_data)  # 预测的结果
        x_length, x_penalty, x_cvterm, pred_y,pred_y_256= net_gen(each_data)

        # 获得初始轮廓
        # IniLSF = np.ones((img.shape[2], img.shape[3]), np.float64)
        # IniLSF[20:40, 20:40] = -1  # 设置初始轮廓
        # IniLSF = -IniLSF
        # IniLSF_tensor = torch.from_numpy(IniLSF)  # 转换为tensor
        # LSF_tensor = IniLSF_tensor


        # 预测出来的初始轮廓
        LSF_tensor = pred_y_256.squeeze(0)
        LSF_tensor = LSF_tensor.squeeze(0)

        x_length = x_length.squeeze(0).squeeze(0)
        x_penalty = x_penalty.squeeze(0).squeeze(0)
        x_cvterm  = x_cvterm.squeeze(0).squeeze(0)
        #


        # 通过水平集预测出来的结果
        img_tensor = img.squeeze(1)
        # 通过CV模型得到的输出 LSF_tensor_output
        LSF_tensor_output = level_model(LSF_tensor, img_tensor, x_length, x_penalty, x_cvterm)


        # level_model(LSF_tensor,img,mu,nu,epison)

        #print("test")



        # step5.计算损失函数，反向传播，进行梯度下降，将之前的梯度清空
        # 计算KL散度前，需要对输出进行归一化处理   (x(i) - min)/(max-min)
        LSF_tensor_output = (LSF_tensor_output - LSF_tensor_output.min()) / (LSF_tensor_output.max() - LSF_tensor_output.min()) + 0.001
        each_label2 = (each_label2 - each_label2.min()) / (each_label2.max() - each_label2.min()) + 0.001
        loss1 = loss_func_kl(LSF_tensor_output, each_label2)  # 计算损失值

        # loss2 = loss_func_kl(pred_y,each_label2) # 计算损失值
        # loss = loss1 + loss2

        pred_y = (pred_y - pred_y.min()) / (
                    pred_y.max() - pred_y.min()) + 0.001
        each_label3 = (each_label3 - each_label3.min()) / (each_label3.max() - each_label3.min()) + 0.001
        loss2 = loss_func_kl(pred_y.squeeze(0),each_label3)

        loss = loss1 + loss2
        print(f"epoch:{epoch_num},[{i}],loss:{loss}/{loss1}:{loss2}")
        loss_list.append(loss) # 为了可视化使用的
        loss.backward()  # 反向传播，求梯度
        optimizer.step()  # 进行梯度下降

        if i % 20 == 0:
            print(f"Epoch:{epoch_num} {i} , loss:{loss.item()}")

    # step6.验证结果的准确率
    # 训练完成之后进行验证。
    # ....
    return sum(loss_list)/len(loss_list) # 计算损失的平均值


if __name__ == '__main__':

    EPOCH = 100
    # 判断是否使用GPU
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # 用不到了
    transforms = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量结构
        transforms.Normalize((0.1037,), (0.3081,))  # 对数据进行标准化
    ])

    # step1：获取数据集
    my_dataset = MyDataSet(dataset_path)
    # 将数据导入迭代器DataLoader之中， shuffle表示是否要将数据打乱
    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 只识别10个，且输入通道为1
    # net = Model_index(input_channel=3,input_K=10) step2 加载模型
    net_resnet = resnet50(num_classes=20).to(device) # 去掉分类的resnet
    load_weight(net_resnet, "./resnet50-19c8e357.pth")
    net_gen = Model_geng2(net_resnet, K=3).to(device) # 整个网络结构，K代表输出的参数个数3，因为CV模型是3个参数



    # step3.定义损失函数，梯度下降算法
    # 定义损失函数
    # loss_func = nn.CrossEntropyLoss()
    # loss_func = MyEntropyLoss()
    loss_func_kl = MyKLLos()    #KL损失

    # 定义梯度下降的优化器Adam
    # optimizer = torch.optim.Adam(params=net_gen.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(params=net_gen.parameters(), lr=0.001)
    for params in net_gen.parameters():
        params.requires_grad = True

    # 定义水平集的cv模型
    level_model = CV_chan(num=10,step=0.1)

    #  开始训练  训练100个epoch
    for epoch_num in range(EPOCH):
        # train_loader 加载训练集
        # epoch_num 第一个epoch
        # level_model 是水平集的模型
        # 整个模型的前向传播，计算损失值，梯度下降
        loss = train_one_epoch(train_loader,epoch_num,level_model,device)


        # step7.保存模型权重
        if epoch_num % 10 == 0:
            params_dict = net_gen.state_dict()
            # with open(f"weight/net-{i}-{loss}.pth")
            torch.save(params_dict, f"weight/net-{epoch_num}-{loss}.pth")

