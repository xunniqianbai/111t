
import sys


from network2.my_resnet import resnet50
import torch
import torch.nn as nn



class Model_geng2(nn.Module):
    def __init__(self,backbone,K=10):
        super(Model_geng2, self).__init__()
        self.backbone = backbone # resnet50
        self.backbone.include_top = False

        # for param in self.backbone.parameters():
        #     param.requires_grad = False



        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1,1),自适应的特征池化下采样，无论输入的高和宽是多少，输出都是(1,1)
        # self.fc = nn.Linear(2048, K)  # 全连接层
        # self.tanh = nn.Tanh()


        # self.up_cv_1_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_cv_1_1 = nn.ConvTranspose2d(in_channels=512,out_channels=256,stride=1,kernel_size=3,padding=0,dilation=1)

        # in_channels: int,
        # out_channels: int,
        # kernel_size: _size_2_t,
        # stride: _size_2_t = 1,
        # padding: _size_2_t = 0,
        self.up_cv_1_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,stride=2,padding=1)
        self.up_cv_1_1_relu = nn.ReLU()

        self.up_cv_2_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128,  kernel_size=4,stride=2,padding=1)
        self.up_cv_2_1_relu = nn.ReLU()
        self.up_cv_2_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,  kernel_size=4,stride=2,padding=1)
        self.up_cv_2_2_relu = nn.ReLU()
        self.up_cv_2_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,  kernel_size=4,stride=2,padding=1)
        self.up_cv_2_3_relu = nn.ReLU()

        self.up_cv_3_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,  kernel_size=4,stride=2,padding=1)
        self.up_cv_3_1_relu = nn.ReLU() # 隐藏层 ，激活函数用relu
        self.up_cv_3_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64,  kernel_size=4,stride=2,padding=1)
        self.up_cv_3_2_relu = nn.ReLU()
        self.up_cv_3_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64,  kernel_size=4,stride=2,padding=1)
        self.up_cv_3_3_relu = nn.ReLU()

        self.up_4_1 = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False)
        self.up_cv_4_1_tanh = nn.Tanh()
        self.up_4_2 = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False)
        self.up_cv_4_2_tanh = nn.Tanh()
        self.up_4_3 = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False)
        self.up_cv_4_3_tanh = nn.Tanh()


        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1_conv = nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False)
        self.up1_relu = nn.ReLU()

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2_conv = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False)
        self.up2_relu = nn.ReLU()

        self.origion_corner = nn.Conv2d(512,1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.up_operate = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        # 定义下面的网络结构
        '''
        self.conv2_1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(1024)
        self.relu2_1 = nn.ReLU()
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1,1),自适应的特征池化下采样，无论输入的高和宽是多少，输出都是(1,1)

        self.fc2_2 = nn.Linear(1024, 512)  # 全连接层
        self.relu2_3 = nn.ReLU()

        self.fc2_3 = nn.Linear(512, 2)  # 全连接层
        self.tanh2_3 = nn.Tanh()
        '''

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

    def forward(self,x):
        # self.backbone(x) resnet50
        feature_map,x_layer1, x_layer2, x_layer3, x_layer4 = self.backbone(x)
        print("")
        # x = self.avgpool(feature_map)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        # # tanh的值域
        # x = self.tanh(x) * 200

        # 预测初始轮廓
        # feature_map [1,2048,8,8]
        x = self.up1(feature_map)  #[1,2048,16,16]
        x = self.up1_conv(x) # [1,1024,16,16]
        x = self.up1_relu(x)  # [1,1024,16,16]
        x = x + x_layer3


        x = self.up2(x)
        x = self.up2_conv(x)
        x = self.up2_relu(x)
        x = x + x_layer2    # [1,512,32,32]

        # ========================================================
        x1 = self.up_cv_1_1(x)
        # length
        x1_1 = self.up_cv_2_1(x1)
        x1_1 = self.up_cv_2_1_relu(x1_1)
        x1_1 = self.up_cv_3_1(x1_1)
        x1_1 = self.up_cv_3_1_relu(x1_1)
        x1_1 = self.up_4_1(x1_1)
        # x1_1 = self.up_cv_4_1_relu(x1_1)
        x1_1 = self.up_cv_4_1_tanh(x1_1) * 5 # [-5,5]

        # Penalty
        x1_2 = self.up_cv_2_2(x1)
        x1_2 = self.up_cv_2_2_relu(x1_2)
        x1_2 = self.up_cv_3_2(x1_2)
        x1_2 = self.up_cv_3_2_relu(x1_2)
        x1_2 = self.up_4_2(x1_2)
        # x1_2 = self.up_cv_4_2_relu(x1_2)
        x1_2 = self.up_cv_4_2_tanh(x1_2) * 5  # [-5,5]

        # CVTerm
        x1_3 = self.up_cv_2_3(x1)
        x1_3 = self.up_cv_2_3_relu(x1_3)
        x1_3 = self.up_cv_3_3(x1_3)
        x1_3 = self.up_cv_3_3_relu(x1_3)
        x1_3 = self.up_4_3(x1_3)
        # x1_3 = self.up_cv_4_3_relu(x1_3)
        x1_3 = self.up_cv_4_3_tanh(x1_3) * 5  # [-5,5]



        # =============================================================

        x2 = self.origion_corner(x)
        x2 = self.sigmoid(x2) # 把数值映射到[0,1]之间



        # # 上面的结构
        # x = self.conv1_1(feature_map)
        # x = self.bn1_1(x)
        # x = self.relu1_1(x)
        #
        # x = self.conv1_2(x)
        # x = self.bn1_2(x)
        # x = self.relu1_2(x)
        #
        # x = self.avgpool1(x)
        # x = torch.flatten(x, 1)
        #
        # x = self.fc1_3(x)
        # x = self.relu1_3(x)
        #
        # x = self.fc1_4(x)
        # x = self.tanh1_4(x)


        # 下面的结构
        """
        y = self.conv2_1(feature_map)
        y = self.bn2_1(y)
        y = self.relu2_1(y)

        y = self.avgpool2(y)
        y = torch.flatten(y, 1)

        y = self.fc2_2(y)
        y = self.relu2_3(y)

        y = self.fc2_3(y)
        y = self.tanh2_3(y)
        return x,y
        """
        # 将图片上采样4倍
        # 双线行插值，要么是反卷积
        x3 = self.up_operate(x2) # 64
        x3 = self.up_operate(x3) # 128
        x3 = self.up_operate(x3) # 256
        # 二值化
        x3[x3 >= 0.5] = 1
        x3[x3 <  0.5] = -1
        # 判断是否为空
        if x3.sum() < -1* (x3.size()[2] + x3.size()[3]) + 10:
        # if x3.sum() < 20: # 如果没有初始轮廓，给一个空的初始轮廓
            x3[:,:,10:20,10:20] = 1
        # x3是x2上采样的结果
        return x1_1,x1_2,x1_3,x2,x3


if __name__ == '__main__':
    net = resnet50(num_classes=20)

    input = torch.randn((1,1,256,256))

    net = Model_geng2(net)

    output = net(input)
    print(output[0].size(),output[1].size(),output[2].size(),output[3].size(),output[4].size())





