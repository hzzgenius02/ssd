import torch.nn as nn
import torch


class Bottleneck(nn.Module):
    expansion = 4

    # 定义了多种层，forward定义了调用该网络正向传播时，各层的顺序；此处downsample由调用时给定
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 这些变量名在ssd中都会被弃用
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*4,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    # 残差网络结构（类型2）
    def forward(self, x):
        identity = x
        # 对应虚线
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        # conv1
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # conv2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 64,128,256，512分别是conv2，3，4，5的第一block的第一层的卷积核个数;
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # conv3
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # conv4
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # 修改conv4_block1的步距，从2->1
        self.layer3[0].conv2.stride = (1, 1)
        self.layer3[0].downsample[0].stride = (1, 1)
        # self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        # 加不加上最后的池化和全连接层
        # if self.include_top:
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        # 第一个block，只调整特征矩阵的深度，不改变高度和宽度
        downsample = nn.Sequential(
            nn.Conv2d(self.in_channel, channel * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(channel * block.expansion))

        # channel作为out_channel参数，在一个layer（包括多个block）中保持不变，如64，128...
        # 无论一个block的输入层数多少，输出层数总是不变量out_channel的4倍
        layers = [block(self.in_channel, channel, downsample=downsample, stride=stride)]
        self.in_channel = channel * block.expansion

        # 后面几个block
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        # if self.include_top:
        #     x = self.avgpool(x)
        #     x = torch.flatten(x, 1)
        #     x = self.fc(x)

        return x


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6], num_classes=num_classes, include_top=include_top)

