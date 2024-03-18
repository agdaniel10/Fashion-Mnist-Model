'This is an Implementaion of the Googlenet inception architecture from scratch'

import torch
import torch.nn as nn

class GoogleNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000):
        super(GoogleNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        self.conv1 = conv_block(in_channels=3, 
                                out_channels=64, 
                                kernel_size=(7, 7), 
                                padding=3,
                               stride=2)
        
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=(3,3), stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1= InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool3(x)

        x = self.inception4a(x)
        # Auxilary softmax classifier 1
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxilary Softmax classifier 2
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.max_pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x

class inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(inception_block, self).__init__()
        # 1x1 convolution layer
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        # 3x3 convolution branch
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # 3x3 convolution branch
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_pool, kernel_size=1)
        )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)

        # print(branch1_out.size(), branch2_out.size(), branch3_out.size(), branch4_out.size())

        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], 1)
    
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class conv_block(nn.Module):
    def  __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchNorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchNorm(self.conv(x)))
    

if __name__ == '__main__':
    batch_size = 5
    x = torch.randn(batch_size, 3, 224, 224)
    model = GoogleNet(aux_logits=True, num_classes=1000)
    print(model(x)[2].shape)
    assert model(x)[2].shape == torch.Size([batch_size, 1000])

