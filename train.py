import torch
from model_util import *
class Transformer(torch.nn.Module):
    def __init__(self,num_heads=8, num_layers=6,embed_dim=1024,img_size=192, patch_size=16):
        super().__init__()
        self.embed_rgb_depth = EmbedLayer(img_size, patch_size, embed_dim=embed_dim, in_c=4)
        self.embed_amp_phs = EmbedLayer(img_size, patch_size, embed_dim=embed_dim, in_c=6)
        self.encoder = TransformerEncoder(embed_dim,num_heads, num_layers)
        self.decoder = TransformerDecoder(embed_dim, num_heads, num_layers)
        self.transformer_out = ResNetEncoder()
        self.amp_phase_out = ResNetDecoder()

    def forward(self, x, y):
        x = self.embed_rgb_depth(x)
        y = self.embed_amp_phs(y)

        # 编码层计算
        # [b, 50, 32] -> [b, 50, 32]
        x = self.encoder(x)

        # 解码层计算
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.decoder(y, x, x)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 39]
        # 编码器前向传播
        encoded = self.transformer_out(y)
        print(encoded.shape)  # 输出维度为 (8, 64, 72, 72)

        # 解码器前向传播
        decoded = self.amp_phase_out(encoded)
        print(decoded.shape)

        return y

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ResNetDecoder(nn.Module):
    def __init__(self):
        super(ResNetDecoder, self).__init__()
        self.conv1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=(192, 192), mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.upsample(x)
        x = self.conv2(x)
        return x


if __name__ == '__main__':
 #创建输入数据
x = torch.randn(8, 1024, 144, 144)  # 输入维度为 (batch_size, channels, height, width)

