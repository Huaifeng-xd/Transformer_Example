import torch

from MyDataset import MyDataset
from model_util import *
from matplotlib import pyplot as plt
class Transformer(torch.nn.Module):
    # 由于希望输出的是6通道的，所以尝试embed_dim=16*16*6=1536
    def __init__(self,num_heads=8, num_layers=6,embed_dim=1536,img_size=192, patch_size=16):
        super().__init__()
        self.embed_rgb_depth = EmbedLayer(img_size, patch_size, embed_dim=embed_dim, in_c=4)
        self.embed_amp_phs = EmbedLayer(img_size, patch_size, embed_dim=embed_dim, in_c=6)
        self.encoder = TransformerEncoder(embed_dim,num_heads, num_layers)
        self.decoder = TransformerDecoder(embed_dim, num_heads, num_layers)
        self.res_encoder = ResNetEncoder()
        self.res_decoder = ResNetDecoder()
        self.patch_num = img_size // patch_size
        self.conv = nn.Conv2d(embed_dim, out_channels=6, kernel_size=patch_size, stride=patch_size)


    def forward(self, x, y):
        x = self.embed_rgb_depth(x)
        y = self.embed_amp_phs(y)

        # 编码层计算
        # [b, 50, 32] -> [b, 50, 32]
        encoder_output = self.encoder(x, None)
        mask_attn_rows = self.patch_num ** 2
        mask_attn_columns = mask_attn_rows
        batch_size = x.size(0)
        attn_shape = (batch_size, mask_attn_rows, mask_attn_columns)
        tgt_mask = get_attn_tgt_mask(attn_shape)
        expanded_tensor = tgt_mask.unsqueeze(1)
        tgt_mask = expanded_tensor.repeat(1, 8, 1, 1).cuda()
        # 解码层计算
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.decoder(y, encoder_output, None, tgt_mask)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 39]

        #
        # # 解码器前向传播
        B, PatchNum, PatchEmbeddings = y.shape
        # holo_output = self.res_decoder(encoded)
        hologram_patch = y.reshape(B, PatchNum, 6, int(math.sqrt(PatchEmbeddings // 6)), int(math.sqrt(PatchEmbeddings // 6)))
        hologram = torch.reshape(hologram_patch,(B,6,192,192))
        # 编码器前向传播
        # encoded = self.res_encoder(y)
        # print(encoded.shape)  # 输出维度为 (8, 64, 72, 72)
        # print(hologram)
        # amp = hologram[:, :3, :, :]
        # phase = hologram[:, 3:, :, :]

        # 将每个图片形状改为 (3, 192, 192)
        return hologram

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
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # x = torch.randn(33, 4, 192, 192).to(device)  # 输入维度为 (batch_size, channels, height, width)
    # y = torch.rand(33, 6, 192, 192).to(device)
    # transformer = Transformer().to(device)
    # amp, phase = transformer(x, y)
    # input_tensor = amp.cpu().detach().numpy()
    # # 将张量中的图片提取并展示
    # for i in range(33):
    #     image = input_tensor[i]
    #     plt.imshow(image.transpose(1, 2, 0))
    #     plt.axis('off')
    #     plt.show()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor

    # 设置训练参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 1000
    batch_size = 32
    learning_rate = 0.001

    # 创建数据集
    dataset = MyDataset(root_dir=r'D:\HoloDoc\data_mit_4k\test_192')

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建 Transformer 模型实例
    model = Transformer().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    total_step = len(dataloader)
    for epoch in range(num_epochs):
        for i, (rgb_depth, amp_phs) in enumerate(dataloader):
            # 将数据移动到设备上
            rgb_depth = rgb_depth.to(device)
            amp_phs = amp_phs.to(device)

            # 前向传播
            outputs = model(rgb_depth, amp_phs)

            # 计算损失
            loss = criterion(outputs, amp_phs)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印训练信息
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")

    # 保存模型
    torch.save(model.state_dict(), "model.pth")