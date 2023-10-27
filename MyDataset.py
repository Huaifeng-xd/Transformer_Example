import math

import OpenEXR
import numpy as np
import torch
from PIL import Image
import os
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.depth_dir = os.path.join(root_dir, 'depth')
        self.img_dir = os.path.join(root_dir, 'img')
        self.amp_dir = os.path.join(root_dir, 'amp')
        self.phs_dir = os.path.join(root_dir, 'phs')

        self.depth_files = os.listdir(self.depth_dir)

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        depth_name = self.depth_files[idx]
        depth_path = os.path.join(self.depth_dir, depth_name)
        img_path = os.path.join(self.img_dir, depth_name.replace('.exr', '.exr'))
        amp_path = os.path.join(self.amp_dir, depth_name.replace('.exr', '.exr'))
        phs_path = os.path.join(self.phs_dir, depth_name.replace('.exr', '.exr'))



        # 读取 img、amp 和 phs 图像
        img = read_exr(img_path)
        # 读取 depth 图像
        depth = read_exr(depth_path, True)
        amp = read_exr(amp_path)
        phs = read_exr(phs_path)
        width = int(math.sqrt(img.shape[1]))
        # 将图像转换为张量并进行归一化处理
        depth = torch.from_numpy(depth.copy()).unsqueeze(0).reshape(1,width,width)
        img = torch.from_numpy(img.copy()).reshape(3,width,width)
        amp = torch.from_numpy(amp.copy()).reshape(3,width,width)
        phs = torch.from_numpy(phs.copy()).reshape(3,width,width)

        # # 将 depth 转换为灰度图
        # depth = depth.mean(dim=2, keepdim=True)

        # RGB-depth 四通道图像
        rgb_depth = torch.cat((img, depth), dim=0)

        # amp-phs 六通道图像
        amp_phs = torch.cat((amp, phs), dim=0)

        return rgb_depth, amp_phs

import OpenEXR
import Imath
import numpy as np
def read_exr(file_path, depth=False):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    # size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channel_names = header['channels'].keys()
    # channel_format = header['channels'].values()
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    image = [np.frombuffer(exr_file.channel(c, pixel_type), dtype=np.float32) for c in channel_names]

    # image = np.stack(channels, axis=1)
    # # image = np.flipud(image)
    if depth:
        image = image[0]
    else:
        image = np.array(image)
    return image
