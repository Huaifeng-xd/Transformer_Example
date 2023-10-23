import torch
from model_util import *
class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_rgb_depth = EmbedLayer()
        self.embed_amp_phs = EmbedLayer()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.fc_out = torch.nn.Linear(32, 39)

    def forward(self, x, y):
        x = self.embed_rgb_depth(x)
        y = self.embed_amp_phs(y)

        # 编码层计算
        # [b, 50, 32] -> [b, 50, 32]
        x = self.encoder(x, mask_pad_x)

        # 解码层计算
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 39]
        y = self.fc_out(y)

        return y