import math
import torch
import torch.nn as nn
import numpy as np


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # Linear projections
        query = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        x = torch.matmul(self.dropout(attention_weights), value)

        # Concatenate heads and linear projection
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        x = self.fc(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).to(query.device))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(self.dropout(attention_weights), value)

        # Concatenate and linear projection
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                            self.num_heads * self.head_dim)
        x = self.fc(attended_values)

        return x


class FullyConnectedOutput(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(p=0.1),
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=32,
                                       elementwise_affine=True)

    def forward(self, x):
        # 保留下原始的x,后面要做短接用
        clone_x = x.clone()

        # 规范化
        x = self.norm(x)

        # 线性全连接运算
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(x)

        # 做短接
        out = clone_x + out

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        position_encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        position_encoding = position_encoding.unsqueeze(0)

        self.register_buffer('position_encoding', position_encoding)

    def forward(self, x):
        x = x + self.position_encoding[:, :x.size(1)]
        return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.conv = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        # 这里的HW就是经过卷积后留下的patchsize，c就是embedding dim
        x = self.conv(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# 位置编码层
class EmbedLayer(nn.Module):
    def __init__(self, img_size=192, patch_size=16, in_c=4, embed_dim=1024, norm_layer=None):
        super().__init__()
        patch_num = img_size // patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_c, embed_dim)
        self.position_embed = PositionalEncoding(patch_num, self.patch_embed)

    def forward(self, x):
        pae = self.patch_embed(x)
        embed_out = self.position_embed(pae)
        return embed_out


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class HoloGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = FeedForwardLayer(1024, 2048, 4096)
        self.mlp2 = FeedForwardLayer(4096, 2048, 1024)

    def forward(self, x):
        # B, N, dim = x.shape
        x = self.mlp(x)
        res = self.mlp(x)
        return res


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, enc_outputs, src_mask, tgt_mask):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_outputs, src_mask, tgt_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardLayer(d_model)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_outputs, src_mask, tgt_mask):
        # Self-attention
        residual = x
        x = self.layer_norm1(x + self.dropout(self.self_attention(x, x, x, tgt_mask)))

        # Encoder-decoder attention
        x = self.layer_norm2(x + self.dropout(self.enc_dec_attention(x, enc_outputs, enc_outputs, src_mask)))

        # Feed-forward network
        x = self.layer_norm3(x + self.dropout(self.feed_forward(x)))

        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        origin = x.clone()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = origin + self.norm(x)
        return x


def plot_hologram(images):
    import numpy as np
    import matplotlib.pyplot as plt

    # # 假设你有一个形状为 (32, 6, 192, 192) 的数组
    # images = np.random.random((32, 6, 192, 192))

    # 定义子图的行数和列数
    num_rows = 8
    num_cols = 4

    # 创建一个新的图像窗口，并设置子图的布局
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 24))

    # 遍历每个子图位置，并在每个位置显示一个图像
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j  # 计算图像在数组中的索引
            image = images[index]  # 获取当前索引对应的图像

            # 由于图像是三通道的，需要将其转换为正确的图像格式
            image = np.transpose(image, (1, 2, 0))

            # 显示图像在当前子图位置
            axes[i, j].imshow(image)
            axes[i, j].axis('off')  # 禁用坐标轴

    # 调整子图之间的间距
    plt.tight_layout()

    # 展示图像
    plt.show()
