import math
import torch
import torch.nn as nn

from model import DropPath
import numpy as np


# 注意力计算函数
# def attention(Q, K, V, mask):
#     # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
#     # Q,K,V = [b, 4, 50, 8]
#
#     # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
#     # Q,K矩阵相乘,求每个词相对其他所有词的注意力
#     score = torch.matmul(Q, K.permute(0, 1, 3, 2))
#
#     # 除以每个头维数的平方根,做数值缩放
#     score /= 8 ** 0.5
#
#     # mask遮盖,mask是true的地方都被替换成-inf,这样在计算softmax的时候,-inf会被压缩到0
#     # mask = [b, 1, 50, 50] masked_filled_是库里的函数
#     score = score.masked_fill_(mask, -float('inf'))
#     score = torch.softmax(score, dim=-1)
#
#     # 以注意力分数乘以V,得到最终的注意力结果
#     # [b, 4, 50, 50] * [b, 4, 50, 8] -> [b, 4, 50, 8]
#     score = torch.matmul(score, V)
#
#     # 每个头计算的结果合一
#     # [b, 4, 50, 8] -> [b, 50, 32]
#     score = score.permute(0, 2, 1, 3).reshape(-1, 50, 32)
#
#     return score

# class Attention(nn.Module):
#     def __init__(self,
#                  dim,  # 输入token的dim
#                  num_heads=8,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  attn_drop_ratio=0.,
#                  proj_drop_ratio=0.):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop_ratio)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop_ratio)
#
#     def generate_mask(self, dim):
#         # 此处是 sequence mask ，防止 decoder窥视后面时间步的信息。
#         # padding mask 在数据输入模型之前完成。
#         matirx = np.ones((dim, dim))
#         mask = torch.Tensor(np.tril(matirx))
#
#         return mask == 1
#
#     def forward(self, x, require_mask=False):
#         # [batch_size, num_patches + 1, total_embed_dim]
#         B, N, C = x.shape
#
#         # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
#         # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
#         # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         if require_mask:
#             mask = self.generate_mask(N)
#             # masked_fill 函数中，对Mask位置为True的部分进行Mask
#             attn.masked_fill(mask, value=float("-inf"))  # 注意这里的小Trick，不需要将Q,K,V 分别MASK,只MASKSoftmax之前的结果就好了
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
#         # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


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


# 多头注意力计算层
# class MultiHead(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc_Q = torch.nn.Linear(32, 32)
#         self.fc_K = torch.nn.Linear(32, 32)
#         self.fc_V = torch.nn.Linear(32, 32)
#
#         self.out_fc = torch.nn.Linear(32, 32)
#
#         # 规范化之后,均值是0,标准差是1
#         # BN是取不同样本做归一化
#         # LN是取不同通道做归一化
#         # affine=True,elementwise_affine=True,指定规范化后,再计算一个线性映射
#         # norm = torch.nn.BatchNorm1d(num_features=4, affine=True)
#         # print(norm(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)))
#         """
#         [[[-1.1761, -1.0523, -0.9285, -0.8047],
#          [-1.1761, -1.0523, -0.9285, -0.8047],
#          [-1.1761, -1.0523, -0.9285, -0.8047],
#          [-1.1761, -1.0523, -0.9285, -0.8047]],
#
#         [[ 0.8047,  0.9285,  1.0523,  1.1761],
#          [ 0.8047,  0.9285,  1.0523,  1.1761],
#          [ 0.8047,  0.9285,  1.0523,  1.1761],
#          [ 0.8047,  0.9285,  1.0523,  1.1761]]]"""
#
#         # norm = torch.nn.LayerNorm(normalized_shape=4, elementwise_affine=True)
#         # print(norm(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)))
#         """
#         [[[-1.3416, -0.4472,  0.4472,  1.3416],
#          [-1.3416, -0.4472,  0.4472,  1.3416],
#          [-1.3416, -0.4472,  0.4472,  1.3416],
#          [-1.3416, -0.4472,  0.4472,  1.3416]],
#
#         [[-1.3416, -0.4472,  0.4472,  1.3416],
#          [-1.3416, -0.4472,  0.4472,  1.3416],
#          [-1.3416, -0.4472,  0.4472,  1.3416],
#          [-1.3416, -0.4472,  0.4472,  1.3416]]]"""
#
#         self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)
#
#         self.dropout = torch.nn.Dropout(p=0.1)
#
#     def forward(self, Q, K, V, mask):
#         # b句话,每句话50个词,每个词编码成32维向量
#         # Q,K,V = [b, 50, 32]
#         b = Q.shape[0]
#
#         # 保留下原始的Q,后面要做短接用
#         clone_Q = Q.clone()
#
#         # 规范化
#         Q = self.norm(Q)
#         K = self.norm(K)
#         V = self.norm(V)
#
#         # 线性运算,维度不变
#         # [b, 50, 32] -> [b, 50, 32]
#         K = self.fc_K(K)
#         V = self.fc_V(V)
#         Q = self.fc_Q(Q)
#
#         # 拆分成多个头
#         # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
#         # [b, 50, 32] -> [b, 4, 50, 8]
#         Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
#         K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
#         V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
#
#         # 计算注意力
#         # [b, 4, 50, 8] -> [b, 50, 32]
#         score = attention(Q, K, V, mask)
#
#         # 计算输出,维度不变
#         # [b, 50, 32] -> [b, 50, 32]
#         score = self.dropout(self.out_fc(score))
#
#         # 短接
#         score = clone_Q + score
#         return score


# 全连接输出层
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


# class Mlp(nn.Module):
#     """
#     MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#         self.norm = nn.LayerNorm(out_features)
#
#     def forward(self, x):
#         origin = x
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         x = origin + self.norm(x)
#         return x

# class EncoderBlock(nn.Module):
#     """
#     可直接调用无需添加其余模块的encoder block
#     """
#
#     def __init__(self,
#                  dim,
#                  num_heads,
#                  mlp_ratio=4.,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop_ratio=0.,
#                  attn_drop_ratio=0.,
#                  drop_path_ratio=0.,
#                  act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm):
#         super(EncoderBlock, self).__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                               attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = FeedForwardLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
#                                     drop=drop_ratio)
#
#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


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


# class DecoderLayer:
#     def __init__(self, in_dim=1024):
#         super().__init__()
#
#         self.mh1 = Attention(in_dim)
#         self.mh2 = Attention(in_dim)
#
#         self.fc = FeedForwardLayer()
#
#     def forward(self, x, y, require_mask=False):
#         def generate_mask(self, dim):
#             # 此处是 sequence mask ，防止 decoder窥视后面时间步的信息。
#             # padding mask 在数据输入模型之前完成。
#             matirx = np.ones((dim, dim))
#             mask = torch.Tensor(np.tril(matirx))
#
#         # 先计算y的自注意力,维度不变
#         # [b, 50, 32] -> [b, 50, 32]
#         y = self.mh1(y, y, y, require_mask=True)
#
#         # 结合x和y的注意力计算,维度不变
#         # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
#         y = self.mh2(y, x, x, require_mask=True)
#
#         # 全连接输出,维度不变
#         # [b, 50, 32] -> [b, 50, 32]
#         y = self.fc(y)
#
#         return y



# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super(DecoderLayer, self).__init__()
#
#         self.self_attention = Attention(d_model, num_heads)
#         self.enc_dec_attention = Attention(d_model, num_heads)
#         self.feed_forward = FeedForwardLayer(d_model)
#
#         self.layer_norm1 = nn.LayerNorm(d_model)
#         self.layer_norm2 = nn.LayerNorm(d_model)
#         self.layer_norm3 = nn.LayerNorm(d_model)
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, enc_outputs):
#         # Self-attention
#         residual = x
#         x = self.layer_norm1(x + self.dropout(self.self_attention(x, True)))
#
#         # Encoder-decoder attention
#         x = self.layer_norm2(x + self.dropout(self.enc_dec_attention(x, f)))
#
#         # Feed-forward network
#         x = self.layer_norm3(x + self.dropout(self.feed_forward(x)))
#
#         return x

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
