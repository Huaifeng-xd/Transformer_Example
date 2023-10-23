import torch
import torch.nn as nn
import math
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

    def __init__(self, img_size=192, patch_size=16, in_c=4, embed_dim=1024, norm_layer=None):
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

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Self-attention
        residual = x
        x = self.layer_norm1(x + self.dropout(self.self_attention(x, x, x, src_mask)))

        # Feed-forward network
        x = self.layer_norm2(x + self.dropout(self.feed_forward(x)))

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

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)

        return x

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
