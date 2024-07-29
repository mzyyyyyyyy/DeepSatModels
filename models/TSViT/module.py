import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, gt, **kwargs):
        return self.fn(self.norm(x), gt, **kwargs)


class PreNormLocal(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        # print('before fn: ', x.shape)
        x = self.fn(x, **kwargs)
        # print('after fn: ', x.shape)
        return x


class Conv1x1Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, gt):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, gt):
        # print(x.shape)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # print(q.shape, k.shape, v.shape)
        # shape = (3456, 4, 79, 32)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # shape = (3456, 4, 79, 79)
        # 1, einsum 这个函数处理矩阵乘法也太方便了，
        # 直接把输入输出的维度写出来即可定义矩阵乘法了。
        # 2, self.scale 通常是一个预定义的缩放因子，通常设置为 1 / sqrt(head_dim)。
        # 这个缩放因子是为了防止在计算点积注意力分数时值过大，导致梯度消失或梯度爆炸。
        # 3，q 和 k 的数量可能是不同的嘛？这里为啥一个是 i, 一个是 j 呢？

        attn = dots.softmax(dim=-1)
        # shape = (3456, 4, 79, 79)

        # 选择特定类别的标签值
        indices = torch.where(gt == 19)[0]
        attn_cls = attn[indices]




        self.vis_attn(attn)


        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # shape = (3456, 4, 79, 32)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # shape = (3456, 79, 128)
        out = self.to_out(out)
        # 这是一个非常标准的 attention 模块。
        return out
    
    def vis_attn(self, attn):
        # 将 19 个子图取平均，只得到 4 个 head 的注意力图
        vis = torch.mean(torch.mean(attn, dim=0)[:, -19:, :60], dim=1)
        for i in range(4):
            plt.figure(figsize=(10, 5))
            plt.plot(vis[i])
            plt.title(f"Average Line Plot {i+1}")
            plt.xlim([0, 59])
            plt.xlabel("Index")
            plt.ylabel("Average Value")
            plt.show()

        # 绘制 19 个 cls token 的注意力
        # vis = torch.mean(attn, dim=0)[:, -19:, :60]
        # for i in range(4):
        #     fig, axes = plt.subplots(5, 4, figsize=(20, 20))  # 创建 5 行 4 列的子图网格
        #     axes = axes.flatten()  # 将 2D 数组展平成 1D 数组

        #     for j in range(19):
        #         ax = axes[j]
        #         ax.plot(vis[i, j])
        #         # ax.set_title(f"Line Plot {i+1}-{j+1}")
        #         ax.set_xlim([0, 59])

        #     # 删除多余的子图
        #     for j in range(19, 20):
        #         fig.delaxes(axes[j])

        #     plt.tight_layout()
        #     plt.show()



class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class LeFF(nn.Module):

    def __init__(self, dim=192, scale=4, depth_kernel=3):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(scale_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                     )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=14, w=14)
            )

        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                       Rearrange('b n c -> b c n'),
                                       nn.BatchNorm1d(dim),
                                       nn.GELU(),
                                       Rearrange('b c n -> b n c')
                                       )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x


class LCAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q[:, :, -1, :].unsqueeze(2)  # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

