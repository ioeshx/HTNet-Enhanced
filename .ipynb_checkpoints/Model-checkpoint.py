from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )

class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class HTNet(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2 # 3*7*7
        fmap_size = image_size // patch_size # 28/7 = 4
        blocks = 2 ** (num_hierarchies - 1) # 4

        seq_len = (fmap_size // blocks) ** 2   # =1, sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies))) # 2 1 0
        mults = [2 ** i for i in reversed(hierarchies)] # 1 2 4

        layer_heads = list(map(lambda t: t * heads, mults)) # 3,6,12
        layer_dims = list(map(lambda t: t * dim, mults)) # 256,512,1024
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]  # 256,512,1024,1024
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:]) # ((256,512), (512,1024), (1024,1024))
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),    # rearrange后h和w就是 image_size / patch_size
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))


        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers) # 3

        # add after patch embedding?

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level # 4,2,1
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)
        return self.mlp_head(x)

# This function is to confuse three models
class Fusionmodel(nn.Module):
  def __init__(self):
    #  extend from original
    super(Fusionmodel,self).__init__()
    self.fc1 = nn.Linear(15, 3)
    self.bn1 = nn.BatchNorm1d(3)
    self.d1 = nn.Dropout(p=0.5)
    self.fc_2 = nn.Linear(6, 3)
    self.relu = nn.ReLU()
    # forward layers is to use these layers above
  def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
    fuse_four_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
    fuse_out = self.fc1(fuse_four_features)
    fuse_out = self.relu(fuse_out)
    fuse_out = self.d1(fuse_out) # drop out
    fuse_whole_four_parts = torch.cat(
        (whole_feature,fuse_out), 0)
    fuse_whole_four_parts = self.relu(fuse_whole_four_parts)
    fuse_whole_four_parts = self.d1(fuse_whole_four_parts)
    out = self.fc_2(fuse_whole_four_parts)
    return out


class HTNet_Enhanced(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2 # 3*7*7
        fmap_size = image_size // patch_size # 28/7 = 4
        blocks = 2 ** (num_hierarchies - 1) # 4

        seq_len = (fmap_size // blocks) ** 2   # =1, sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies))) # 2 1 0
        mults = [2 ** i for i in reversed(hierarchies)] # 1 2 4

        layer_heads = list(map(lambda t: t * heads, mults)) # 3,6,12
        layer_dims = list(map(lambda t: t * dim, mults)) # 256,512,1024
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]  # 256,512,1024,1024
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:]) # ((256,512), (512,1024), (1024,1024))
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),    # rearrange后h和w就是 image_size / patch_size
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        #TODO 1 init another branch
        self.globalTransformer = GlobalTransformer_2(
            dim_in=layer_dims[0],   # 256
            dim_out=layer_dims[-1], # 1024
            seq_len=image_size ** 2, # image size should be
            depth=2,
            heads=layer_heads[0], # 3
        )

        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim*2),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim*2, num_classes)
        )
    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers) # 3

        # x_Global_transformer = x.clone()  # 使用 clone() 方法复制张量

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level # 4,2,1
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)

        #TODO 2 fusion, rerrange 堆叠到batch维度上
        x_2 = self.globalTransformer(img)
        # print(x.size())
        # print(x_2.size())
        x_fused = torch.cat((x, x_2), dim=1)

        return self.mlp_head(x_fused)
        
## TODO 3
class GlobalTransformer(nn.Module):
    def __init__(self, dim_in, dim_out, seq_len, depth, heads, mlp_mult=4, dropout=0.):
        super().__init__()
        self.transformer = Transformer(dim_in, seq_len=seq_len, depth=depth, heads=heads, mlp_mult=mlp_mult, dropout=dropout)
        # 这部分换成类似block aggregate的结构
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3)
        self.LN = LayerNorm(dim_out)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=2)
        

    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]

        x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = h, b2 = w)
        x = self.transformer(x)
        x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = h, b2 = w)
        x = self.conv1(x)
        x = self.LN(x)
        x = self.maxpool(x)
        return x
    
class GlobalTransformer_2(nn.Module):
    def __init__(self, dim_in, dim_out, seq_len, depth, heads, mlp_mult=4, dropout=0.):
        super().__init__()
        self.conv0 = nn.Conv2d(3, dim_in, kernel_size=3, padding=1)
        self.transformer = Transformer(dim_in, seq_len=seq_len, depth=depth, heads=heads, mlp_mult=mlp_mult, dropout=dropout)
        self.conv1 = nn.Conv2d(dim_in, dim_in*2, kernel_size=7, stride=7) # output: 4*4 fmap
        self.conv2 = nn.Conv2d(dim_in*2, dim_out, kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        
    # 整个28*28特征图直接输入transformer -> conv2d 7*7 -> conv2d 2*2, 最后avgpool输出 1*1，升维度到1024
    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]
        
        x = self.conv0(x)
        x = self.transformer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        return x

