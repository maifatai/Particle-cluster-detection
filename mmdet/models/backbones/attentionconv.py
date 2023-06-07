import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#添加新模块要加
from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'# 可能是由于是MacOS系统的原因

def auto_pad(kernal_size):
    return kernal_size//2


class Attention(nn.Module):
    '''
    Scaled Dot-product Attention
    '''
    def __init__(self,dim,num_heads,qkv_bias=False, qk_scale=None,attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.dim=dim
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale=qk_scale or head_dim**-0.5#scaled dot-product

        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        '''
        :param x: [B, N, C] N=W*H
        :return:
        '''
        B,N,C=x.shape
        # [B, N, C]->[B, N, 3C]->[B, N, 3,num_head,C/num_heads]->[3,B,num_head,N,C/num_heads]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]#[B,num_head,N,C/num_heads]

        q=q*self.scale
        attn=q@k.transpose(-2,-1)
        attn=self.attn_drop(self.softmax(attn))
        x=(attn@v).transpose(1,2).reshape(B,N,C)
        x=self.proj_drop(self.proj(x))
        return x

class Conv(nn.Module):
    def __init__(self,in_channels,kernel_size=7,norm_layer=nn.LayerNorm,act_layer=nn.GELU,conv_scale=4,drop_path=0.):
        super(Conv, self).__init__()
        self.dwconv=nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,padding=auto_pad(kernel_size),groups=in_channels)#depthwise conv
        self.norm=norm_layer(in_channels)
        self.pwconv=nn.Conv2d(in_channels,int(conv_scale*in_channels),kernel_size=1)
        self.act=act_layer()
        self.pwconv1=nn.Conv2d(int(conv_scale*in_channels),in_channels,kernel_size=1)
        self.drop_path=DropPath(drop_path) if drop_path>0. else nn.Identity()

    def forward(self,x):
        """
        :param x: [B,C,H,W]
        :return:
        """
        input=x
        x=self.dwconv(x)
        x=x.permute(0,2,3,1).contiguous()#[B,C,H,W]->[B,H,W,C]
        x=self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()#[B,H,W,C]->[B,C,H,W]
        x=self.pwconv(x)
        x=self.act(x)
        x=self.pwconv1(x)
        x=input+self.drop_path(x)#[B,C,H,W]
        return x

class PatchEmbed(nn.Module):
    def __init__(self,img_size=224,patch_size=4,in_channels=3,embed_dim=96):
        super(PatchEmbed, self).__init__()
        img_size=to_2tuple(img_size)
        patch_size=to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim=embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        # self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,stride=patch_size)  # [bn,embed_dim,patch_res,patch_res]
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self,x):
        B, C, H, W = x.shape#[2,3,512,512]
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x=x.float().to(x.device)
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B,C,Ph,Pw]->[B,C,Ph*Pw]->[B Ph*Pw C]
        x=self.norm(x)
        x=x.permute(0,2,1).reshape(B,self.embed_dim,H//self.patch_size[0],W//self.patch_size[1])#[B,C,Ph,Pw]
        return x

class Block(nn.Module):
    def __init__(self,dim,num_heads,qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,kernel_size=7,norm_layer=nn.LayerNorm,act_layer=nn.GELU,conv_scale=4, drop_path=0.,
                 ):
        super(Block, self).__init__()

        self.norm=norm_layer(dim)
        self.attn=Attention(dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=proj_drop)
        self.conv=Conv(in_channels=dim,kernel_size=kernel_size,norm_layer=norm_layer,act_layer=act_layer,conv_scale=conv_scale,drop_path=drop_path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # if downsample is None:
        #     self.downsample=PatchEmbed(img_size,patch_size=patch_size,in_channels=in_channels,embed_dim=dim)
        # else:
        #     self.downsample=Downsample(in_channels=dim,down_scale=down_scale)
    def forward(self,x):
        '''
        :param x: [B,C,H,W]
        :return:
        '''
        # x=self.downsample(x)
        B,C,H,W=x.shape
        x=x.flatten(2).permute(0,2,1).contiguous()#[B, N, C]
        x=x+self.drop_path(self.attn(self.norm(x)))
        x=x.permute(0,2,1).reshape(B,C,H,W)#[B, N, C]->[B,C,H,W]
        x=x+self.drop_path(self.conv(x))
        # print(x.shape)
        return x
 
class Downsample(nn.Module):
    def __init__(self,in_channels,kernel_size=3,stride=2,down_scale=2):
        super(Downsample, self).__init__()
        self.conv=nn.Conv2d(in_channels,in_channels*down_scale,kernel_size=kernel_size,stride=stride,padding=auto_pad(kernel_size))
        self.act=nn.GELU()
        
    def forward(self,x):
        return self.act(self.conv(x))

@BACKBONES.register_module()#增加新的模块需要 
class AttentionConv(nn.Module):
    def __init__(self,img_size=224,patch_size=4,in_channels=3,dim=96,
                 num_heads=[3,6,12,24],depths=[2,2,6,2],out_indices=(0, 1, 2, 3),
                 qkv_bias=False, qk_scale=None,attn_drop=0., proj_drop=0.,
                 kernel_size=7,norm_layer=nn.LayerNorm,act_layer=nn.GELU,conv_scale=4, drop_path=0.1,down_scale=2):
        super(AttentionConv, self).__init__()
        # self.num_class=num_classes
        self.num_stage=len(depths)
        self.embed_dim=dim
        self.num_feature=[int(dim*down_scale**i) for i in range(self.num_stage)]

        self.patch_embed=PatchEmbed(img_size=img_size,patch_size=patch_size,in_channels=in_channels,embed_dim=self.embed_dim)
        # self.down_sample=Downsample(in_channels=down_in_channel,down_scale=down_scale)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]  # stochastic depth decay rule

        self.layers=nn.ModuleList()
        for stage in range(self.num_stage):
            if stage==0:
                self.layers.append(self.patch_embed)
            else:
                self.layers.append(Downsample(in_channels=self.num_feature[stage-1],down_scale=down_scale))
            for depth in range(depths[stage]):
                layer=Block(dim=self.num_feature[stage],
                            num_heads=num_heads[stage],
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            kernel_size=kernel_size,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            conv_scale=conv_scale,
                            drop_path=dpr[sum(depths[:stage])+depth],
                            )
                self.layers.append(layer)
        self.depths=[sum(depths[:i+1]) for i in range(len(depths))]
        self.out_indices=[self.depths[i] for i in out_indices]
        
        # self.apply(self._init_weights)


    # def _init_weights(self,m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
    #         if isinstance(m, nn.Conv2d) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)


    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
            
    def forward(self,x):
        outs=[]
        i=0
        for layer in self.layers:
            x=layer(x).contiguous()
            if isinstance(layer, Block):
                i+=1
                if i in self.out_indices:
                    outs.append(x)
        return tuple(outs)

# if __name__ == '__main__':
#     x=torch.rand(10,3,64,64)
#     model=AttentionConv(img_size=64)
#     out=model(x)
#     # print(model)
#     print(out.shape)




