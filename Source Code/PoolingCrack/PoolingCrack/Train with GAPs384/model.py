import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.ops as OPS
from torch.utils.checkpoint import checkpoint


GROUP_NORM_CHANNELS_PER_GROUP = 16
MLP_SIZE_MULTIPLIER = 4
CHANNEL_REDUCTION_RATIO = 2 #16,8,4
    
class ConvPatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super(ConvPatchEmbedding, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 
                      in_channels, 
                      kernel_size=kernel_size, 
                      groups=in_channels,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)      
        )
    def forward(self, x):
        #print(f'X before patch embedding: {x.size()}')
        return self.conv(x)
    


class PoolingBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, num_blocks, padding=1):
        super(PoolingBlock, self).__init__()
        
        num_groups = in_channels // GROUP_NORM_CHANNELS_PER_GROUP

        self.first_block = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.AvgPool2d(kernel_size, stride=1, padding=padding) #then sum residual
        )

        self.second_block = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.Conv2d(in_channels, in_channels * MLP_SIZE_MULTIPLIER, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels * MLP_SIZE_MULTIPLIER, in_channels, kernel_size=1)
        )

        self.num_blocks = num_blocks

    def forward(self, x):
        #print(f'Number of pooling blocks: {self.num_blocks}')
        for i in range(self.num_blocks):
            residual = x
            x = self.first_block(x)
            #residual = nn.functional.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + residual
            #print(x.size())
            x = self.second_block(x) + x
        return x

class FSM(nn.Module):
    def __init__(self, channels):
        super(FSM, self).__init__()

        self.fr = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.fs = nn.Conv2d(channels, channels // CHANNEL_REDUCTION_RATIO, kernel_size=1)

    def forward(self, x):
        importance_vector = self.fr(x)
        mult = importance_vector * x
        reconstructed_feature_map = mult + x
        return self.fs(reconstructed_feature_map)
    
# pytorch linear.py Linear class has reset_parameters, as nn.Parameters does not get initialized by torch
# code based on medium.com Deformable Convolutional Operation article
class FAM(nn.Module):
    def __init__(self, x_channels, u_channels, kernel_size, groups=1, bias=True):
        super(FAM, self).__init__()

        self.in_channels = u_channels + x_channels
        self.out_channels = 2 * (kernel_size ** 2)
        self.weight = nn.Parameter(
            torch.Tensor(u_channels,
                         u_channels // groups, 
                         kernel_size, 
                         kernel_size)
                        )
        self.padding = kernel_size // 2

        if bias:
            self.bias = nn.Parameter(torch.Tensor(u_channels))
        else:
            self.register_parameter('bias', None)
        
        self.offset_conv = nn.Conv2d(
            in_channels= self.in_channels,
            out_channels= self.out_channels,
            kernel_size=kernel_size,
            padding= self.padding
        )

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


    def forward(self, x, u):
        temp_u = u
        if x.shape[2:] != u.shape[2:]:
            temp_u = TF.resize(u, size=x.shape[2:])
        
        #print(f'U size: {u.size()}')
        #print(f'X size: {x.size()}')

        concat = torch.cat((x, temp_u), dim=1)
        #(f'X: {x.size()}')
        #print(f'U: {u.size()}')
        #print(f'CONCAT: {concat.size()}')

        offset = self.offset_conv(concat)

        return OPS.deform_conv2d(u, offset, self.weight, self.bias, padding = self.padding)
        

class PoolingCrack(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, 
        features=[64, 128, 320], num_pooling_blocks=[6, 6, 18], 
        bottleneck_feature=512, bottleneck_pool_block=6
    ):
        super(PoolingCrack, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        
        self.FSMs = nn.ModuleList()
        for feature in features:
            self.FSMs.append(FSM(feature))

        self.FAMs = nn.ModuleList()
        for feature in features:
            self.FAMs.append(FAM(feature//CHANNEL_REDUCTION_RATIO, feature//CHANNEL_REDUCTION_RATIO, 3))

        # Down
        for feature, pooling_blocks in zip(features, num_pooling_blocks):
            if feature == features[0] and pooling_blocks == num_pooling_blocks[0]:
                self.downs.append(ConvPatchEmbedding(in_channels, feature, kernel_size=5, stride=4))
            else:
                self.downs.append(ConvPatchEmbedding(in_channels, feature, kernel_size=3, stride=2))
            in_channels = feature
            self.downs.append(PoolingBlock(in_channels, 3, pooling_blocks))

        # Bottleneck
        self.bottleneck.append(ConvPatchEmbedding(in_channels, bottleneck_feature, kernel_size=3, stride=2))
        in_channels = bottleneck_feature
        self.bottleneck.append(PoolingBlock(in_channels, 3, bottleneck_pool_block))
        self.bottleneck.append(nn.Conv2d(in_channels, in_channels //CHANNEL_REDUCTION_RATIO, kernel_size=1))
        self.bottleneck.append(nn.ConvTranspose2d(in_channels //CHANNEL_REDUCTION_RATIO, features[-1] //CHANNEL_REDUCTION_RATIO, kernel_size=2, stride=2))
        in_channels = features[-1]//CHANNEL_REDUCTION_RATIO
        
        self.ups.append(nn.ConvTranspose2d(in_channels, features[-2]//CHANNEL_REDUCTION_RATIO, kernel_size=2, stride=2))
        self.ups.append(nn.ConvTranspose2d(features[-2]//CHANNEL_REDUCTION_RATIO, features[-3]//CHANNEL_REDUCTION_RATIO, kernel_size=2, stride=2))
        self.ups.append(nn.ConvTranspose2d(features[-3]//CHANNEL_REDUCTION_RATIO, out_channels, kernel_size=4, stride=4))


    def forward(self, x):
        fsms = []

        # DOWN
        for i in range(0, len(self.downs), 2):
            #print(f'DOWN i-{i//2} : {x}')
            x = self.downs[i](x)
            #print(f'DOWN i-{i//2} after conv patch emb: {x}')

            #x = checkpoint(self.downs[i+1], x) #PoolingBlock, checkpoint is to avoid torch.OutOfMemory
            x = self.downs[i+1](x)
            fsms.append(self.FSMs[i//2](x))
            #print(f"FSM dimensions: {fsms[i//2].size()}")
        


        #print(f'Before bottleneck: {x}')
        # BOTTLENECK
        for i in range(4):
            #print(i)
            x = self.bottleneck[i](x)

        #print(f'Bottleneck MAX: {x.max()}')
        #print(f'Bottleneck MIN: {x.min()}')

        #print(f'After Bottleneck: {x}')
        # UP
        for idx in range(0, len(self.ups)):
            fsm = fsms[-(idx +1)]
            fam = self.FAMs[-(idx +1)]

            #print(f'FSM {idx} MAX: {torch.max(fsm)}')
            #print(f'FSM {idx} MIN: {torch.min(fsm)}')

            #print(f'FAM in loop: {(fam(fsm, x).size())}')
            #print(f'FSM in loop:{fsm.size()}')
            #print(f'FSM {idx}: {fsm}')
            x = fsm + fam(fsm, x)
            #print(f'UP i MAX-{idx}: {x.max()}')
            #print(f'UP i MIN-{idx}: {x.min()}')
            #print(f'UP i-{idx}: {x}')

            x = self.ups[idx](x)
            #print(f'UP i-{idx} after upsampling: {x}')
            #print(f'UP i-{idx} after upsampling MAX: {x.max()}')
            #print(f'UP i-{idx} after upsampling MIN: {x.min()}')


        #print(f'Final X: {x}')
        return x
