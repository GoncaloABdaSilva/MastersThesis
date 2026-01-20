import torch.nn as nn


GROUP_NORM_CHANNELS_PER_GROUP = 16
MLP_SIZE_MULTIPLIER = 4
CHANNEL_REDUCTION_RATIO = 16
    
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
        for i in range(self.num_blocks):
            residual = x
            x = self.first_block(x)
            x = x + residual
            x = self.second_block(x) + x
        return x


class PoolingCrackEncoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, 
        features=[64, 128, 320], num_pooling_blocks=[6, 6, 18], 
        bottleneck_feature=512, bottleneck_pool_block=6
    ):
        super(PoolingCrackEncoder, self).__init__()
        self.downs = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(bottleneck_feature, out_channels)
        
        # Down
        for feature, pooling_blocks in zip(features, num_pooling_blocks):
            if feature == features[0] and pooling_blocks == num_pooling_blocks[0]:
                self.downs.append(ConvPatchEmbedding(in_channels, feature, kernel_size=5, stride=4))
            else:
                self.downs.append(ConvPatchEmbedding(in_channels, feature, kernel_size=3, stride=2))
            in_channels = feature
            self.downs.append(PoolingBlock(in_channels, 3, pooling_blocks))

        self.bottleneck.append(ConvPatchEmbedding(in_channels, bottleneck_feature, kernel_size=3, stride=2))
        in_channels = bottleneck_feature
        self.bottleneck.append(PoolingBlock(in_channels, 3, bottleneck_pool_block))
        

    def forward(self, x):
        # DOWN
        for i in range(0, len(self.downs), 2):
            x = self.downs[i](x)
            x = self.downs[i+1](x)

        for i in range(2):
            x = self.bottleneck[i](x)
     
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
