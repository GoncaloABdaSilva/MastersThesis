import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            #nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            #nn.Conv2d(out_channels, out_channels, 3, 1, 0, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class DoubleConvWithDilation(nn.Module):
    def __init__(self, in_channels, out_channels, dillation_n):
        super(DoubleConvWithDilation, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=False, dilation=dillation_n),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 0, bias=False, dilation=dillation_n),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
            #self, in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class ThinCrack_UNET(nn.Module):
    def __init__(
            #self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
            self, in_channels=3, out_channels=1, features=[16, 32, 64, 128],
    ):
        super(ThinCrack_UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.tile_padding = nn.ConstantPad2d(69, 0)
        #self.tile_padding = nn.ReplicationPad2d(69)

        # Down part of ThinCrackUNET
        self.downs.append(DoubleConv(in_channels, features[0]))
        self.downs.append(DoubleConv(features[0], features[1])) 
        self.downs.append(DoubleConvWithDilation(features[1], features[2], 2))
        self.downs.append(DoubleConvWithDilation(features[2], features[3], 4))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        #Up part of ThinCrackUNET
        self.ups.append(nn.Conv2d(features[3]*2, features[3], 3, 1, 1, bias=False))
        self.ups.append(DoubleConvWithDilation(features[3]*2, features[3], 4))
        
        self.ups.append(nn.Conv2d(features[3], features[2], 3, 1, 1, bias=False))
        self.ups.append(DoubleConvWithDilation(features[3], features[2], 2))
        
        self.ups.append(nn.Conv2d(features[2], features[1], 3, 1, 1, bias=False))
        self.ups.append(DoubleConv(features[2], features[1]))

        self.ups.append(nn.Conv2d(features[1], features[0], 3, 1, 1, bias=False))
        self.ups.append(DoubleConv(features[1], features[0]))

        self.final_conv = nn.ModuleList()
        self.final_conv.append(nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2))
        self.final_conv.append(nn.Conv2d(features[0], out_channels, kernel_size=1))
        self.final_conv.append(nn.Sigmoid())


    def forward(self, x):
        skip_connections = []

        #print(f"Fisrt Torch: {x}")

        x = self.tile_padding(x)
        #x = TF.resize(x, size=[650,650])
        x = self.pool(x)

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            #x=x As in X is coppied
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                #skip_connection = TF.resize(skip_connection, size=x.shape[2:])
                side_crop = int((skip_connection.shape[2]-x.shape[2])/2)
                
                skip_connection = skip_connection[:,:,side_crop:-side_crop,side_crop:-side_crop]            

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip) 
        
        x = self.final_conv[0](x) #514x514, 512x512 needed

        #x = TF.resize(x, size=[512, 512])
        x = x[:,:,1:-1,1:-1]
        
        x = self.final_conv[1](x)
        #print(f"Last Torch before sigmoid: {x}")

        #temp = nn.Sigmoid()
        #con = temp(x)
        #print(f"Last Torch After sigmoid: {con}")

        return self.final_conv[2](x)
