import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import CBAM

# Basic Convolution Block (Modified to optionally include CBAM)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_cbam=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_cbam:
            x = self.cbam(x)
        return x

# Siamese Nested U-Net (SNUNet) with Attention
class SNUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_channel=32, use_attention=True):
        super(SNUNet, self).__init__()
        
        self.use_attention = use_attention

        # Encoder (Siamese - Shared Weights)
        # Level 1
        self.conv0_0 = ConvBlock(in_channels, base_channel)
        # Level 2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1_0 = ConvBlock(base_channel, base_channel * 2)
        # Level 3
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2_0 = ConvBlock(base_channel * 2, base_channel * 4)
        # Level 4
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv3_0 = ConvBlock(base_channel * 4, base_channel * 8)
        # Level 5
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv4_0 = ConvBlock(base_channel * 8, base_channel * 16)

        # Decoder (Nested / UNet++) with optional Attention
        # Upsampling layers
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Node 0_1
        # Inputs: E1_0(C)+E2_0(C) + Up(E1_1)(2C)+Up(E2_1)(2C) = 6C
        self.conv0_1 = ConvBlock(base_channel*2 + base_channel*4, base_channel, use_cbam=use_attention) 
        # Inputs: E1_1(2C)+E2_1(2C) + Up(E1_2)(4C)+Up(E2_2)(4C) = 12C
        self.conv1_1 = ConvBlock(base_channel*4 + base_channel*8, base_channel * 2, use_cbam=use_attention)
        # Inputs: E1_2(4C)+E2_2(4C) + Up(E1_3)(8C)+Up(E2_3)(8C) = 24C
        self.conv2_1 = ConvBlock(base_channel*8 + base_channel*16, base_channel * 4, use_cbam=use_attention)
        # Inputs: E1_3(8C)+E2_3(8C) + Up(E1_4)(16C)+Up(E2_4)(16C) = 48C
        self.conv3_1 = ConvBlock(base_channel*16 + base_channel*32, base_channel * 8, use_cbam=use_attention)

        # Node 0_2
        self.conv0_2 = ConvBlock(base_channel*2 + base_channel*2 + base_channel, base_channel, use_cbam=use_attention)
        self.conv1_2 = ConvBlock(base_channel*4 + base_channel*4 + base_channel*2, base_channel * 2, use_cbam=use_attention)
        self.conv2_2 = ConvBlock(base_channel*8 + base_channel*8 + base_channel*4, base_channel * 4, use_cbam=use_attention)
        
        # X_0_3
        self.conv0_3 = ConvBlock(base_channel*2 + base_channel*2 + base_channel + base_channel, base_channel, use_cbam=use_attention)
        self.conv1_3 = ConvBlock(base_channel*4 + base_channel*4 + base_channel*2 + base_channel*2, base_channel * 2, use_cbam=use_attention)
        
        # X_0_4
        self.conv0_4 = ConvBlock(base_channel*2 + base_channel*2 + base_channel + base_channel + base_channel, base_channel)
        
        # Classifier
        self.final = nn.Conv2d(base_channel, num_classes, kernel_size=1)
        
    def forward(self, x1, x2):
        # Encoder 1
        x1_0_0 = self.conv0_0(x1)
        x1_1_0 = self.conv1_0(self.pool1(x1_0_0))
        x1_2_0 = self.conv2_0(self.pool2(x1_1_0))
        x1_3_0 = self.conv3_0(self.pool3(x1_2_0))
        x1_4_0 = self.conv4_0(self.pool4(x1_3_0))
        
        # Encoder 2
        x2_0_0 = self.conv0_0(x2)
        x2_1_0 = self.conv1_0(self.pool1(x2_0_0))
        x2_2_0 = self.conv2_0(self.pool2(x2_1_0))
        x2_3_0 = self.conv3_0(self.pool3(x2_2_0))
        x2_4_0 = self.conv4_0(self.pool4(x2_3_0))
        
        # Decoder - same flow, just with potentially different ConvBlocks (with CBAM)
        
        # Node 0_1
        up_1_0_1 = self.up(x1_1_0)
        up_1_0_2 = self.up(x2_1_0)
        x0_1 = self.conv0_1(torch.cat([x1_0_0, x2_0_0, up_1_0_1, up_1_0_2], dim=1))
        
        # Node 1_1
        up_2_0_1 = self.up(x1_2_0)
        up_2_0_2 = self.up(x2_2_0)
        x1_1 = self.conv1_1(torch.cat([x2_1_0, x1_1_0, up_2_0_1, up_2_0_2], dim=1)) # Concat order matters relative to definition? Actually if just concat, no.
        
        # Node 2_1
        up_3_0_1 = self.up(x1_3_0)
        up_3_0_2 = self.up(x2_3_0)
        x2_1 = self.conv2_1(torch.cat([x1_2_0, x2_2_0, up_3_0_1, up_3_0_2], dim=1))
        
        # Node 3_1
        up_4_0_1 = self.up(x1_4_0)
        up_4_0_2 = self.up(x2_4_0)
        x3_1 = self.conv3_1(torch.cat([x1_3_0, x2_3_0, up_4_0_1, up_4_0_2], dim=1))
        
        # Node 0_2
        up_1_1 = self.up(x1_1)
        x0_2 = self.conv0_2(torch.cat([x1_0_0, x2_0_0, x0_1, up_1_1], dim=1))
        
        # Node 1_2
        up_2_1 = self.up(x2_1)
        x1_2 = self.conv1_2(torch.cat([x1_1_0, x2_1_0, x1_1, up_2_1], dim=1))
        
        # Node 2_2
        up_3_1 = self.up(x3_1)
        x2_2 = self.conv2_2(torch.cat([x1_2_0, x2_2_0, x2_1, up_3_1], dim=1))
        
        # Node 0_3
        up_1_2 = self.up(x1_2)
        x0_3 = self.conv0_3(torch.cat([x1_0_0, x2_0_0, x0_1, x0_2, up_1_2], dim=1))
        
        # Node 1_3
        up_2_2 = self.up(x2_2)
        x1_3 = self.conv1_3(torch.cat([x1_1_0, x2_1_0, x1_1, x1_2, up_2_2], dim=1))
        
        # Node 0_4
        up_1_3 = self.up(x1_3)
        x0_4 = self.conv0_4(torch.cat([x1_0_0, x2_0_0, x0_1, x0_2, x0_3, up_1_3], dim=1))
        
        # Output
        out = self.final(x0_4) 
        
        return out

if __name__ == '__main__':
    # Test
    model = SNUNet(3, 2)
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    y = model(x1, x2)
    print(y.shape)
