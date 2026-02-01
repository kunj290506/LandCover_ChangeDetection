import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class FCEF(nn.Module):
    """Fully Convolutional Early Fusion"""
    def __init__(self, in_channels=6, num_classes=2):
        super(FCEF, self).__init__()
        # Encoder
        self.conv1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = ConvBlock(64, 128)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv5 = ConvBlock(128 + 128, 64) 
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(64 + 64, 32)
        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(32 + 32, 16)
        self.up4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.conv8_pre = ConvBlock(16 + 16, 16) # Add block to process final cat
        self.conv8 = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        
        u1 = self.up1(p4)
        u1 = torch.cat((u1, c4), dim=1)
        c5 = self.conv5(u1)
        
        u2 = self.up2(c5)
        u2 = torch.cat((u2, c3), dim=1)
        c6 = self.conv6(u2)
        
        u3 = self.up3(c6)
        u3 = torch.cat((u3, c2), dim=1)
        c7 = self.conv7(u3)
        
        u4 = self.up4(c7)
        u4 = torch.cat((u4, c1), dim=1)
        c8 = self.conv8_pre(u4)
        out = self.conv8(c8)
        return out

class FCSiamDiff(nn.Module):
    """Fully Convolutional Siamese Difference"""
    def __init__(self, in_channels=3, num_classes=2):
        super(FCSiamDiff, self).__init__()
        # Encoder (Shared)
        self.conv1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = ConvBlock(64, 128)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv5 = ConvBlock(128 + 128, 64) 
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(64 + 64, 32)
        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(32 + 32, 16)
        self.up4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.conv8_pre = ConvBlock(16 + 16, 16)
        self.conv8 = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x1, x2):
        # Stream 1
        c1_1 = self.conv1(x1); p1_1 = self.pool1(c1_1)
        c2_1 = self.conv2(p1_1); p2_1 = self.pool2(c2_1)
        c3_1 = self.conv3(p2_1); p3_1 = self.pool3(c3_1)
        c4_1 = self.conv4(p3_1); p4_1 = self.pool4(c4_1)
        
        # Stream 2
        c1_2 = self.conv1(x2); p1_2 = self.pool1(c1_2)
        c2_2 = self.conv2(p1_2); p2_2 = self.pool2(c2_2)
        c3_2 = self.conv3(p2_2); p3_2 = self.pool3(c3_2)
        c4_2 = self.conv4(p3_2); p4_2 = self.pool4(c4_2)
        
        bn_diff = torch.abs(p4_1 - p4_2)
        
        u1 = self.up1(bn_diff)
        skip3 = torch.abs(c4_1 - c4_2)
        u1 = torch.cat((u1, skip3), dim=1)
        c5 = self.conv5(u1)
        
        u2 = self.up2(c5)
        skip2 = torch.abs(c3_1 - c3_2)
        u2 = torch.cat((u2, skip2), dim=1)
        c6 = self.conv6(u2)
        
        u3 = self.up3(c6)
        skip1 = torch.abs(c2_1 - c2_2)
        u3 = torch.cat((u3, skip1), dim=1)
        c7 = self.conv7(u3)
        
        u4 = self.up4(c7)
        skip0 = torch.abs(c1_1 - c1_2)
        u4 = torch.cat((u4, skip0), dim=1)
        c8 = self.conv8_pre(u4)
        out = self.conv8(c8)
        return out

class FCSiamConc_Fixed(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(FCSiamConc_Fixed, self).__init__()
        # Encoder (Shared)
        self.conv1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = ConvBlock(64, 128)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) 
        self.conv5 = ConvBlock(128 + 64 + 64, 64) 
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(64 + 32 + 32, 32)
        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(32 + 16 + 16, 16)
        self.up4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.conv8_pre = ConvBlock(16 + 16 + 16, 16)
        self.conv8 = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x1, x2):
        c1_1 = self.conv1(x1); p1_1 = self.pool1(c1_1)
        c2_1 = self.conv2(p1_1); p2_1 = self.pool2(c2_1)
        c3_1 = self.conv3(p2_1); p3_1 = self.pool3(c3_1)
        c4_1 = self.conv4(p3_1); p4_1 = self.pool4(c4_1)
        
        c1_2 = self.conv1(x2); p1_2 = self.pool1(c1_2)
        c2_2 = self.conv2(p1_2); p2_2 = self.pool2(c2_2)
        c3_2 = self.conv3(p2_2); p3_2 = self.pool3(c3_2)
        c4_2 = self.conv4(p3_2); p4_2 = self.pool4(c4_2)
        
        bn = torch.cat((p4_1, p4_2), dim=1)
        
        u1 = self.up1(bn)
        u1 = torch.cat((u1, c4_1, c4_2), dim=1) # skip c4
        c5 = self.conv5(u1)
        
        u2 = self.up2(c5)
        u2 = torch.cat((u2, c3_1, c3_2), dim=1) # skip c3
        c6 = self.conv6(u2)
        
        u3 = self.up3(c6)
        u3 = torch.cat((u3, c2_1, c2_2), dim=1) # skip c2
        c7 = self.conv7(u3)
        
        u4 = self.up4(c7)
        u4 = torch.cat((u4, c1_1, c1_2), dim=1) # skip c1
        c8 = self.conv8_pre(u4)
        out = self.conv8(c8)
        return out
