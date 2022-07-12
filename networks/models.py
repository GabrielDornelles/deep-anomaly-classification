import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """(Convolution => BN => Leaky ReLU => MaxPool)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
       
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    """(Transposed Convolution => BN => Leaky ReLU => UpSample)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
    
    def forward(self, x):
        return self.deconv(x)
    
        
class DeepAutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.rep_dim = 512 
        
        # Encoder 
        self.conv1 = DownBlock(1,32)
        self.conv2 = DownBlock(32,64)
        self.conv3 = DownBlock(64,64)
        self.conv4 = DownBlock(64,64)
        self.conv5 = DownBlock(64,32)
    
        # Decoder
        self.deconv1 = UpBlock(32, 32)
        self.deconv2 = UpBlock(32,64)
        self.deconv3 = UpBlock(64,64)
        self.deconv4 = UpBlock(64,64)
        self.deconv5 = UpBlock(64,1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
    
        # Decoder
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.rep_dim = 512
        
        # Encoder (must match the Deep SVDD network above)
        self.conv1 = DownBlock(1,32)
        self.conv2 = DownBlock(32,64)
        self.conv3 = DownBlock(64,64)
        self.conv4 = DownBlock(64,64)
        self.conv5 = DownBlock(64,32)


    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1) # 512 latent space
        return x

if __name__ == "__main__":

    print("Encoder")
    sample = torch.randn((1,3,128,128))
    model = Encoder()
    output = model(sample)
    print(output.shape)

    print("AutoEncoder")
    sample = torch.randn((1,3,128,128))
    model = DeepAutoEncoder()
    output = model(sample)
    print(output.shape)
