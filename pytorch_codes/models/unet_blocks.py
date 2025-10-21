"""
PyTorch U-Net Building Blocks
Converted from TensorFlow/Keras to PyTorch

기본 U-Net 구성 요소:
- ConvBlock: Double convolution with dropout
- EncoderBlock: ConvBlock + MaxPooling
- DecoderBlock: UpSampling + Concatenate + ConvBlock
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with dropout
    
    TensorFlow equivalent:
        Conv2D -> Dropout -> Conv2D
    
    Args:
        use_batchnorm: If True, use BatchNorm (modern). If False, original U-Net style.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.0, use_batchnorm=True):
        super(ConvBlock, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                               padding=kernel_size//2, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 
                               padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        
        # Initialize weights (He initialization)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock + MaxPooling
    
    TensorFlow equivalent:
        unetOneEncoderBlock
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.0, use_batchnorm=True):
        super(EncoderBlock, self).__init__()
        
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, dropout_rate, use_batchnorm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        conv = self.conv_block(x)
        pool = self.pool(conv)
        return conv, pool  # Return both for skip connection


class DecoderBlock(nn.Module):
    """Decoder block: UpSampling + Concatenate + ConvBlock
    
    TensorFlow equivalent:
        unetOneDecoderBlock
    """
    
    def __init__(self, in_channels, skip_channels, out_channels, 
                 kernel_size=3, dropout_rate=0.0, use_batchnorm=True):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_block = ConvBlock(in_channels + skip_channels, out_channels, 
                                    kernel_size, dropout_rate, use_batchnorm)
    
    def forward(self, x, skip_connection):
        x = self.upsample(x)
        
        # Handle size mismatch (crop or pad skip connection)
        if x.size() != skip_connection.size():
            diff_h = skip_connection.size(2) - x.size(2)
            diff_w = skip_connection.size(3) - x.size(3)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        
        x = torch.cat([x, skip_connection], dim=1)  # Concatenate along channel dim
        x = self.conv_block(x)
        
        return x


class OutputBlock(nn.Module):
    """Final output convolution layer
    
    TensorFlow equivalent:
        Convolution2D(outputChannelNo, (1, 1), activation)
    """
    
    def __init__(self, in_channels, out_channels, activation='linear'):
        super(OutputBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        if activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:  # 'linear' or None
            self.activation = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


if __name__ == "__main__":
    # Test blocks
    print("Testing U-Net Blocks...")
    
    # Test ConvBlock
    conv_block = ConvBlock(3, 64, kernel_size=3, dropout_rate=0.2)
    x = torch.randn(2, 3, 256, 256)
    out = conv_block(x)
    print(f"ConvBlock: {x.shape} -> {out.shape}")
    
    # Test EncoderBlock
    encoder_block = EncoderBlock(3, 64, kernel_size=3, dropout_rate=0.2)
    conv, pool = encoder_block(x)
    print(f"EncoderBlock: {x.shape} -> conv: {conv.shape}, pool: {pool.shape}")
    
    # Test DecoderBlock
    decoder_block = DecoderBlock(128, 64, 64, kernel_size=3, dropout_rate=0.2)
    x2 = torch.randn(2, 128, 64, 64)
    skip = torch.randn(2, 64, 128, 128)
    out = decoder_block(x2, skip)
    print(f"DecoderBlock: {x2.shape} + skip {skip.shape} -> {out.shape}")
    
    # Test OutputBlock
    output_block = OutputBlock(64, 2, activation='softmax')
    out_final = output_block(out)
    print(f"OutputBlock: {out.shape} -> {out_final.shape}")
    
    print("\n✓ All blocks working correctly!")

