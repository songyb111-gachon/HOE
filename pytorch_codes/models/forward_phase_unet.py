"""
PyTorch Forward Phase Map Prediction U-Net Model
Random Pillar Pattern → Phase Map

Features:
- Input: Binary mask (random pillar pattern)
- Output: Phase map (continuous values)
- Based on U-Net architecture for image-to-image regression
"""

import torch
import torch.nn as nn
from .unet_blocks import ConvBlock, EncoderBlock, DecoderBlock, OutputBlock


class ForwardPhaseUNet(nn.Module):
    """U-Net for forward phase map prediction
    
    Predicts phase map from random pillar pattern (MEEP surrogate)
    
    Args:
        in_channels: Number of input channels (default: 1 for binary mask)
        out_channels: Number of output channels (default: 1 for phase map)
        layer_num: Number of encoder/decoder layers (4-7)
        base_features: Base number of features
        kernel_size: Convolution kernel size
        dropout_rate: Dropout rate
        output_activation: Output activation ('linear', 'tanh', 'sigmoid')
    """
    
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 layer_num=5,
                 base_features=64,
                 kernel_size=3,
                 dropout_rate=0.2,
                 output_activation='linear',
                 use_batchnorm=True):
        super(ForwardPhaseUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_num = layer_num
        self.use_batchnorm = use_batchnorm
        
        # Feature numbers for each layer
        self.feature_nums = [base_features * (2 ** i) for i in range(layer_num)]
        
        # Encoder path
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for i in range(layer_num):
            self.encoders.append(
                EncoderBlock(current_channels, self.feature_nums[i], 
                           kernel_size, dropout_rate, use_batchnorm)
            )
            current_channels = self.feature_nums[i]
        
        # Bottleneck
        bottleneck_features = self.feature_nums[-1] * 2
        self.bottleneck = ConvBlock(self.feature_nums[-1], 
                                    bottleneck_features, 
                                    kernel_size, dropout_rate, use_batchnorm)
        
        # Decoder path
        self.decoders = nn.ModuleList()
        current_channels = bottleneck_features
        for i in range(layer_num - 1, -1, -1):
            self.decoders.append(
                DecoderBlock(current_channels, self.feature_nums[i], 
                           self.feature_nums[i], kernel_size, dropout_rate, use_batchnorm)
            )
            current_channels = self.feature_nums[i]
        
        # Output layer (phase map)
        self.output_layer = OutputBlock(base_features, out_channels, 
                                       activation=output_activation)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W) - random pillar pattern
            
        Returns:
            Phase map tensor (B, C_out, H, W)
        """
        # Encoder path (save skip connections)
        skip_connections = []
        for encoder in self.encoders:
            conv, x = encoder(x)
            skip_connections.append(conv)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path (use skip connections in reverse order)
        skip_connections = skip_connections[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        
        # Output (phase map)
        output = self.output_layer(x)
        
        return output
    
    def get_model_summary(self):
        """Print model summary"""
        print("=" * 80)
        print(f"Forward Phase Map Prediction U-Net")
        print("=" * 80)
        print(f"  • Task: Random Pillar Pattern → Phase Map")
        print(f"  • Input channels: {self.in_channels} (binary mask)")
        print(f"  • Output channels: {self.out_channels} (phase map)")
        print(f"  • Number of layers: {self.layer_num}")
        print(f"  • Feature progression: {self.feature_nums}")
        print(f"  • Bottleneck features: {self.feature_nums[-1] * 2}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  • Total parameters: {total_params:,}")
        print(f"  • Trainable parameters: {trainable_params:,}")
        print("=" * 80)


class MultiScalePhaseUNet(nn.Module):
    """Multi-scale U-Net for better phase map prediction
    
    Uses multiple decoder outputs at different scales
    """
    
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 layer_num=5,
                 base_features=64,
                 kernel_size=3,
                 dropout_rate=0.2,
                 use_batchnorm=True):
        super(MultiScalePhaseUNet, self).__init__()
        
        self.layer_num = layer_num
        self.feature_nums = [base_features * (2 ** i) for i in range(layer_num)]
        self.use_batchnorm = use_batchnorm
        
        # Encoder path
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for i in range(layer_num):
            self.encoders.append(
                EncoderBlock(current_channels, self.feature_nums[i], 
                           kernel_size, dropout_rate, use_batchnorm)
            )
            current_channels = self.feature_nums[i]
        
        # Bottleneck
        bottleneck_features = self.feature_nums[-1] * 2
        self.bottleneck = ConvBlock(self.feature_nums[-1], 
                                    bottleneck_features, 
                                    kernel_size, dropout_rate, use_batchnorm)
        
        # Decoder path
        self.decoders = nn.ModuleList()
        current_channels = bottleneck_features
        for i in range(layer_num - 1, -1, -1):
            self.decoders.append(
                DecoderBlock(current_channels, self.feature_nums[i], 
                           self.feature_nums[i], kernel_size, dropout_rate, use_batchnorm)
            )
            current_channels = self.feature_nums[i]
        
        # Multi-scale outputs (from each decoder level)
        self.output_layers = nn.ModuleList()
        for i in range(layer_num):
            self.output_layers.append(
                nn.Conv2d(self.feature_nums[i], out_channels, kernel_size=1)
            )
        
        # Final fusion
        self.final_conv = nn.Conv2d(layer_num * out_channels, out_channels, 
                                    kernel_size=1)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder path
        skip_connections = []
        for encoder in self.encoders:
            conv, x = encoder(x)
            skip_connections.append(conv)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with multi-scale outputs
        skip_connections = skip_connections[::-1]
        decoder_outputs = []
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
            
            # Generate output at this scale
            scale_output = self.output_layers[i](x)
            
            # Upsample to input size
            if scale_output.shape[2:] != input_size:
                scale_output = nn.functional.interpolate(
                    scale_output, size=input_size, 
                    mode='bilinear', align_corners=False
                )
            
            decoder_outputs.append(scale_output)
        
        # Fuse multi-scale outputs
        fused = torch.cat(decoder_outputs, dim=1)
        output = self.final_conv(fused)
        
        return output


class PhaseAmplitudeUNet(nn.Module):
    """U-Net that predicts both phase and amplitude
    
    Two-output model for complete electromagnetic field prediction
    """
    
    def __init__(self,
                 in_channels=1,
                 layer_num=5,
                 base_features=64,
                 kernel_size=3,
                 dropout_rate=0.2,
                 use_batchnorm=True):
        super(PhaseAmplitudeUNet, self).__init__()
        
        self.layer_num = layer_num
        self.feature_nums = [base_features * (2 ** i) for i in range(layer_num)]
        self.use_batchnorm = use_batchnorm
        
        # Shared encoder
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for i in range(layer_num):
            self.encoders.append(
                EncoderBlock(current_channels, self.feature_nums[i], 
                           kernel_size, dropout_rate, use_batchnorm)
            )
            current_channels = self.feature_nums[i]
        
        # Shared bottleneck
        bottleneck_features = self.feature_nums[-1] * 2
        self.bottleneck = ConvBlock(self.feature_nums[-1], 
                                    bottleneck_features, 
                                    kernel_size, dropout_rate, use_batchnorm)
        
        # Separate decoders for phase and amplitude
        self.phase_decoders = self._build_decoder(bottleneck_features, dropout_rate, kernel_size, use_batchnorm)
        self.amplitude_decoders = self._build_decoder(bottleneck_features, dropout_rate, kernel_size, use_batchnorm)
        
        # Output layers
        self.phase_output = OutputBlock(base_features, 1, activation='linear')
        self.amplitude_output = OutputBlock(base_features, 1, activation='sigmoid')
    
    def _build_decoder(self, bottleneck_features, dropout_rate, kernel_size, use_batchnorm):
        """Build a decoder path"""
        decoders = nn.ModuleList()
        current_channels = bottleneck_features
        
        for i in range(self.layer_num - 1, -1, -1):
            decoders.append(
                DecoderBlock(current_channels, self.feature_nums[i], 
                           self.feature_nums[i], kernel_size, dropout_rate, use_batchnorm)
            )
            current_channels = self.feature_nums[i]
        
        return decoders
    
    def forward(self, x):
        """
        Returns:
            phase: Phase map (B, 1, H, W)
            amplitude: Amplitude map (B, 1, H, W)
        """
        # Shared encoder
        skip_connections = []
        for encoder in self.encoders:
            conv, x = encoder(x)
            skip_connections.append(conv)
        
        # Shared bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections
        skip_connections = skip_connections[::-1]
        
        # Phase decoder
        phase_x = x
        for i, decoder in enumerate(self.phase_decoders):
            phase_x = decoder(phase_x, skip_connections[i])
        phase = self.phase_output(phase_x)
        
        # Amplitude decoder
        amp_x = x
        for i, decoder in enumerate(self.amplitude_decoders):
            amp_x = decoder(amp_x, skip_connections[i])
        amplitude = self.amplitude_output(amp_x)
        
        return phase, amplitude


if __name__ == "__main__":
    # Test models
    print("\n=== Testing Forward Phase U-Net ===\n")
    
    # Basic model
    model = ForwardPhaseUNet(
        in_channels=1,
        out_channels=1,
        layer_num=5,
        base_features=64
    )
    
    model.get_model_summary()
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)  # Random pillar pattern
    phase_map = model(x)
    print(f"\nInput (pillar pattern) shape: {x.shape}")
    print(f"Output (phase map) shape: {phase_map.shape}")
    
    # Multi-scale model
    print("\n=== Testing Multi-Scale Phase U-Net ===\n")
    model_ms = MultiScalePhaseUNet(
        in_channels=1,
        out_channels=1,
        layer_num=4,
        base_features=32
    )
    
    output = model_ms(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Phase-Amplitude model
    print("\n=== Testing Phase-Amplitude U-Net ===\n")
    model_pa = PhaseAmplitudeUNet(
        in_channels=1,
        layer_num=4,
        base_features=32
    )
    
    phase, amplitude = model_pa(x)
    print(f"Input shape: {x.shape}")
    print(f"Phase output shape: {phase.shape}")
    print(f"Amplitude output shape: {amplitude.shape}")
    
    print("\n✓ All models working correctly!")

