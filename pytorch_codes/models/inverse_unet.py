"""
PyTorch Inverse Design U-Net Model
Converted from TensorFlow/Keras inverse_codes

Features:
- Variable depth (1-7 encoder/decoder layers)
- Multiple output heads
- Flexible feature numbers
"""

import torch
import torch.nn as nn
from .unet_blocks import ConvBlock, EncoderBlock, DecoderBlock, OutputBlock


class InverseUNet(nn.Module):
    """U-Net for inverse design tasks
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (list for multiple outputs)
        layer_num: Number of encoder/decoder layers (1-7)
        base_features: Base number of features (doubled at each layer)
        kernel_size: Convolution kernel size
        dropout_rate: Dropout rate
        output_activations: Activation for each output ('linear', 'softmax', 'sigmoid')
    """
    
    def __init__(self, 
                 in_channels=1,
                 out_channels=[1],  # List for multiple outputs
                 layer_num=4,
                 base_features=32,
                 kernel_size=3,
                 dropout_rate=0.2,
                 output_activations=['linear'],
                 use_batchnorm=True):
        super(InverseUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels if isinstance(out_channels, list) else [out_channels]
        self.layer_num = layer_num
        self.num_outputs = len(self.out_channels)
        self.use_batchnorm = use_batchnorm
        
        # Calculate feature numbers for each layer
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
        self.bottleneck = ConvBlock(self.feature_nums[-1], 
                                    self.feature_nums[-1] * 2, 
                                    kernel_size, dropout_rate, use_batchnorm)
        
        # Decoder path
        self.decoders = nn.ModuleList()
        current_channels = self.feature_nums[-1] * 2
        for i in range(layer_num - 1, -1, -1):
            self.decoders.append(
                DecoderBlock(current_channels, self.feature_nums[i], 
                           self.feature_nums[i], kernel_size, dropout_rate, use_batchnorm)
            )
            current_channels = self.feature_nums[i]
        
        # Output layers (multiple heads)
        self.output_layers = nn.ModuleList()
        for i in range(self.num_outputs):
            activation = output_activations[i] if i < len(output_activations) else 'linear'
            self.output_layers.append(
                OutputBlock(base_features, self.out_channels[i], activation)
            )
    
    def forward(self, x):
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
        
        # Multiple outputs
        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(x))
        
        if self.num_outputs == 1:
            return outputs[0]
        else:
            return outputs
    
    def get_model_summary(self):
        """Print model summary"""
        print("=" * 80)
        print(f"Inverse U-Net Model Summary")
        print("=" * 80)
        print(f"  • Input channels: {self.in_channels}")
        print(f"  • Output channels: {self.out_channels}")
        print(f"  • Number of layers: {self.layer_num}")
        print(f"  • Feature progression: {self.feature_nums}")
        print(f"  • Bottleneck features: {self.feature_nums[-1] * 2}")
        print(f"  • Number of outputs: {self.num_outputs}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  • Total parameters: {total_params:,}")
        print(f"  • Trainable parameters: {trainable_params:,}")
        print("=" * 80)


class MultiTaskInverseUNet(nn.Module):
    """Multi-task U-Net with separate decoder paths for each task
    
    TensorFlow equivalent: Models with multiple encoder/decoder paths
    """
    
    def __init__(self,
                 in_channels=1,
                 task_out_channels=[1, 1, 1],  # Output channels for each task
                 layer_num=4,
                 base_features=32,
                 kernel_size=3,
                 dropout_rate=0.2,
                 output_activations=['linear', 'linear', 'linear'],
                 use_batchnorm=True):
        super(MultiTaskInverseUNet, self).__init__()
        
        self.num_tasks = len(task_out_channels)
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
        self.bottleneck = ConvBlock(self.feature_nums[-1], 
                                    self.feature_nums[-1] * 2, 
                                    kernel_size, dropout_rate, use_batchnorm)
        
        # Task-specific decoders
        self.task_decoders = nn.ModuleList()
        self.task_outputs = nn.ModuleList()
        
        for task_idx in range(self.num_tasks):
            # Decoder for this task
            task_decoder = nn.ModuleList()
            current_channels = self.feature_nums[-1] * 2
            for i in range(layer_num - 1, -1, -1):
                task_decoder.append(
                    DecoderBlock(current_channels, self.feature_nums[i], 
                               self.feature_nums[i], kernel_size, dropout_rate, use_batchnorm)
                )
                current_channels = self.feature_nums[i]
            self.task_decoders.append(task_decoder)
            
            # Output for this task
            activation = output_activations[task_idx] if task_idx < len(output_activations) else 'linear'
            self.task_outputs.append(
                OutputBlock(base_features, task_out_channels[task_idx], activation)
            )
    
    def forward(self, x):
        # Shared encoder
        skip_connections = []
        for encoder in self.encoders:
            conv, x = encoder(x)
            skip_connections.append(conv)
        
        # Shared bottleneck
        x = self.bottleneck(x)
        
        # Task-specific decoders
        task_outputs = []
        skip_connections_reversed = skip_connections[::-1]
        
        for task_idx in range(self.num_tasks):
            # Decode for this task
            task_x = x
            for i, decoder in enumerate(self.task_decoders[task_idx]):
                task_x = decoder(task_x, skip_connections_reversed[i])
            
            # Generate output for this task
            task_out = self.task_outputs[task_idx](task_x)
            task_outputs.append(task_out)
        
        return task_outputs


if __name__ == "__main__":
    # Test model
    print("\n=== Testing Inverse U-Net ===\n")
    
    # Single output model
    model = InverseUNet(
        in_channels=1,
        out_channels=[1],
        layer_num=4,
        base_features=32,
        dropout_rate=0.2
    )
    
    model.get_model_summary()
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Multiple output model
    print("\n=== Testing Multi-Output Inverse U-Net ===\n")
    model_multi = InverseUNet(
        in_channels=1,
        out_channels=[1, 1, 2],
        layer_num=4,
        base_features=32,
        output_activations=['linear', 'linear', 'softmax']
    )
    
    outputs = model_multi(x)
    print(f"Input shape: {x.shape}")
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output {i+1} shape: {out.shape}")
    
    # Multi-task model
    print("\n=== Testing Multi-Task U-Net ===\n")
    model_task = MultiTaskInverseUNet(
        in_channels=1,
        task_out_channels=[1, 1, 1],
        layer_num=4,
        base_features=32
    )
    
    task_outputs = model_task(x)
    print(f"Input shape: {x.shape}")
    print(f"Number of tasks: {len(task_outputs)}")
    for i, out in enumerate(task_outputs):
        print(f"  Task {i+1} output shape: {out.shape}")
    
    print("\n✓ All models working correctly!")

