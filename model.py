import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    residual block with two convolutional layers, batch normalization, and relu activation.
    """
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        # residual connection
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # residual connection
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    """
    convolutional neural network for chess position evaluation.
    architecture: initial conv → residual blocks → policy head + value head.
    
    input: (batch_size, 18, 8, 8) - board tensor from encode.py
    policy output: (batch_size, 8, 8, 73) - log probabilities over moves
    value output: (batch_size, 1) - position value in [-1, 1]
    """
    
    def __init__(self, num_residual_blocks=8, num_channels=256):
        """
        initialize the chess neural network.
        
        args:
            num_residual_blocks: number of residual blocks (default: 8)
            num_channels: number of channels in residual blocks (default: 256)
        """
        super(ChessNet, self).__init__()
        
        self.num_residual_blocks = num_residual_blocks
        self.num_channels = num_channels
        
        # initial convolution: 18 input planes → num_channels
        self.initial_conv = nn.Conv2d(18, num_channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(num_channels)
        
        # residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_residual_blocks)
        ])
        
        # policy head: outputs 8×8×73 move probabilities
        self.policy_conv = nn.Conv2d(num_channels, 73, kernel_size=1)
        
        # value head: outputs single scalar value
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        initialize network weights using he initialization for conv layers
        and xavier initialization for linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # special initialization for value head final layer to help it learn
        # initialize to small random values so it doesn't start at exactly 0
        if hasattr(self, 'value_fc2'):
            nn.init.normal_(self.value_fc2.weight, mean=0.0, std=0.01)
            if self.value_fc2.bias is not None:
                nn.init.constant_(self.value_fc2.bias, 0.0)
    
    def forward(self, x):
        """
        forward pass through the network.
        
        args:
            x: input tensor of shape (batch_size, 18, 8, 8)
        
        returns:
            policy: log probabilities of shape (batch_size, 8, 8, 73)
            value: position value of shape (batch_size, 1)
        """
        # initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # policy head
        policy = self.policy_conv(x)  # (batch_size, 73, 8, 8)
        policy = policy.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 73)
        policy = F.log_softmax(policy, dim=-1)  # log probabilities
        
        # value head
        value = self.value_conv(x)  # (batch_size, 1, 8, 8)
        value = value.view(value.size(0), -1)  # (batch_size, 64)
        value = F.relu(self.value_fc1(value))  # (batch_size, 256)
        value = torch.tanh(self.value_fc2(value))  # (batch_size, 1)
        
        return policy, value

