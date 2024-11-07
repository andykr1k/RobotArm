import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN architecture for processing image observations in reinforcement learning.
    
    Architecture:
    1. Three convolutional layers with increasing channels (32->64->128)
    2. Adaptive max pooling to reduce spatial dimensions
    3. Flattening layer
    4. Final linear layer to project to desired feature dimension
    
    Args:
        observation_space (gym.spaces.Box): The observation space of the environment
        features_dim (int): Number of features to extract (default: 512)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # CNN architecture
        self.cnn = nn.Sequential(
            # Layer 1: (N, n_input_channels, H, W) -> (N, 32, H/2, W/2)
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 2: (N, 32, H/2, W/2) -> (N, 64, H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 3: (N, 64, H/4, W/4) -> (N, 128, H/4, W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Adaptive pooling to fixed size
            nn.AdaptiveMaxPool2d((4, 4)),
            nn.Flatten(),
        )

        # Compute the flatten size dynamically
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # Final linear layer to get desired feature dimension
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        # Optional: Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Process the observations through the CNN and linear layers.
        
        Args:
            observations (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, features_dim)
        """
        return self.linear(self.cnn(observations))


class CustomCnnTD3Policy(TD3Policy):
    """
    Custom TD3 policy that uses the CustomCNN features extractor.
    
    This policy can be used with environments that have image observations.
    """
    def __init__(self, *args, **kwargs):
        super(CustomCnnTD3Policy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )