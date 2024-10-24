import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3,
                      stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 4)),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(
                observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CustomCnnTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super(CustomCnnTD3Policy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(
                features_dim=512),
        )
