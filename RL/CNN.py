import gymnasium as gym
import tensorflow as tf
from stable_baselines3.common.policies import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy


class CustomCNN(BaseFeaturesExtractor):
    """Custom CNN architecture"""
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn = tf.keras.Sequential([
            tf.keras.layers.InputLayer(
                input_shape=observation_space.shape),
            tf.keras.layers.Conv2D(
                1024, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(
                512, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(
                256, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.GlobalMaxPooling2D(),
        ])


        self.linear = tf.keras.Sequential([
            tf.keras.layers.Dense(features_dim, activation='relu')
        ])

    def call(self, observations: tf.Tensor) -> tf.Tensor:
        """Process the observations through the CNN and linear layers."""
        x = self.cnn(observations)
        return self.linear(x)


class CustomCnnTD3Policy(TD3Policy):
    """Custom TD3 policy that uses the CustomCNN features extractor."""
    def __init__(self, *args, **kwargs):
        super(CustomCnnTD3Policy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )
