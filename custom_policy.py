
import torch as th
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

class CentralizedCriticExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=64)
        input_dim = observation_space.shape[-1]  # e.g. (n_cores, max_queue_size * num_features)

        self.shared_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(observation_space.shape), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.shared_net(observations)
