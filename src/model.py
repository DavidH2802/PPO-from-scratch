import torch
import torch.nn as nn
from torch.distributions import Normal


def init_weights(module, gain):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(256, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.net.apply(lambda m: init_weights(m, gain=2**0.5))
        init_weights(self.mean_head, gain=0.01)

    def forward(self, obs):
        return self.mean_head(self.net(obs))

    def get_action(self, obs):
        mean = self.forward(obs)
        dist = Normal(mean, self.log_std.exp())
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def get_log_prob(self, obs, actions):
        mean = self.forward(obs)
        dist = Normal(mean, self.log_std.exp())
        return dist.log_prob(actions).sum(dim=-1)


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(256, 1)

        self.net.apply(lambda m: init_weights(m, gain=2**0.5))
        init_weights(self.value_head, gain=1.0)

    def forward(self, obs):
        return self.value_head(self.net(obs)).squeeze(-1)

