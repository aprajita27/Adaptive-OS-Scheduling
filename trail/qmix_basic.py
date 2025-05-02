# trying out basic qmix code, to understand how it works

import torch
import torch.nn as nn

import torch.nn.functional as F

# Individual agent network (could be shared or separate per agent)
class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        return self.fc2(x)

# Mixing network combines agent Q-values into a global Q-value
class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=32):
        super(MixingNetwork, self).__init__()
        self.hyper_w_1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_w_2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b_1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b_2 = nn.Linear(state_dim, 1)
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

    def forward(self, agent_qs, state):
        bs = agent_qs.size(0)

        w1 = torch.abs(self.hyper_w_1(state)).view(bs, self.n_agents, self.hidden_dim)
        b1 = self.hyper_b_1(state).view(bs, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(agent_qs.view(bs, 1, self.n_agents), w1) + b1)

        w2 = torch.abs(self.hyper_w_2(state)).view(bs, self.hidden_dim, 1)
        b2 = self.hyper_b_2(state).view(bs, 1, 1)
        y = torch.bmm(hidden, w2) + b2
        return y.view(-1, 1)

# Sample usage
n_agents = 3
obs_dim = 10
action_dim = 5
state_dim = 30
batch_size = 4

# Create networks
agents = [AgentNetwork(obs_dim, action_dim) for _ in range(n_agents)]
mixer = MixingNetwork(n_agents, state_dim)

# Dummy inputs
obs = [torch.randn(batch_size, obs_dim) for _ in range(n_agents)]
state = torch.randn(batch_size, state_dim)

# Forward pass
agent_qs = torch.stack([agent(o) for agent, o in zip(agents, obs)], dim=1)  # (bs, n_agents, action_dim)
chosen_actions = agent_qs.gather(2, torch.randint(0, action_dim, (batch_size, n_agents, 1)))  # (bs, n_agents, 1)
total_q = mixer(chosen_actions.squeeze(-1), state)

print("Total Q value from mixer:", total_q)
