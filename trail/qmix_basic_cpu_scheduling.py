
import torch
import torch.nn as nn

import torch.nn.functional as F

class CPUAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_dim)

    def forward(self, obs):
        return self.fc2(F.relu(self.fc1(obs)))

class CPUMixer(nn.Module):
    def __init__(self, n_agents, state_dim, hidden=32):
        super().__init__()
        self.w1 = nn.Linear(state_dim, n_agents * hidden)
        self.b1 = nn.Linear(state_dim, hidden)
        self.w2 = nn.Linear(state_dim, hidden)
        self.b2 = nn.Linear(state_dim, 1)
        self.n_agents = n_agents
        self.hidden = hidden

    def forward(self, agent_qs, state):
        bs = agent_qs.size(0)
        w1 = torch.abs(self.w1(state)).view(bs, self.n_agents, self.hidden)
        b1 = self.b1(state).view(bs, 1, self.hidden)
        hidden = F.elu(torch.bmm(agent_qs.view(bs, 1, self.n_agents), w1) + b1)
        w2 = torch.abs(self.w2(state)).view(bs, self.hidden, 1)
        b2 = self.b2(state).view(bs, 1, 1)
        return torch.bmm(hidden, w2) + b2

n_cpus = 4
obs_dim = 6   # process features like arrival, burst, priority, etc.
act_dim = 5   # number of possible processes to pick
state_dim = 20
batch_size = 8

agents = [CPUAgent(obs_dim, act_dim) for _ in range(n_cpus)]
mixer = CPUMixer(n_cpus, state_dim)

# Fake environment inputs
observations = [torch.rand(batch_size, obs_dim) for _ in range(n_cpus)]
global_state = torch.rand(batch_size, state_dim)

agent_qs = torch.stack([agent(obs) for agent, obs in zip(agents, observations)], dim=1)  # (bs, n_agents, act_dim)

# Assume action selection externally; here we sample randomly
chosen_actions = agent_qs.gather(2, torch.randint(0, act_dim, (batch_size, n_cpus, 1)))
global_q = mixer(chosen_actions.squeeze(-1), global_state)

print("Global Q (system-wide scheduling quality):", global_q)

