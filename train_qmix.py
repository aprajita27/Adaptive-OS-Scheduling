print("--- hello in train_qmix, fixed mixing network forward function ---")

import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from multi_core_env import MultiCoreSchedulingEnv

MODEL_PATH = "qmix_cpu_scheduler.pt"

# when using colab
#MODEL_PATH = "/content/drive/MyDrive/CSCI566-S25-Material/DL Project/Adaptive-OS-Scheduling/qmix_cpu_scheduler.pt"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {device}")

# AgentNetwork and MixingNetwork 

# Each CPU core (agent) has its own Q-network.
class AgentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, output_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q = self.q_out(x)
        return q

# mixing network combines agent Q-values into a global Q-value
# conditioned with global q value
# f combines individual q values to global q value
# f is a NN - allows a non linear combination of Q values

class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * 32)
        self.hyper_b1 = nn.Linear(state_dim, 32)
        self.hyper_w2 = nn.Linear(state_dim, 32)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, agent_qs, state):
        """
        agent_qs: Tensor of shape [batch_size, n_agents]
        state: Tensor of shape [batch_size, state_dim]
        """
        bs = agent_qs.size(0)

        # hyper_w1 and hyper_w2 weights for mixing layer
        # hyper_b1 and hyper_b2 generate biases

        # this is the first mixing layer 
        w1 = torch.abs(self.hyper_w1(state)).view(bs, self.n_agents, 32)
        b1 = self.hyper_b1(state).view(bs, 1, 32)

        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # output layer
        # monotonicity - increasing the i/p doesn’t decrease the o/p 
        # monotonicity means function continuously inc or dec 
        w2 = torch.abs(self.hyper_w2(state)).view(bs, 32, 1)  # non neg maintains monotonicitty
        b2 = self.hyper_b2(state).view(bs, 1, 1)

        total_q = torch.bmm(hidden, w2) + b2  # shape: (bs, 1, 1)

        return total_q.squeeze(-1).squeeze(-1)  # return shape: (bs,)


#  QMIX train function
MAX_STEPS_PER_EPISODE = 500

def train_qmix(env, n_agents, state_dim, obs_dim, action_dim, episodes=1000, batch_size=32, gamma=0.99):
    agent_nets = [AgentNetwork(obs_dim, action_dim).to(device) for _ in range(n_agents)]
    target_agent_nets = [AgentNetwork(obs_dim, action_dim).to(device) for _ in range(n_agents)]
    mixing_net = MixingNetwork(n_agents, state_dim).to(device)
    target_mixing_net = MixingNetwork(n_agents, state_dim).to(device)

    for i in range(n_agents):
        target_agent_nets[i].load_state_dict(agent_nets[i].state_dict())
    
    target_mixing_net.load_state_dict(mixing_net.state_dict())

    agent_optimizers = [optim.Adam(agent_nets[i].parameters(), lr=0.001) for i in range(n_agents)]
    mixing_optimizer = optim.Adam(mixing_net.parameters(), lr=0.001)

    replay_buffer = deque(maxlen=10000)  # buffer implemented as q

    # why queue? - consecutive states might be highly co related, old experiences are less efficient as time passes 

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    for episode in tqdm(range(episodes), desc="Training QMIX"):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            actions = []
            for i in range(n_agents):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
                if random.random() < epsilon:
                    action = random.randint(0, action_dim - 1)
                else:
                    q_values = agent_nets[i](obs_tensor)
                    action = q_values.argmax().item()
                actions.append(action)

            next_obs, reward, done, _ = env.step(actions)
            replay_buffer.append((obs, actions, reward, next_obs, done))
            obs = next_obs
            steps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = zip(*batch)

            obs_tensor = torch.FloatTensor(obs_batch).to(device)      # [batch, n_agents, obs_dim]
            actions_tensor = torch.LongTensor(actions_batch).to(device)
            rewards_tensor = torch.FloatTensor(rewards_batch).to(device)
            next_obs_tensor = torch.FloatTensor(next_obs_batch).to(device)
            dones_tensor = torch.FloatTensor(dones_batch).to(device)

            agent_qs, target_agent_qs = [], []

            for i in range(n_agents):
                q = agent_nets[i](obs_tensor[:, i, :])
                q = q.gather(1, actions_tensor[:, i].unsqueeze(1))
                agent_qs.append(q)

                with torch.no_grad():
                    target_q = target_agent_nets[i](next_obs_tensor[:, i, :])
                    target_q_max = target_q.max(dim=1, keepdim=True)[0]
                    target_agent_qs.append(target_q_max)

            agent_qs = torch.cat(agent_qs, dim=1)
            target_agent_qs = torch.cat(target_agent_qs, dim=1)

            state_tensor = obs_tensor.view(obs_tensor.size(0), -1)
            total_q = mixing_net(agent_qs, state_tensor)

            with torch.no_grad():
                next_state_tensor = next_obs_tensor.view(next_obs_tensor.size(0), -1)
                target_total_q = target_mixing_net(target_agent_qs, next_state_tensor)
                targets = rewards_tensor.unsqueeze(1) + gamma * target_total_q * (1 - dones_tensor.unsqueeze(1))

            loss = F.mse_loss(total_q, targets)
            mixing_optimizer.zero_grad()
            for opt in agent_optimizers:
                opt.zero_grad()
            loss.backward()
            mixing_optimizer.step()
            for opt in agent_optimizers:
                opt.step()

        if episode % 10 == 0:
            for i in range(n_agents):
                target_agent_nets[i].load_state_dict(agent_nets[i].state_dict())
            target_mixing_net.load_state_dict(mixing_net.state_dict())

    return agent_nets, mixing_net


# Evaluation function
# while evaluation, we use a greedy appraoch
def evaluate_qmix(env, agent_nets, mixing_net, max_steps=20000):
    n_agents = env.n_cores
    obs = env.reset()
    done = False
    steps = 0

    # adding max_steps is ensruing that infinite loop is avoided
    while not done and steps < max_steps:
        actions = []
        for i in range(n_agents):
            obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent_nets[i](obs_tensor)
            actions.append(q_values.argmax().item())
        obs, _, done, _ = env.step(actions)
        steps += 1

    if not done:
        print(f"[WARNING] Evaluation exceeded {max_steps} steps.")

    return env.finished_processes

# main function
def train_and_run_qmix(processes, retrain=False):
    env = MultiCoreSchedulingEnv(processes, n_cores=2)  # number of processor or agent
    n_agents = env.n_cores
    obs_shape = env.observation_space.shape
    
    obs_dim = obs_shape[1]
    action_dim = env.action_space.nvec[0]
    state_dim = obs_shape[0] * obs_shape[1]

    if not retrain and os.path.exists(MODEL_PATH):
        print("Evaluation using QMIX --- ")
        print(f"[INFO] Loading model from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        agent_nets = [net.to(device) for net in checkpoint["agent_nets"]]
        mixing_net = checkpoint["mixing_net"].to(device)
    else:
        print("[INFO] Training QMIX model...")
        agent_nets, mixing_net = train_qmix(
            env, n_agents, state_dim, obs_dim, action_dim,
            episodes=1000, batch_size=32, gamma=0.99
        )
        checkpoint = {
            "agent_nets": [net.cpu() for net in agent_nets],
            "mixing_net": mixing_net.cpu()
        }
        torch.save(checkpoint, MODEL_PATH)
        print(f"[INFO] QMIX model saved to {MODEL_PATH}")
        agent_nets = [net.to(device) for net in checkpoint["agent_nets"]]
        mixing_net = checkpoint["mixing_net"].to(device)

    # evaluation code 
    obs = env.reset()
    done = False
    steps = 0
    max_steps = 5000  # To prevent infinite loops

    # disables gradient calculation, during eval
    with torch.no_grad():
        while not done and steps < max_steps:
            actions = []
            for i in range(n_agents):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
                q_values = agent_nets[i](obs_tensor)
                action = q_values.argmax().item()
                actions.append(action)
            obs, _, done, _ = env.step(actions)
            steps += 1

    if not done:
        print(f"[WARNING] QMIX inference exceeded {max_steps} steps. Exiting early.")

    return env.finished_processes

