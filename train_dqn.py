import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim  # adam
import random
from collections import deque
from env import CPUSchedulingEnv
import gym
from tqdm import tqdm  # Progress bar

# Q-network definition
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # dense
            nn.ReLU(),  # non linear
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)  # forward pass

def train_and_run_dqn(all_processes, retrain=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use gpu on colab 
    print(f"Using device: {device}")

    env = CPUSchedulingEnv(all_processes)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n  # discrete actions that the agent can take

    q_net = DQN(obs_size, n_actions).to(device)  # instance of q network
    target_net = DQN(obs_size, n_actions).to(device)  # stable target values
    target_net.load_state_dict(q_net.state_dict())  # initialise with same weights

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    replay_buffer = deque(maxlen=5000)  # experience replay
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.995
    episodes = 5

    if retrain:
        for ep in tqdm(range(episodes), desc="Training DQN"):
            obs = env.reset()
            total_reward = 0

            # if done is true, all processes have been scheduled and completed
            done = False  # if done is true, means episode complete

            step_count = 0   
            max_steps = 1000  # safe limit

            while not done and step_count < max_steps:
                # random action exploration
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)  # Add batch dimension
                        action = q_net(obs_tensor).argmax().item()  # choose best action

                next_obs, reward, done, _ = env.step(action)  # go to next state
                step_count += 1   # this line

                replay_buffer.append((obs, action, reward, next_obs, done))  # add to replay buffer
                obs = next_obs
                total_reward += reward  # add current reward to total reward 

                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    # convert to tensors and move to device
                    states = torch.FloatTensor(np.array(states)).to(device)
                    actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
                    rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
                    next_states = torch.FloatTensor(np.array(next_states)).to(device)
                    dones = torch.BoolTensor(np.array(dones)).unsqueeze(1).to(device)

                    q_values = q_net(states).gather(1, actions)  # predict q-values for all possible actions for each state in the batch
                    with torch.no_grad():
                        max_next_q = target_net(next_states).max(1, keepdim=True)[0]
                        target_q = rewards + gamma * max_next_q * (~dones)  # bit wsie 'not', ensure that we dont propagate future rewards (the next Q-values) if the episode has finished

                    loss = nn.MSELoss()(q_values, target_q)  # mean sq err
                    optimizer.zero_grad()
                    loss.backward()  # backprop
                    optimizer.step()

                if step_count % 100 == 0:
                    print(f"Step {step_count}: Epsilon = {epsilon:.3f}, Buffer size = {len(replay_buffer)}")


            if ep % 10 == 0:
                target_net.load_state_dict(q_net.state_dict())
                print(f"Episode {ep}: Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # save model
        torch.save(q_net.state_dict(), "dqn_model.pth")
        #torch.save(q_net.state_dict(), "/content/drive/MyDrive/CSCI566-S25-Material/DL Project/Adaptive-OS-Scheduling/dqn_model.pth")

    else:
        q_net.load_state_dict(torch.load("dqn_model.pth", map_location=device))
        #q_net.load_state_dict(torch.load("/content/drive/MyDrive/CSCI566-S25-Material/DL Project/Adaptive-OS-Scheduling/dqn_model.pth", map_location=device))

    # Evaluation
    print("--- hello entering evalution --- ")
    eval_env = CPUSchedulingEnv(all_processes)
    obs = eval_env.reset()
    done = False

    step_count = 0
    max_eval_steps = 2000  # <-- Set a limit for evaluation too

    while not done and step_count < max_eval_steps:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = q_net(obs_tensor).argmax().item()
        obs, _, done, _ = eval_env.step(action)
        step_count += 1  # important

    if not done:
        print("--- [Warning] Evaluation hit max_eval_steps without completing. --- ")

    return eval_env.finished_processes


