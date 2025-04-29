# deep q network

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim  # optimiser
import random
import matplotlib.pyplot as plt
from collections import deque  # queue
import time  # to see the training time of dqn
import copy

# random seed values
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("-----------------------------------------------------------")
print("This is the device being used: ", device)

class JobSchedulingEnvironment:
    def __init__(self, processes, num_cpus=4):
        self.original_processes = copy.deepcopy(processes)
        self.num_cpus = num_cpus
        self.reset()
    
    def reset(self):
        # Create a deep copy of the processes list
        self.processes = copy.deepcopy(self.original_processes)
        for p in self.processes:
            p["start_time"] = None
            p["finish_time"] = None
        
        self.current_time = 0
        self.completed_processes = []
        self.running_processes = []
        self.waiting_processes = []
        self.next_process_idx = 0
        
        # Add initial processes to waiting queue
        while (self.next_process_idx < len(self.processes) and 
               self.processes[self.next_process_idx]["arrival_time"] <= self.current_time):
            self.waiting_processes.append(self.processes[self.next_process_idx])
            self.next_process_idx += 1
        
        # State includes time, CPU usage, waiting processes features
        state = self._get_state()
        return state
    
    def _get_state(self):
        # Calculate current CPU usage (percentage)
        cpu_usage = len(self.running_processes) / self.num_cpus
        
        # Get features of the top 5 waiting processes (or pad with zeros)
        waiting_features = []
        for i in range(min(5, len(self.waiting_processes))):
            proc = self.waiting_processes[i]
            # Normalized features
            waiting_features.extend([
                proc["burst_time"] / 10,  # Normalize burst time
                proc["priority"] / 100,   # Normalize priority
                proc["memory"] / 4096,    # Normalize memory
                proc["cpu_req"] / 100     # Normalize CPU requirement
            ])
        
        # Pad with zeros if there are fewer than 5 waiting processes
        waiting_features.extend([0] * (5 - min(5, len(self.waiting_processes))) * 4)
        
        # Combine features
        state = [
            self.current_time / 100,  # Normalize time
            cpu_usage,
            len(self.waiting_processes) / 20,  # Normalize waiting queue length
        ] + waiting_features
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        # Action represents which waiting process to schedule
        # Action 0: Do nothing
        # Action 1-5: Schedule the nth process in the waiting queue
        
        reward = 0
        done = False
        
        # Check if we can schedule a new process
        if action > 0 and action <= len(self.waiting_processes) and len(self.running_processes) < self.num_cpus:
            selected_process = self.waiting_processes[action - 1]
            
            # Remove from waiting and add to running
            self.waiting_processes.pop(action - 1)
            self.running_processes.append(selected_process)
            
            # Set start time
            selected_process["start_time"] = self.current_time
            selected_process["finish_time"] = self.current_time + selected_process["burst_time"]
            
            # Reward for scheduling high priority processes sooner
            waiting_time = self.current_time - selected_process["arrival_time"]
            if waiting_time == 0:
                reward += 2  # Bonus for immediate scheduling
            
            # Prioritize high priority jobs
            reward += selected_process["priority"] / 100
        
        # Advance time to next event (process completion or new arrival)
        next_event_time = float('inf')
        
        # Check for next process completion
        if self.running_processes:
            next_completion_time = min(p["finish_time"] for p in self.running_processes)
            next_event_time = min(next_event_time, next_completion_time)
        
        # Check for next process arrival
        if self.next_process_idx < len(self.processes):
            next_arrival_time = self.processes[self.next_process_idx]["arrival_time"]
            next_event_time = min(next_event_time, next_arrival_time)
        
        # If no more events, end simulation
        if next_event_time == float('inf'):
            next_event_time = self.current_time + 1
            done = True
        
        # Update time
        time_delta = next_event_time - self.current_time
        self.current_time = next_event_time
        
        # Process completions
        completed = []
        for proc in self.running_processes:
            if proc["finish_time"] <= self.current_time:
                completed.append(proc)
                
                # Calculate turnaround time and reward
                turnaround_time = proc["finish_time"] - proc["arrival_time"]
                response_time = proc["start_time"] - proc["arrival_time"]
                
                # Reward for minimizing turnaround and response time
                reward += 5 * (10 / turnaround_time) * (proc["priority"] / 100)
                reward += 2 * (1 / (response_time + 1)) * (proc["priority"] / 100)
        
        # Remove completed processes
        for proc in completed:
            self.running_processes.remove(proc)
            self.completed_processes.append(proc)
        
        # Add new arrivals to waiting queue
        while (self.next_process_idx < len(self.processes) and 
               self.processes[self.next_process_idx]["arrival_time"] <= self.current_time):
            self.waiting_processes.append(self.processes[self.next_process_idx])
            self.next_process_idx += 1
        
        # If waiting queue is empty and all processes are complete, we're done
        if not self.waiting_processes and not self.running_processes and self.next_process_idx >= len(self.processes):
            done = True
            
            # Final reward based on overall performance
            if self.completed_processes:
                avg_turnaround = np.mean([p["finish_time"] - p["arrival_time"] for p in self.completed_processes])
                avg_response = np.mean([p["start_time"] - p["arrival_time"] for p in self.completed_processes])
                reward += 100 / (avg_turnaround + 1) + 50 / (avg_response + 1)
        
        # Small penalty for time passing to encourage quick scheduling
        reward -= 0.1 * time_delta
        
        # Get next state
        next_state = self._get_state()
        
        # Penalty for idle CPUs
        cpu_idle_rate = 1 - len(self.running_processes) / self.num_cpus
        reward -= 0.5 * cpu_idle_rate * time_delta
        
        return next_state, reward, done, {}
    
    def get_all_processes(self):
        # Return all processes for evaluation
        return self.processes


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Q Network
        self.policy_net = DQN(state_size, action_size).to(device)
        
        # Target Network
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state, train=True):
        if train and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute V(s_{t+1}) for all next states
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1).detach()
        
        # Compute the expected Q values
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = self.criterion(current_q_values, expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train_dqn_agent(processes, episodes=500, retrain=True):
    # Environment setup
    env = JobSchedulingEnvironment(processes, num_cpus=4)
    state_size = 23  # Time, CPU usage, waiting queue length, and 5 processes with 4 features each
    action_size = 6  # Do nothing, or schedule 1st through 5th waiting process
    
    # DQN agent
    agent = DQNAgent(state_size, action_size)
    model_path = 'dqn_scheduler.pth'
    
    # Check if we should retrain or load an existing model
    if not retrain:
        try:
            agent.load(model_path)
            print("Loaded existing DQN model.")
            return run_dqn(agent, processes)
        except:
            print("No existing model found. Training new model.")
            retrain = True
    
    if retrain:
        # Training parameters
        batch_size = 32
        target_update_freq = 10
        
        # Training metrics
        episode_rewards = []
        turnaround_times = []
        waiting_times = []
        losses = []
        
        print("Training DQN agent...")
        start_time = time.time()
        
        # Training loop
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            episode_loss = 0
            steps = 0
            
            while not done:
                # Choose action
                action = agent.act(state)
                
                # Take action
                next_state, reward, done, _ = env.step(action)
                
                # Store in replay memory
                agent.memorize(state, action, reward, next_state, done)
                
                # Move to next state
                state = next_state
                total_reward += reward
                
                # Train with experience replay
                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)
                    episode_loss += loss
                    steps += 1
            
            # Update target network periodically
            if e % target_update_freq == 0:
                agent.update_target_net()
            
            # Track metrics
            processes = env.get_all_processes()
            valid_processes = [p for p in processes if p.get("start_time") is not None and p.get("finish_time") is not None]
            
            if valid_processes:
                avg_turnaround = sum(p["finish_time"] - p["arrival_time"] for p in valid_processes) / len(valid_processes)
                avg_waiting = sum(p["start_time"] - p["arrival_time"] for p in valid_processes) / len(valid_processes)
                
                episode_rewards.append(total_reward)
                turnaround_times.append(avg_turnaround)
                waiting_times.append(avg_waiting)
                losses.append(episode_loss / steps if steps > 0 else 0)
            
            # Print progress
            if (e + 1) % 50 == 0:
                print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, "
                      f"Avg Turnaround: {avg_turnaround:.2f}, "
                      f"Epsilon: {agent.epsilon:.2f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the trained model
        agent.save(model_path)
        print(f"Model saved as '{model_path}'")
        
        # Plot training progress
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(2, 2, 2)
        plt.plot(turnaround_times)
        plt.title('Average Turnaround Time')
        plt.xlabel('Episode')
        plt.ylabel('Time')
        
        plt.subplot(2, 2, 3)
        plt.plot(waiting_times)
        plt.title('Average Waiting Time')
        plt.xlabel('Episode')
        plt.ylabel('Time')
        
        plt.subplot(2, 2, 4)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('dqn_training_progress.png')
        
    return run_dqn(agent, processes)


def run_dqn(agent, processes):
    # Run the trained agent on the provided processes
    env = JobSchedulingEnvironment(processes, num_cpus=4)
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state, train=False)  # No exploration during evaluation
        next_state, _, done, _ = env.step(action)
        state = next_state
    
    # Return all processes (including those that weren't scheduled)
    return env.get_all_processes()


def train_and_run_dqn(processes, retrain=False, episodes=500):
    """Function to train DQN agent and run it on the given processes"""
    return train_dqn_agent(processes, episodes=episodes, retrain=retrain)