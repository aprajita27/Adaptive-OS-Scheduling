# RL-Based Adaptive CPU Scheduling

This project explores Reinforcement Learning (RL) for adaptive CPU process scheduling. It implements a single-core scheduling simulator with both traditional algorithms (FCFS, SJF, Priority Preemptive, Round Robin) and a learning-based scheduler using Proximal Policy Optimization (PPO).

## Features
- Custom Gym-compatible environment for CPU scheduling (`CPUSchedulingEnv`)
- PPO-based RL agent trained using Stable-Baselines3
- Evaluation against traditional scheduling algorithms
- Metric tracking: Waiting Time, Turnaround Time, Response Time, Throughput
- TensorBoard logs for analysis

## Requirements
- Python 3.10+
- `stable-baselines3`, `gym`, `matplotlib`, `numpy`, `torch`

Install dependencies and run:
```bash
pip install -r requirements.txt
python main.py




