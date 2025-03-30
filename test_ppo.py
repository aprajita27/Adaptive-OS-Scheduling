from stable_baselines3 import PPO
from env import CPUSchedulingEnv
from generate_processes import generate_dynamic_processes
from traditional_algorithms import calculate_metrics

processes = generate_dynamic_processes(n=100)
env = CPUSchedulingEnv(processes)
model = PPO.load("ppo_scheduler")

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)

print("\nPPO Results:")
calculate_metrics(env.finished)