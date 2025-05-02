
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from multi_core_env import MultiCoreSchedulingEnv
from custom_policy import CentralizedCriticExtractor
import torch as th
import os
import numpy as np

MODEL_PATH = "mappo_cpu_scheduler.zip"

class CTDEPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CTDEPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CentralizedCriticExtractor,
            features_extractor_kwargs={},
            **kwargs,
        )
        self._build(lr_schedule)

def train_and_run_mappo(processes, retrain=False, n_cores=4):
    def make_env():
        return MultiCoreSchedulingEnv(processes, n_cores=n_cores)

    env = DummyVecEnv([make_env])

    if retrain or not os.path.exists(MODEL_PATH):
        model = PPO(
            CTDEPolicy,
            env,
            verbose=1,
            learning_rate=1e-4,
            gamma=0.99,
            n_steps=1024,
            batch_size=128,
            n_epochs=15,
            ent_coef=0.01,
            clip_range=0.2,
            gae_lambda=0.95,
            tensorboard_log="./mappo_logs"
        )
        model.learn(total_timesteps=30000)
        model.save(MODEL_PATH)
        print(f"[INFO] MAPPO model saved to {MODEL_PATH}")
    else:
        print(f"[INFO] Loading MAPPO model from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env, policy=CTDEPolicy)

    obs = env.reset()
    done = False
    steps = 0
    max_steps = 20000

    print(f"[DEBUG] Initial observation shape: {obs.shape}")
    
    while not done and steps < max_steps:
        actions, _ = model.predict(obs)
        obs, _, done, _ = env.step(actions)
        steps += 1

    if not done:
        print(f"[WARNING] MAPPO inference exceeded {max_steps} steps. Exiting early.")

    return env.get_attr("finished_processes")[0]
