from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from multi_core_env import MultiCoreSchedulingEnv
import numpy as np
import os

MODEL_PATH = "mappo_cpu_scheduler.zip"

def train_and_run_mappo(processes, retrain=False, n_cores=2):
    """
    Train or run a Multi-Agent PPO scheduler with n CPU cores.

    Args:
        processes: List of process dictionaries.
        retrain: Bool, whether to retrain from scratch.
        n_cores: Number of CPU cores (agents).
    
    Returns:
        List of finished processes after scheduling.
    """
    def make_env():
        return MultiCoreSchedulingEnv(processes, n_cores=n_cores)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    if retrain or not os.path.exists(MODEL_PATH):
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            gamma=0.99,
            n_steps=1024,
            batch_size=128,
            n_epochs=15,
            ent_coef=0.005,
            clip_range=0.2,
            gae_lambda=0.95,
            tensorboard_log="./mappo_logs"
        )
        model.learn(total_timesteps=30000)
        model.save(MODEL_PATH)
        env.save("vec_normalize_mappo.pkl")
        print(f"[INFO] MAPPO model saved to {MODEL_PATH}")
    else:
        print(f"[INFO] Loading MAPPO model from {MODEL_PATH}")
        env = DummyVecEnv([make_env])
        env = VecNormalize.load("vec_normalize_mappo.pkl", env)
        model = PPO.load(MODEL_PATH, env=env)

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
