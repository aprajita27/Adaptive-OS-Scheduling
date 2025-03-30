from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import CPUSchedulingEnv
import os

MODEL_PATH = "ppo_cpu_scheduler.zip"

def train_and_run_ppo(processes, retrain=False):
    def make_env():
        return CPUSchedulingEnv(processes)

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
            tensorboard_log="./ppo_logs"
        )
        model.learn(total_timesteps=30000)
        model.save(MODEL_PATH)
        env.save("vec_normalize.pkl")
        print(f"[INFO] PPO model saved to {MODEL_PATH}")
    else:
        print(f"[INFO] Loading PPO model from {MODEL_PATH}")
        env = DummyVecEnv([make_env])
        env = VecNormalize.load("vec_normalize.pkl", env)
        model = PPO.load(MODEL_PATH, env=env)

    obs = env.reset()
    done = False
    steps = 0
    max_steps = 20000

    print(f"[DEBUG] Initial observation shape: {obs.shape}")
    print(f"[DEBUG] Initial ready queue: {env.get_attr('ready_queue')[0]}")

    while not done and steps < max_steps:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        steps += 1

    if not done:
        print(f"[WARNING] PPO inference exceeded {max_steps} steps. Exiting early.")

    return env.get_attr("finished_processes")[0]
