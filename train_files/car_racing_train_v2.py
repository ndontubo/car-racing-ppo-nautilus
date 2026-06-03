"""
PPO training on CarRacing-v3 with explicit reward logging.
Uses Monitor wrapper to ensure episode rewards get logged regardless of vec env quirks.
"""
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

LOG_DIR = "/pvcvolume/tb_logs"
MODEL_DIR = "/pvcvolume/models"
MONITOR_DIR = "/pvcvolume/monitor_logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MONITOR_DIR, exist_ok=True)


class RewardLoggerCallback(BaseCallback):
    """Logs every completed episode's reward to TensorBoard."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check 'infos' for completed episodes (Monitor wrapper populates this)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_count += 1

                # Log raw episode reward
                self.logger.record("custom/ep_reward", float(ep_reward))
                self.logger.record("custom/ep_length", float(ep_length))
                self.logger.record("custom/episode_count", self.episode_count)

                # Rolling average of last 100 episodes
                window = self.episode_rewards[-100:]
                self.logger.record("custom/ep_reward_mean_100", float(np.mean(window)))

                if self.verbose:
                    print(f"[Episode {self.episode_count}] reward={ep_reward:.2f} length={ep_length}")
        return True


def make_env(seed=0):
    env = gym.make("CarRacing-v3", continuous=True)
    env = Monitor(env, filename=f"{MONITOR_DIR}/monitor_{seed}.csv")
    return env


# Vectorized env with Monitor wrapper (Monitor MUST be inside, before DummyVecEnv)
env = DummyVecEnv([lambda: make_env(0)])

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cuda",
    learning_rate=3e-4,
    n_steps=1024,           # Smaller rollouts -> more frequent logging
    batch_size=64,
    n_epochs=10,
)

reward_callback = RewardLoggerCallback(verbose=1)
checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=MODEL_DIR,
    name_prefix="ppo_car_racing",
)

TOTAL_TIMESTEPS = 300_000   # Slightly less than v1 since we're restarting

print(f"Training PPO on CarRacing-v3 for {TOTAL_TIMESTEPS} timesteps")
print(f"TensorBoard logs: {LOG_DIR}")
print(f"Monitor CSVs: {MONITOR_DIR}")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[reward_callback, checkpoint_callback],
    tb_log_name="ppo_run_v2",
)

model.save(f"{MODEL_DIR}/ppo_car_racing_v2_final")
print("Training complete.")
print(f"Total episodes completed: {reward_callback.episode_count}")
print(f"Final 100-episode mean reward: {np.mean(reward_callback.episode_rewards[-100:]):.2f}")