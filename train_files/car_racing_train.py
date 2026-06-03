"""
PPO training on CarRacing-v3 with TensorBoard logging.
Saves model checkpoints and tensorboard logs to /pvcvolume.
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Paths on the PVC
LOG_DIR = "/pvcvolume/tb_logs"
MODEL_DIR = "/pvcvolume/models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Create the environment
def make_env():
    return gym.make("CarRacing-v3", render_mode="rgb_array")

env = DummyVecEnv([make_env])

# PPO with CNN policy (because observations are images)
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cuda",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

# Periodically save model checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=MODEL_DIR,
    name_prefix="ppo_car_racing"
)

# Train
TOTAL_TIMESTEPS = 500_000
print(f"Training PPO on CarRacing for {TOTAL_TIMESTEPS} timesteps")
print(f"TensorBoard logs: {LOG_DIR}")
print(f"Model checkpoints: {MODEL_DIR}")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    tb_log_name="ppo_run",
    progress_bar=True,
)

# Final save
model.save(f"{MODEL_DIR}/ppo_car_racing_final")
print(f"Training complete. Final model: {MODEL_DIR}/ppo_car_racing_final.zip")