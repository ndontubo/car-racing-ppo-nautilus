# This file is modified from https://github.com/cjy1992/gym-carla.git:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see https://opensource.org/licenses/MIT.
#
# Modifications: replaces the SAC.load() call with SAC.learn() to train from scratch
# instead of loading a pretrained model.

import gymnasium as gym
import gym_carla
import carla
from stable_baselines3 import SAC

def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 1,
    'number_of_walkers': 0,
    'display_size': 256,
    'max_past_step': 1,
    'dt': 0.1,
    'discrete': False,
    'discrete_acc': [-3.0, 0.0, 3.0],
    'discrete_steer': [-0.2, 0.0, 0.2],
    'continuous_accel_range': [-3.0, 3.0],
    'continuous_steer_range': [-0.3, 0.3],
    'ego_vehicle_filter': 'vehicle.lincoln*',
    'port': 4000,
    'town': 'Town03',
    'max_time_episode': 1000,
    'max_waypt': 12,
    'obs_range': 32,
    'lidar_bin': 0.125,
    'd_behind': 12,
    'out_lane_thres': 2.0,
    'desired_speed': 8,
    'max_ego_spawn_times': 200,
    'display_route': False,
  }

  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)

  # Create SAC model
  model = SAC(
   "MlpPolicy",
   env, device="cpu",
   buffer_size=10000,
   batch_size=64,
   verbose=1,
   tensorboard_log="./tensorboard_DQN/")

  # Train from scratch
  model.learn(total_timesteps=20000, tb_log_name="sac_run")
  model.save("SAC_dist")

  # Evaluation loop
  obs, info = env.reset()
  while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
      obs, info = env.reset()

if __name__ == '__main__':
  main()
