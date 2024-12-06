import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from RobotGym import RobotArmEnv

env = DummyVecEnv([lambda: RobotArmEnv()])

model = PPO("CnnPolicy", env, verbose=1,
            tensorboard_log="./ppo_robot_tensorboard/")

model.learn(total_timesteps=10000)

model.save("sim_ppo_robot_arm")

env.close()
