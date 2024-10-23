import torch as th
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from CNN import CustomCnnTD3Policy
from RobotGym import RobotArmEnv

seed = 47
th.manual_seed(seed)

env = RobotArmEnv()

n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(mean=th.zeros(
    n_actions), sigma=0.1 * th.ones(n_actions))

model = TD3(
    policy=CustomCnnTD3Policy,
    env=env,
    action_noise=action_noise,
    verbose=1,
    seed=seed
)

model.learn(total_timesteps=100000)

model.save("td3_robot_arm_cnn")

obs, _ = env.reset(seed=seed)
done = False

env.render()

while not done:
    action, _states = model.predict(obs)

    obs, reward, done, truncated, info = env.step(action)

env.stop_render_thread()
env.close()
