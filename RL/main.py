import torch as th
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from CNN import CustomCnnTD3Policy
from RobotGym import RobotArmEnv
import sys
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tb_log = './logs/tensorboard'


def setup_environment(seed: int) -> Tuple[RobotArmEnv, TD3]:
    """Setup the environment and TD3 model"""
    try:
        env = RobotArmEnv()

        env.reset(seed=seed)

        n_actions = env.action_space.shape[-1]

        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        model = TD3(
            policy=CustomCnnTD3Policy,
            env=env,
            action_noise=action_noise,
            verbose=1,
            seed=seed,
            learning_rate=3e-4,
            buffer_size=20000,
            learning_starts=1000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log=tb_log,
            device='cuda' if th.cuda.is_available() else 'cpu'
        )

        return env, model

    except Exception as e:
        logger.error(f"Error during setup: {e}")
        sys.exit(1)


def train_model(
    model: TD3,
    env: RobotArmEnv,
    total_timesteps: int,
    checkpoint_freq: int = 10000
) -> None:
    """Train the model with checkpointing"""
    try:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path="./checkpoints/",
            name_prefix="td3_robot_arm"
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            log_interval=100,
            tb_log_name="train"
        )

        model.save("td3_robot_arm_final")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        model.save("td3_robot_arm_backup")
        raise
    finally:
        env.close()


def main():
    """Main function to run the training and evaluation"""
    seed = 46
    th.manual_seed(seed)
    np.random.seed(seed)

    env = None

    try:
        logger.info("Setting up environment and model...")
        env, model = setup_environment(seed)

        logger.info("\nStarting training...")
        train_model(model, env, total_timesteps=100000)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nAn error occurred: {e}")
    finally:
        if env is not None:
            logger.info("\nCleaning up...")
            env.close()


if __name__ == "__main__":
    main()
