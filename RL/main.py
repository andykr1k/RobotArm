import torch as th
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from CNN import CustomCnnTD3Policy
from RobotGym import RobotArmEnv
import time
import sys
from typing import Tuple
import multiprocessing as mp
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tb_log = './logs/tensorboard'


def setup_environment(seed: int) -> Tuple[RobotArmEnv, TD3]:
    """Setup the environment and TD3 model"""
    try:
        # Set render mode for visualization
        env = RobotArmEnv(render_mode="human")

        # Reset with seed for reproducibility
        env.reset(seed=seed)

        # Get number of actions in action space
        n_actions = env.action_space.shape[-1]

        # Setup action noise for exploration
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        # Create the TD3 model
        model = TD3(
            policy=CustomCnnTD3Policy,
            env=env,
            action_noise=action_noise,
            verbose=1,
            seed=seed,
            learning_rate=3e-4,
            buffer_size=100000,
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


def render_process(env):
    """Separate process to handle environment rendering"""
    try:
        while True:
            env.render()
            time.sleep(1.0 / env.metadata['render_fps'])
    except Exception as e:
        logger.error(f"Error in rendering process: {e}")
    finally:
        env.close()


def train_model(
    model: TD3,
    env: RobotArmEnv,
    total_timesteps: int,
    checkpoint_freq: int = 10000
) -> None:
    """Train the model with checkpointing"""
    try:
        # Setup checkpointing every `checkpoint_freq` timesteps
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path="./checkpoints/",
            name_prefix="td3_robot_arm"
        )

        # Create separate process for rendering
        render_proc = mp.Process(target=render_process, args=(env,))
        render_proc.start()

        # Train the model and log progress
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            log_interval=100,
            tb_log_name="train"
        )

        # Save the final model after training completes
        model.save("td3_robot_arm_final")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        model.save("td3_robot_arm_backup")
        raise
    finally:
        # Ensure environment rendering thread is stopped gracefully
        render_proc.terminate()
        render_proc.join()
        env.close()


def main():
    """Main function to run the training and evaluation"""
    seed = 46
    th.manual_seed(seed)
    np.random.seed(seed)

    env = None

    try:
        # Setup environment and TD3 model
        logger.info("Setting up environment and model...")
        env, model = setup_environment(seed)

        # Start training the model
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
