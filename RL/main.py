import torch as th
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from RobotGym import RobotArmEnv
from CNN import CustomCnnPPOpolicy
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def setup_environment(seed: int):
    env = RobotArmEnv()

    logger.info(f"Observation Space: {env.observation_space}")
    logger.info(f"Observation Space Shape: {env.observation_space.shape}")
    logger.info(f"Observation Space Dtype: {env.observation_space.dtype}")

    sample_obs = env.reset()[0]
    logger.info(f"Sample Observation Shape: {sample_obs.shape}")
    logger.info(f"Sample Observation Dtype: {sample_obs.dtype}")

    env.reset(seed=seed)
    logger.info("Environment initialized.")

    env = DummyVecEnv([lambda: env])


    model = PPO(
        policy=CustomCnnPPOpolicy,
        env=env,
        verbose=1,
        seed=42,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/tensorboard/",
        device='cuda' if th.cuda.is_available() else 'cpu'
    )
    logger.info("PPO model initialized with CNN policy.")
    return env, model


def train_model(model, env, total_timesteps, checkpoint_freq=1000, callback=None):
    logger.info("Starting training...")

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints/",
        name_prefix="ppo_robot_arm"
    )
    logger.info(f"Checkpoints will be saved every {checkpoint_freq} steps.")

    if callback:
        combined_callback = CallbackList([checkpoint_callback, callback])
    else:
        combined_callback = checkpoint_callback

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=combined_callback,
            tb_log_name="PPO_Training"
        )
        logger.info("Training completed. Model saved.")
        model.save("ppo_robot_arm_final")
    except Exception as e:
        logger.error(f"Training failed: {e}")
    finally:
        env.close()


def main():
    seed = 46
    th.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}.")

    env = None
    try:
        logger.info("Setting up PPO environment and model...")
        env, model = setup_environment(seed)

        logger.info("Starting PPO training with custom logging callback...")
        train_model(
            model,
            env,
            total_timesteps=10000,
            checkpoint_freq=100,
        )

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
            logger.info("Environment closed.")


if __name__ == "__main__":
    main()
