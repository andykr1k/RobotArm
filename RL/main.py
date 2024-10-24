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


def setup_environment(seed: int) -> Tuple[RobotArmEnv, TD3]:
    """Setup the environment and TD3 model"""
    try:
        env = RobotArmEnv()

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
            tensorboard_log="./td3_robot_arm_tensorboard/",
            device='cuda' if th.cuda.is_available() else 'cpu'
        )

        return env, model

    except Exception as e:
        print(f"Error during setup: {e}")
        sys.exit(1)


def train_model(
    model: TD3,
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

        # Train the model and log progress
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            log_interval=100
        )

        # Save the final model after training completes
        model.save("td3_robot_arm_final")

    except Exception as e:
        print(f"Error during training: {e}")
        model.save("td3_robot_arm_backup")
        raise


def evaluate_model(
    env: RobotArmEnv,
    model: TD3,
    seed: int,
    n_eval_episodes: int = 5
) -> None:
    """Evaluate the trained model"""
    try:
        for episode in range(n_eval_episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            episode_reward = 0
            step_count = 0
            rewards = []
            actions = []

            print(f"\nStarting evaluation episode {episode + 1}/{n_eval_episodes}")

            while not done:
                # Get action using deterministic policy
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                rewards.append(reward)
                actions.append(action)
                episode_reward += reward
                step_count += 1

                # Print progress every 10 steps
                if step_count % 10 == 0:
                    print(f"Step {step_count}: Reward = {reward:.3f}, "
                          f"Total Reward = {episode_reward:.3f}")

                # Timeout condition for the episode
                if step_count >= env.spec.max_episode_steps:
                    print("Episode timed out")
                    break

                # Small delay to prevent overwhelming the system
                time.sleep(0.01)

            print(
                f"Episode {episode + 1} finished with total reward: {episode_reward:.3f}")
            print(f"Actions taken: {actions}\nRewards: {rewards}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        # Ensure environment rendering thread is safely stopped
        env.stop_render_thread()


def main():
    """Main function to run the training and evaluation"""
    seed = 47
    th.manual_seed(seed)
    np.random.seed(seed)

    env = None

    try:
        # Setup environment and TD3 model
        print("Setting up environment and model...")
        env, model = setup_environment(seed)

        # Start training the model
        print("\nStarting training...")
        train_model(model, total_timesteps=100000)

        # Evaluate the model after training
        print("\nStarting evaluation...")
        evaluate_model(env, model, seed)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if env is not None:
            print("\nCleaning up...")
            env.close()


if __name__ == "__main__":
    main()
