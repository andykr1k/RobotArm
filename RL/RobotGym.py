from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Optional, Any, NamedTuple
import logging
import time
from dataclasses import dataclass
from MotorController import send_to_esp, close_connection
import requests
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ArmConfig:
    """Robot arm configuration parameters"""
    initial_angles: np.ndarray
    min_angles: np.ndarray
    max_angles: np.ndarray
    action_scale: float = 90.0
    sleep_time: float = 0.1

    def __post_init__(self):
        """Validate configuration"""
        if not (len(self.initial_angles) == len(self.min_angles) == len(self.max_angles)):
            raise ValueError("Angle arrays must have same length")
        if not all(mini <= init <= maxi for mini, init, maxi in
                   zip(self.min_angles, self.initial_angles, self.max_angles)):
            raise ValueError("Initial angles must be within limits")


class RewardConfig(NamedTuple):
    """Reward configuration parameters"""
    very_close_threshold: float = 75.0
    close_threshold: float = 150.0
    medium_threshold: float = 250.0
    very_close_reward: float = 5.0
    close_reward: float = 2.0
    medium_reward: float = 1.0
    far_reward: float = -1.0
    completion_reward: float = 10.0


class RobotArmEnv(gym.Env):
    """
    Optimized Robot Arm Environment for Reinforcement Learning
    Handles robot arm control and camera-based observations
    """
    DEFAULT_CONFIG = ArmConfig(
        initial_angles=np.array([90, 120, 0, 90, 0], dtype=np.float64),
        min_angles=np.array([0, 60, 0, 0, 0], dtype=np.float64),
        max_angles=np.array([180, 120, 45, 120, 90], dtype=np.float64)
    )

    def __init__(
        self,
        config: Optional[ArmConfig] = None,
        reward_config: Optional[RewardConfig] = None
    ):
        """Initialize environment with optional configurations"""
        super().__init__()

        self.config = config or self.DEFAULT_CONFIG
        self.reward_config = reward_config or RewardConfig()

        self._init_spaces()

        self._init_state()

    def _init_spaces(self) -> None:
        """Initialize action and observation spaces"""
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.config.initial_angles),),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(480, 640, 3),
            dtype=np.float32
        )

    def _init_state(self) -> None:
        """Initialize internal state variables"""
        self.arm_angles = self.config.initial_angles.copy()
        self.current_step = 0
        self.last_reward = 0.0
        self.done = False
        self._last_action_time = 0.0
        self._frame_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state with improved error handling"""
        super().reset(seed=seed)
        self.arm_angles = self.config.initial_angles.copy()
        self._send_command_with_retry()
        self._init_state()
        state = self._get_observation()
        return state, {}

    def _send_command_with_retry(self, max_retries: int = 3) -> None:
        """Send command to robot arm with retry logic"""
        for attempt in range(max_retries):
            try:
                command = (
                    '{' +
                    f'0:{self.arm_angles[0]},' +
                    f'3:{self.arm_angles[1]},' +
                    f'7:{180 - self.arm_angles[1]},' +
                    f'11:{self.arm_angles[2]},' +
                    f'13:{self.arm_angles[3]},' +
                    f'15:{self.arm_angles[4]}' +
                    '}'
                )
                send_to_esp(command)
                return
            except Exception as e:
                logger.error(f"Command send attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action with improved error handling and rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self._last_action_time
        if time_since_last < self.config.sleep_time:
            time.sleep(self.config.sleep_time - time_since_last)

        self.arm_angles = np.clip(
            self.arm_angles + action * self.config.action_scale,
            self.config.min_angles,
            self.config.max_angles
        )
        self._send_command_with_retry()
        self._last_action_time = time.time()

        state = self._get_observation()
        reward, done = self._calculate_step_results(state)

        self.current_step += 1
        self.last_reward = reward
        self.done = done

        info = {
            "TimeLimit.truncated": False,
            "done": done,
            "reward": reward,
            "arm_angles": self.arm_angles.tolist(),
            "step": self.current_step
        }

        logger.info(f"Step {self.current_step}, Reward: {reward}")

        return state, reward, done, False, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation from shared memory"""
        result = requests.get('http://100.69.34.11:5000/pic_feed_OD')
        return np.array(Image.open(BytesIO(result.content)))

    def _calculate_step_results(self, state: np.ndarray) -> Tuple[float, bool]:
        """Calculate rewards and done state"""
        data = requests.get('http://100.69.34.11:5000/info').json()
        reward = self._calculate_reward(data.get('distance'))
        done = data.get('on_blue')

        if done:
            reward += self.reward_config.completion_reward

        return reward, done

    def _calculate_reward(self, distance) -> float:
        """Calculate reward with improved distance-based logic"""
        if distance < self.reward_config.very_close_threshold:
            return self.reward_config.very_close_reward
        elif distance < self.reward_config.close_threshold:
            return self.reward_config.close_reward
        elif distance < self.reward_config.medium_threshold:
            return self.reward_config.medium_reward
        else:
            return self.reward_config.far_reward

    def close(self) -> None:
        """Close environment and processes gracefully"""
        close_connection()
        logger.info("Robot environment closed.")