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
    initial_angles: np.ndarray
    min_angles: np.ndarray
    max_angles: np.ndarray
    action_scale: float = 90.0
    sleep_time: float = 0.1

    def __post_init__(self):
        if not (len(self.initial_angles) == len(self.min_angles) == len(self.max_angles)):
            raise ValueError("Angle arrays must have same length")
        if not all(mini <= init <= maxi for mini, init, maxi in
                   zip(self.min_angles, self.initial_angles, self.max_angles)):
            raise ValueError("Initial angles must be within limits")


class RewardConfig(NamedTuple):
    gripper_completion_threshold: float = 25.0
    gripper_very_close_threshold: float = 40.0
    gripper_close_threshold: float = 80.0
    gripper_medium_threshold: float = 120.0
    gripper_far_threshold: float = 200.0
    gripper_very_close_reward: float = 10.0
    gripper_close_reward: float = 5.0
    gripper_medium_reward: float = -1.0
    gripper_far_reward: float = -3.0
    gripper_completion_reward: float = 15.0

    very_close_threshold: float = 60.0
    close_threshold: float = 100.0
    medium_threshold: float = 150.0
    far_threshold: float = 200.0
    very_close_reward: float = 10.0
    close_reward: float = 5.0
    medium_reward: float = -1.0
    far_reward: float = -5.0
    completion_reward: float = 50.0



class RobotArmEnv(gym.Env):
    DEFAULT_CONFIG = ArmConfig(
        initial_angles=np.array([90, 90, 90, 90, 90, 90], dtype=np.float64),
        min_angles=np.array([0, 0, 0, 0, 0, 20], dtype=np.float64),
        max_angles=np.array([180, 180, 180, 180, 180, 160], dtype=np.float64)
    )

    def __init__(
        self,
        config: Optional[ArmConfig] = None,
        reward_config: Optional[RewardConfig] = None
    ):
        super().__init__()

        self.config = config or self.DEFAULT_CONFIG
        self.reward_config = reward_config or RewardConfig()

        self._init_spaces()

        self._init_state()

    def _init_spaces(self) -> None:
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.config.initial_angles),),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(480, 640, 3),
            dtype=np.uint8
        )

    def _init_state(self) -> None:
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
        super().reset(seed=seed)
        self.arm_angles = self.config.initial_angles.copy()
        self._send_command_with_retry()
        self._init_state()
        state = self._get_observation()
        return state, {}

    def _send_command_with_retry(self, max_retries: int = 3) -> None:
        for attempt in range(max_retries):
            try:
                command = (
                    '{' +
                    f'0:{int(self.arm_angles[0])},' +
                    f'4:{int(self.arm_angles[1])},' +
                    f'7:{int(self.arm_angles[2])},' +
                    f'8:{int(self.arm_angles[3])},' +
                    f'11:{int(self.arm_angles[4])},' +
                    f'15:{int(self.arm_angles[5])}' +
                    '}'
                )
                send_to_esp(command)
                return
            except Exception as e:
                wait_time = 2 ** attempt
                logger.error(f"Command send attempt {attempt + 1} failed: {e}, retrying in {wait_time} seconds")
                time.sleep(wait_time)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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

        done = bool(done)

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

        logger.info(
            f"Step: {self.current_step}, Reward: {reward:.2f}, Done: {done}, "
            f"Angles: {self.arm_angles.tolist()}"
        )

        # Modify the return to match Gym v26 API
        return state, reward, done, False, info

    def _get_observation(self) -> np.ndarray:
        try:
            result = requests.get(
                'http://100.69.34.11:5000/pic_feed_OD', timeout=5)

            if result.status_code != 200:
                logger.error(f"Failed to fetch image. Status code: {result.status_code}")
                return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

            image = Image.open(BytesIO(result.content))

            # Convert to numpy array in channel-last format
            image_array = np.array(image)

            return image_array
        except requests.RequestException as e:
            logger.error(f"Failed to fetch observation: {e}")
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

    def _calculate_step_results(self, state: np.ndarray) -> Tuple[float, bool]:
        try:
            data = requests.get('http://100.69.34.11:5000/info').json()
            reward = self._calculate_reward(
                data.get('distance_from_pink_to_blue'), data.get('right_gripper_distance'), data.get('left_gripper_distance'), data.get('on_blue'))

            on_blue = data.get('on_blue')

            done = bool(on_blue)

            if done:
                reward += self.reward_config.completion_reward

            return reward, done

        except Exception as e:
            logger.error(f"Error in _calculate_step_results: {e}")
            return 0.0, False


    def _calculate_reward(self, distance: float, right_gripper: float, left_gripper: float) -> float:
        reward = 0.0

        if distance < self.reward_config.very_close_threshold:
            distance_reward = self.reward_config.very_close_reward
        elif distance < self.reward_config.close_threshold:
            distance_reward = self.reward_config.close_reward
        elif distance < self.reward_config.medium_threshold:
            distance_reward = self.reward_config.medium_reward
        else:
            distance_reward = self.reward_config.far_reward

        gripper_distance = max(right_gripper, left_gripper)
        if gripper_distance < self.reward_config.gripper_completion_threshold:
            gripper_reward = self.reward_config.gripper_completion_reward
        if gripper_distance < self.reward_config.gripper_very_close_threshold:
            gripper_reward = self.reward_config.gripper_very_close_reward
        elif gripper_distance < self.reward_config.gripper_close_threshold:
            gripper_reward = self.reward_config.gripper_close_reward
        elif gripper_distance < self.reward_config.gripper_medium_threshold:
            gripper_reward = self.reward_config.gripper_medium_reward
        else:
            gripper_reward = self.reward_config.gripper_far_reward

        reward += distance_reward + gripper_reward

        logger.debug(
            f"Distance: {distance}, Distance Reward: {distance_reward}, Right Gripper: {right_gripper}, Left Gripper: {left_gripper}, Gripper Reward: {gripper_reward}, Total Reward: {reward}"
        )
        return reward

    def close(self) -> None:
        """Close environment and processes gracefully"""
        close_connection()
        logger.info("Robot environment closed.")
