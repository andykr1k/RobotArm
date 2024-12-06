import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import cv2
from typing import Tuple, Dict, Optional, Any, NamedTuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardConfig(NamedTuple):
    max_reward_distance: float = 700.0
    min_reward_distance: float = 25.0
    max_distance_reward: float = 10.0
    min_distance_reward: float = -10.0

    max_gripper_distance: float = 700.0
    min_gripper_distance: float = 25.0
    max_gripper_reward: float = 10.0
    min_gripper_reward: float = -10.0


def scale_reward(value: float, max_value: float, min_value: float, max_reward: float, min_reward: float) -> float:
    if value <= min_value:
        return max_reward
    elif value >= max_value:
        return min_reward
    result = ((max_value - value) / (max_value - min_value)) * \
        (max_reward - min_reward) + min_reward
    return round(result, 3)

class RobotArmEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            "simulation/kuka_experimental/kuka_lbr_iiwa_support/urdf/lbr_iiwa_14_r820.urdf",
            [0, 0, 0],
            useFixedBase=1,
        )
        p.setGravity(0, 0, -9.81)

        self.num_joints = p.getNumJoints(self.robot_id)
        self.target_positions = np.zeros(self.num_joints)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
        )

        self.reward_config = RewardConfig()
        self.current_step = 0
        self.done = False

    def reset(self, seed: Optional[int] = 42, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
            p.setSeed(seed)

        self.current_step = 0
        self.done = False
        self.target_positions = np.zeros(self.num_joints)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            "simulation/kuka_experimental/kuka_lbr_iiwa_support/urdf/lbr_iiwa_14_r820.urdf",
            [0, 0, 0],
            useFixedBase=1,
        )
        return self._get_observation(), {}


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.target_positions += action
        self.target_positions = np.clip(self.target_positions, -np.pi, np.pi)

        p.setJointMotorControlArray(
            self.robot_id,
            range(self.num_joints),
            p.POSITION_CONTROL,
            targetPositions=self.target_positions.tolist(),
        )
        p.stepSimulation()

        obs = self._get_observation()
        reward, done = self._calculate_step_results(obs)
        self.current_step += 1

        return obs, reward, done, False, {}

    def _get_observation(self) -> np.ndarray:
        _, _, rgb_img, _, _ = p.getCameraImage(
            width=640,
            height=480,
            viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.5],
                distance=2,
                yaw=210,
                pitch=-25,
                roll=0,
                upAxisIndex=2,
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(
                fov=60, aspect=640 / 480, nearVal=0.1, farVal=100
            ),
        )

        bgr_img = np.array(rgb_img, dtype=np.uint8)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return np.array(rgb_img, dtype=np.uint8)
    
    def _calculate_step_results(self, frame: np.ndarray) -> Tuple[float, bool]:
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([75, 255, 255])

        lower_pink = np.array([120, 50, 190])
        upper_pink = np.array([160, 90, 240])

        lower_blue = np.array([90, 130, 128])
        upper_blue = np.array([120, 255, 255])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        pink_center = self.detect_largest_object(frame, mask_pink, (1, 0, 1))
        blue_center = self.detect_largest_object(frame, mask_blue, (1, 0, 0))
        green_center = self.detect_largest_object(frame, mask_green, (0, 1, 0))

        distance_from_pink_to_blue, right_gripper_distance, left_gripper_distance, on_blue = 0, 0, 0, 0
        if pink_center and blue_center and green_center:
            distance_from_pink_to_blue = int(np.linalg.norm(
                np.array(pink_center) - np.array(blue_center)))
            right_gripper_distance = int(np.linalg.norm(
                np.array(pink_center) - np.array(green_center)))

            on_blue = (abs(pink_center[0] - blue_center[0]) <
                       38 and abs(pink_center[1] - blue_center[1]) < 38)

        reward = self._calculate_reward(
            distance_from_pink_to_blue, right_gripper_distance, left_gripper_distance)

        done = bool(on_blue)

        if done:
            reward += self.reward_config.completion_reward

        return reward, done
    
    def detect_largest_object(self, frame, mask, grid_color):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No object detected in the specified mask.")
            return None

        largest_contour = max(contours, key=cv2.contourArea, default=None)

        if largest_contour is not None:
            area = cv2.contourArea(largest_contour)
            if area > 50:
                self.draw_contour_in_pybullet(largest_contour, grid_color)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    self.display_center_in_pybullet(center_x, center_y)
                    return (center_x, center_y)
        return None
    
    def draw_contour_in_pybullet(self, contour, grid_color):
        for i in range(len(contour)):
            start_point = tuple(contour[i][0])
            end_point = tuple(contour[(i + 1) % len(contour)][0])
            p.addUserDebugLine(start_point, end_point,
                               lineColorRGB=grid_color, lineWidth=2)

    def display_center_in_pybullet(self, x, y):
        p.addUserDebugText(f"Center: ({x}, {y})", textPosition=(
            x / 640.0, y / 480.0, 0.1), textColorRGB=(1, 0, 0), lifeTime=0.1)

    def _calculate_reward(self, distance: float, right_gripper: float, left_gripper: float) -> float:
        distance_reward = scale_reward(
            distance,
            self.reward_config.max_reward_distance,
            self.reward_config.min_reward_distance,
            self.reward_config.max_distance_reward,
            self.reward_config.min_distance_reward
        )

        # gripper_distance = max(right_gripper, left_gripper)
        # gripper_reward = scale_reward(
        #     gripper_distance,
        #     self.reward_config.max_gripper_distance,
        #     self.reward_config.min_gripper_distance,
        #     self.reward_config.max_gripper_reward,
        #     self.reward_config.min_gripper_reward
        # )

        gripper_reward = 0

        reward = distance_reward + gripper_reward

        logger.info(
            f"Distance: {distance}, Distance Reward: {distance_reward}, "
            f"Right Gripper: {right_gripper}, Left Gripper: {left_gripper}, "
            f"Gripper Reward: {gripper_reward}, Total Reward: {reward}"
        )
        return reward

    def close(self) -> None:
        """Close environment and processes gracefully"""
        logger.info("Robot environment closed.")
