import gymnasium as gym
import numpy as np
import threading
import cv2
from MotorController import send_to_esp, close_connection
from CameraUtils import CameraProcessor
import time
from typing import Tuple, Dict, Optional


class RobotArmEnv(gym.Env):
    """
    Robot Arm Environment for Reinforcement Learning
    Handles robot arm control and camera-based observations
    """

    def __init__(self):
        super().__init__()

        # Initialize camera
        try:
            self.camera_processor = CameraProcessor()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera: {e}")

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(128, 128, 3),
            dtype=np.float32
        )

        # Initialize state variables
        self.arm_angles = np.array([90, 120, 0, 90, 0], dtype=np.float64)
        self.current_step = 0
        self.last_reward = 0.0
        self.done = False

        # Thread control
        self.render_flag = True
        self.lock = threading.Lock()
        self.render_thread = None

        # Movement constraints
        self.angle_limits = {
            'min': np.array([0, 60, 0, 0, 0]),
            'max': np.array([180, 120, 45, 120, 90])
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Reset arm position
        with self.lock:
            self.arm_angles = np.array([90, 120, 0, 90, 0], dtype=np.float64)
            self.process_command()

            # Reset internal state
            self.current_step = 0
            self.last_reward = 0.0
            self.done = False

        # Get initial observation
        state = self.camera_processor.capture_image()
        processed_state = self.camera_processor.process_image(state)

        # Start render thread if not running
        if self.render_thread is None or not self.render_thread.is_alive():
            self.start_render_thread()

        return processed_state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return new state"""
        with self.lock:
            # Update arm angles with clipping
            self.arm_angles = np.clip(
                self.arm_angles + action * 90,
                self.angle_limits['min'],
                self.angle_limits['max']
            )

            # Send command to hardware
            self.process_command()

        # Wait for movement to complete
        time.sleep(5)

        # Get new state
        state = self.camera_processor.capture_image()
        processed_state = self.camera_processor.process_image(state)

        # Calculate reward and check completion
        reward = self.calculate_reward(state)
        done = self.check_done(state)

        # Update internal state
        with self.lock:
            self.current_step += 1
            self.last_reward = reward
            self.done = done

        # Debug info
        print(f"Step {self.current_step}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Done: {done}")

        info = {
            "TimeLimit.truncated": False,
            "done": done,
            "reward": reward,
            "arm_angles": self.arm_angles.tolist()
        }

        return processed_state, reward, done, False, info

    def render(self) -> None:
        """Render environment state"""
        cap = self.camera_processor.cap

        while self.render_flag and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Warning: Failed to read a valid camera frame")
                time.sleep(0.1)
                continue

            try:
                # Process frame
                frame = cv2.flip(frame, 1)
                frame = self.camera_processor.zoom_camera(frame)

                # Check frame format
                if frame is None or frame.dtype != np.uint8:
                    print("Warning: Frame format is invalid")
                    continue

                # Add overlay information
                with self.lock:
                    step_text = f"Step: {self.current_step}"
                    reward_text = f"Reward: {self.last_reward:.2f}"
                    done_text = "Done" if self.done else "Not Done"

                # Render overlay
                cv2.putText(frame, step_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, reward_text, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, done_text, (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Check the frame before showing it
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print("Warning: Frame does not have 3 channels or is malformed")
                    continue
                else:
                    cv2.imshow('RobotArmEnv', frame)

                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.render_flag = False
                    break

            except Exception as e:
                print(f"Warning: Render error occurred: {e}. Frame shape: {frame.shape if 'frame' in locals() else 'Not defined'}")
                time.sleep(0.1)

        cv2.destroyAllWindows()


    def start_render_thread(self) -> None:
        """Start the rendering thread"""
        self.render_flag = True
        self.render_thread = threading.Thread(target=self.render)
        self.render_thread.daemon = True
        self.render_thread.start()

    def stop_render_thread(self) -> None:
        """Stop the rendering thread"""
        self.render_flag = False
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)

    def process_command(self) -> None:
        """Send command to robot arm"""
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

    def calculate_reward(self, state: np.ndarray) -> float:
        """Calculate reward based on current state"""
        distance_from_center = self.camera_processor.calculate_distance(
            state) + 1e-6
        return 1 / distance_from_center

    def check_done(self, state: np.ndarray) -> bool:
        """Check if episode is complete"""
        return self.camera_processor.check_if_pink_on_blue(state)

    def close(self) -> None:
        """Clean up resources"""
        self.stop_render_thread()
        cv2.destroyAllWindows()
        self.camera_processor.release()
        close_connection()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.close()
