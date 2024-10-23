import gymnasium as gym
import numpy as np
import threading
import cv2
from MotorController import send_to_esp, close_connection
from CameraUtils import CameraProcessor


class RobotArmEnv(gym.Env):
    def __init__(self):
        super(RobotArmEnv, self).__init__()

        self.camera_processor = CameraProcessor()

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(128, 128, 3), dtype=np.float32)

        self.arm_angles = np.array([90, 120, 0, 90, 0], dtype=np.float64)

        self.current_step = 0
        self.last_reward = 0.0
        self.done = False
        self.render_flag = True
        self.lock = threading.Lock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.arm_angles = np.array([90, 120, 0, 90, 0], dtype=np.float64)
        self.process_command()

        state = self.camera_processor.capture_image()
        processed_state = self.camera_processor.process_image(state)

        self.current_step = 0
        self.done = False

        self.start_render_thread()

        return processed_state, {}

    def step(self, action):
        self.arm_angles = np.clip(
            self.arm_angles + action * 90,
            [0, 0, 0, 0, 0],
            [180, 180, 180, 180, 25]
        )

        self.process_command()

        state = self.camera_processor.capture_image()
        processed_state = self.camera_processor.process_image(state)

        reward = self.calculate_reward(state)

        done = self.check_done(state)

        info = {
            "TimeLimit.truncated": False,
            "done": done,
            "reward": reward,
            "arm_angles": self.arm_angles.tolist()
        }

        self.current_step += 1
        self.last_reward = reward
        self.done = done

        print(f"Action taken: {action}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")

        return processed_state, reward, done, False, info

    def process_command(self):
        command = (
            '{0:' + str(self.arm_angles[0]) +
            ',3:' + str(self.arm_angles[1]) +
            ',7:' + str(180 - self.arm_angles[1]) +
            ',11:' + str(self.arm_angles[2]) +
            ',13:' + str(self.arm_angles[3]) +
            ',15:' + str(self.arm_angles[4]) + '}'
        )

        send_to_esp(command)

    def calculate_reward(self, state):
        distance_from_center = self.camera_processor.calculate_distance(
            state) + 1e-6

        reward = 1 / distance_from_center
        return reward

    def check_done(self, state):
        return self.camera_processor.check_if_pink_on_blue(state)

    def render(self, mode="human"):
        while self.render_flag:
            image = self.camera_processor.capture_image()

            image = (image * 255).astype(np.uint8)

            step_text = f"Step: {self.current_step}"
            reward_text = f"Reward: {self.last_reward:.2f}"
            done_text = "Done" if self.done else "Not Done"

            cv2.putText(image, step_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, reward_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, done_text, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('RobotArmEnv', image)

            cv2.waitKey(1)

        cv2.destroyAllWindows()

    def start_render_thread(self):
        render_thread = threading.Thread(target=self.render)
        render_thread.start()

    def stop_render_thread(self):
        self.render_flag = False

    def close(self):
        self.render_flag = False
        self.camera_processor.release()
        close_connection()
