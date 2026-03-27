# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run the trained CNN behavioral cloning policy on the LeKiwi robot.

The policy takes a camera frame, resizes it to 84x84, and predicts
discrete velocity commands (x.vel, y.vel, theta.vel) which are sent
to the robot at 30 FPS.

Usage:
    python examples/lekiwi/run_policy.py --model /path/to/cnn_policy.pth
    python examples/lekiwi/run_policy.py --model /path/to/cnn_policy.pth --camera front
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Allow importing CNNPolicy from the BC repo
BC_REPO = os.environ.get(
    "BC_REPO", "/home/calvin/Documents/school/data/BC/ECE534_BehaviorCloning"
)
sys.path.insert(0, BC_REPO)
from models.cnn_policy import CNNPolicy

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30
REMOTE_IP = os.environ.get("REMOTE_IP")
IMAGE_SIZE = 84   # model expects 84x84
CAMERA = "front"  # camera key in observation dict

STOP_ACTION = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Resize camera frame to 84x84 RGB uint8."""
    return cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


def init_stop_listener():
    """Listen for SPACE (pause/resume) and ESC (stop) in a background thread."""
    stopped = {"value": False}
    paused = {"value": False}

    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.esc:
                print("\n[ESC] Emergency stop!")
                stopped["value"] = True
            elif key == keyboard.Key.space:
                paused["value"] = not paused["value"]
                state = "PAUSED" if paused["value"] else "RESUMED"
                print(f"\n[SPACE] {state}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener, stopped, paused
    except Exception:
        print("Warning: pynput not available, use Ctrl+C to stop.")
        return None, stopped, paused


def main():
    parser = argparse.ArgumentParser(description="Run CNN BC policy on LeKiwi")
    parser.add_argument("--model",  required=True, help="Path to cnn_policy.pth checkpoint")
    parser.add_argument("--camera", default=CAMERA, help="Camera key in observation (default: front)")
    parser.add_argument("--remote-ip", default=REMOTE_IP, help="Robot IP (or set REMOTE_IP env var)")
    args = parser.parse_args()

    remote_ip = args.remote_ip
    if not remote_ip:
        sys.exit("Error: Set REMOTE_IP env var or pass --remote-ip <ip>")
    if not os.path.exists(args.model):
        sys.exit(f"Error: Model file not found: {args.model}")

    # Load policy
    print(f"Loading model from {args.model} ...")
    policy = CNNPolicy()
    policy.load(args.model)
    policy.model.eval()
    print("Model loaded.")

    # Connect to robot
    robot_config = LeKiwiClientConfig(remote_ip=remote_ip, id="lekiwi")
    robot = LeKiwiClient(robot_config)
    robot.connect()

    if not robot.is_connected:
        sys.exit("Error: Robot not connected.")

    init_rerun(session_name="lekiwi_policy")

    listener, stopped, paused = init_stop_listener()

    print(f"Running policy at {FPS} FPS.")
    print("Controls: SPACE = pause/resume, ESC = emergency stop, Ctrl+C = quit")
    print("=" * 50)

    try:
        while not stopped["value"]:
            t0 = time.perf_counter()

            # Paused — send zero action and wait
            if paused["value"]:
                robot.send_action(STOP_ACTION)
                print("\r[PAUSED] Press SPACE to resume...          ", end="", flush=True)
                precise_sleep(1.0 / FPS)
                continue

            # Get camera frame from robot
            obs = robot.get_observation()
            frame = obs.get(args.camera)

            if frame is None:
                print("Warning: no camera frame received, skipping.")
                precise_sleep(1.0 / FPS)
                continue

            # Preprocess and run inference
            img = preprocess(frame)
            controls = policy.get_controls(img)

            # Map policy output keys to robot action keys
            action = {
                "x.vel":     controls["x_vel"],
                "y.vel":     controls["y_vel"],
                "theta.vel": controls["theta_vel"],
            }

            # Send to robot
            robot.send_action(action)

            # Print current action
            print(
                f"\rx={action['x.vel']:+.1f}  y={action['y.vel']:+.1f}  θ={action['theta.vel']:+.0f}°/s  [SPACE=pause, ESC=stop]",
                end="",
                flush=True,
            )

            # Visualize
            log_rerun_data(observation=obs, action=action)

            dt_s = time.perf_counter() - t0
            precise_sleep(max(1.0 / FPS - dt_s, 0.0))

    except KeyboardInterrupt:
        print("\nCtrl+C received.")
    finally:
        print("\nSending stop command to robot...")
        robot.send_action(STOP_ACTION)
        robot.disconnect()
        if listener:
            listener.stop()
        print("Done.")


if __name__ == "__main__":
    main()
