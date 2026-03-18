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

import os
import sys
import time

from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

NUM_EPISODES = 50
FPS = 30
EPISODE_TIME_SEC = 20
RESET_TIME_SEC = 20
TASK_DESCRIPTION = "Go towards a green block, no obstacles"
HF_REPO_ID = os.environ.get("HF_REPO_ID")
REMOTE_IP = os.environ.get("REMOTE_IP")


def clear_events(events):
    """Clear all event flags to prevent stale key presses from carrying over."""
    events["exit_early"] = False
    events["rerecord_episode"] = False


def run_episode(robot, keyboard, events, fps, control_time_s, phase_label="", dataset=None, task=None):
    """Run one episode: drive the robot with keyboard, optionally saving to dataset."""
    timestamp = 0
    start_t = time.perf_counter()

    while timestamp < control_time_s:
        t0 = time.perf_counter()
        remaining = int(control_time_s - timestamp)

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Print status every second
        if int(timestamp * fps) % fps == 0:
            print(f"\r{phase_label} | Time remaining: {remaining}s", end="", flush=True)

        # Get observation (camera frames + robot state)
        obs = robot.get_observation()

        # Get keyboard input and convert to velocity commands
        keyboard_keys = keyboard.get_action()
        action = robot._from_keyboard_to_base_action(keyboard_keys)

        # Send action to robot
        robot.send_action(action)

        # Save frame to dataset
        if dataset is not None:
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": task}
            dataset.add_frame(frame)

        # Visualize
        log_rerun_data(observation=obs, action=action)

        dt_s = time.perf_counter() - t0
        precise_sleep(max(1.0 / fps - dt_s, 0.0))
        timestamp = time.perf_counter() - start_t

    print()  # newline after the status line


def main():
    if not HF_REPO_ID:
        sys.exit("Error: Set HF_REPO_ID env var, e.g. export HF_REPO_ID=username/dataset_name")
    if not REMOTE_IP:
        sys.exit("Error: Set REMOTE_IP env var, e.g. export REMOTE_IP=192.168.1.100")

    # Create the robot and teleoperator configurations
    robot_config = LeKiwiClientConfig(remote_ip=REMOTE_IP, id="lekiwi")
    keyboard_config = KeyboardTeleopConfig()

    # Initialize the robot and teleoperator
    robot = LeKiwiClient(robot_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Connect the robot and teleoperator
    robot.connect()
    keyboard.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_record")

    try:
        if not robot.is_connected or not keyboard.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print(f"Starting recording: {NUM_EPISODES} episodes, {EPISODE_TIME_SEC}s each")
        print("Controls: W/A/S/D to drive, Right arrow = end episode, Left arrow = re-record, ESC = stop all")
        print("=" * 60)

        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            clear_events(events)
            print(f"\n>>> RECORDING episode {recorded_episodes + 1}/{NUM_EPISODES} — drive toward the green block!")
            time.sleep(1)  # brief pause so you can read the message

            # Record episode
            run_episode(
                robot=robot,
                keyboard=keyboard,
                events=events,
                fps=FPS,
                control_time_s=EPISODE_TIME_SEC,
                phase_label=f"RECORDING ep {recorded_episodes + 1}/{NUM_EPISODES}",
                dataset=dataset,
                task=TASK_DESCRIPTION,
            )

            if events["stop_recording"]:
                break

            # Check for re-record before saving
            if events["rerecord_episode"]:
                print("<<< RE-RECORDING this episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            recorded_episodes += 1
            print(f"<<< Episode {recorded_episodes}/{NUM_EPISODES} saved!")

            # Reset phase (no recording) — skip for last episode
            if recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
                clear_events(events)
                print(f"\n--- RESET PHASE — move the block back, reposition robot. Press right arrow when ready.")

                run_episode(
                    robot=robot,
                    keyboard=keyboard,
                    events=events,
                    fps=FPS,
                    control_time_s=RESET_TIME_SEC,
                    phase_label="RESETTING",
                )

        print(f"\nDone! Recorded {recorded_episodes} episodes.")
    finally:
        # Clean up
        log_say("Stop recording")
        robot.disconnect()
        keyboard.disconnect()
        listener.stop()

        dataset.finalize()
        dataset.push_to_hub()
        print(f"Dataset uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
