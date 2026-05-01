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
Record a LeKiwi dataset locally. Does NOT upload to HuggingFace.
Use upload_dataset.py to push to the Hub afterwards.

Examples:
    python examples/lekiwi/record.py --repo-id user/my_dataset --num-episodes 10
    python examples/lekiwi/record.py --repo-id user/my_dataset --num-episodes 5 --resume
"""

import argparse
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

FPS = 30
TASK_DESCRIPTION = "Go towards a green block, no obstacles"

# W/A/S/D for translation, Q/E for rotation
TELEOP_KEYS = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "rotate_left": "q",
    "rotate_right": "e",
    "speed_up": "r",
    "speed_down": "f",
    "quit": "p",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Record LeKiwi dataset locally")
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID"),
                        help="Dataset id (e.g. user/dataset_name). Falls back to HF_REPO_ID env var.")
    parser.add_argument("--remote-ip", default=os.environ.get("REMOTE_IP"),
                        help="Robot IP. Falls back to REMOTE_IP env var.")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Number of episodes to record (default: 50)")
    parser.add_argument("--episode-time", type=int, default=20,
                        help="Max seconds per episode (default: 20)")
    parser.add_argument("--reset-time", type=int, default=20,
                        help="Seconds for reset between episodes (default: 20)")
    parser.add_argument("--task", default=TASK_DESCRIPTION,
                        help=f"Task description (default: '{TASK_DESCRIPTION}')")
    parser.add_argument("--resume", action="store_true",
                        help="Append episodes to an existing local dataset instead of creating new.")
    return parser.parse_args()


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

        if int(timestamp * fps) % fps == 0:
            print(f"\r{phase_label} | Time remaining: {remaining}s", end="", flush=True)

        obs = robot.get_observation()

        keyboard_keys = keyboard.get_action()
        action = robot._from_keyboard_to_base_action(keyboard_keys)

        robot.send_action(action)

        if dataset is not None:
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": task}
            dataset.add_frame(frame)

        log_rerun_data(observation=obs, action=action)

        dt_s = time.perf_counter() - t0
        precise_sleep(max(1.0 / fps - dt_s, 0.0))
        timestamp = time.perf_counter() - start_t

    print()


def main():
    args = parse_args()

    if not args.repo_id:
        sys.exit("Error: Pass --repo-id or set HF_REPO_ID env var.")
    if not args.remote_ip:
        sys.exit("Error: Pass --remote-ip or set REMOTE_IP env var.")

    robot_config = LeKiwiClientConfig(
        remote_ip=args.remote_ip,
        id="lekiwi",
        teleop_keys=TELEOP_KEYS,
    )
    keyboard_config = KeyboardTeleopConfig()

    robot = LeKiwiClient(robot_config)
    robot.speed_index = 2  # start at fast — slow speed can't strafe (omniwheel friction)
    keyboard = KeyboardTeleop(keyboard_config)

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    if args.resume:
        print(f"Resuming dataset: {args.repo_id}")
        dataset = LeRobotDataset(args.repo_id)
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(num_processes=0, num_threads=4 * len(robot.cameras))
        starting_episodes = dataset.meta.total_episodes
        print(f"Existing episodes: {starting_episodes}")
    else:
        print(f"Creating new dataset: {args.repo_id}")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
        starting_episodes = 0

    robot.connect()
    keyboard.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_record")

    try:
        if not robot.is_connected or not keyboard.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print(f"Starting recording: {args.num_episodes} episodes, {args.episode_time}s each")
        print("Drive: W/A/S/D (translate), Q/E (rotate), R/F (speed)")
        print("Recording: Right arrow = end episode, Left arrow = re-record, ESC = stop")
        print("=" * 60)

        recorded_episodes = 0
        while recorded_episodes < args.num_episodes and not events["stop_recording"]:
            clear_events(events)
            ep_total = starting_episodes + recorded_episodes + 1
            print(f"\n>>> RECORDING episode {recorded_episodes + 1}/{args.num_episodes} (dataset total: {ep_total})")
            time.sleep(1)

            run_episode(
                robot=robot,
                keyboard=keyboard,
                events=events,
                fps=FPS,
                control_time_s=args.episode_time,
                phase_label=f"RECORDING {recorded_episodes + 1}/{args.num_episodes}",
                dataset=dataset,
                task=args.task,
            )

            if events["stop_recording"]:
                break

            if events["rerecord_episode"]:
                print("<<< RE-RECORDING this episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1
            print(f"<<< Episode {recorded_episodes}/{args.num_episodes} saved!")

            if recorded_episodes < args.num_episodes and not events["stop_recording"]:
                clear_events(events)
                print(f"\n--- RESET PHASE — reposition. Press right arrow when ready.")

                run_episode(
                    robot=robot,
                    keyboard=keyboard,
                    events=events,
                    fps=FPS,
                    control_time_s=args.reset_time,
                    phase_label="RESETTING",
                )

        print(f"\nDone! Recorded {recorded_episodes} new episodes (dataset total: {starting_episodes + recorded_episodes}).")
    finally:
        log_say("Stop recording")
        robot.disconnect()
        keyboard.disconnect()
        listener.stop()

        dataset.finalize()
        cache_path = dataset.root
        print(f"\nDataset saved locally at: {cache_path}")
        print(f"To upload to HuggingFace, run:")
        print(f"  python examples/lekiwi/upload_dataset.py --repo-id {args.repo_id}")


if __name__ == "__main__":
    main()
