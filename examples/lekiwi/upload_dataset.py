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
Upload a locally-recorded LeRobot dataset to HuggingFace.

Examples:
    python examples/lekiwi/upload_dataset.py --repo-id user/my_dataset
    python examples/lekiwi/upload_dataset.py --repo-id user/my_dataset --private
    python examples/lekiwi/upload_dataset.py --repo-id user/my_dataset --tags lekiwi green-block
"""

import argparse
import os
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Upload local LeRobot dataset to HuggingFace")
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID"),
                        help="Dataset id (e.g. user/dataset_name). Falls back to HF_REPO_ID env var.")
    parser.add_argument("--private", action="store_true",
                        help="Upload as a private dataset.")
    parser.add_argument("--tags", nargs="*", default=None,
                        help="Optional list of tags to attach to the dataset.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.repo_id:
        sys.exit("Error: Pass --repo-id or set HF_REPO_ID env var.")

    print(f"Loading local dataset: {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id)

    print(f"  Episodes: {dataset.meta.total_episodes}")
    print(f"  Frames:   {dataset.meta.total_frames}")
    print(f"  Local path: {dataset.root}")

    print(f"\nUploading to https://huggingface.co/datasets/{args.repo_id} ...")
    dataset.push_to_hub(tags=args.tags, private=args.private)

    print(f"\nDone! Dataset available at:")
    print(f"  https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
