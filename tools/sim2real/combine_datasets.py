#!/usr/bin/env python
"""
Duplicate episodes from an existing LeRobot dataset.

Takes the first N episodes and copies each one K times,
producing N*K total episodes with all proper V3 metadata.

Usage:
    python sim2real/duplicate_dataset.py \
        --src MikeChenYZ/soarm-fmb \
        --dst MikeChenYZ/soarm-fmb-x5 \
        --episodes 0 1 2 3 \
        --copies 5
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Duplicate episodes from a LeRobot dataset")
    parser.add_argument("--src", required=True, help="Source dataset repo_id (e.g. MikeChenYZ/soarm-fmb)")
    parser.add_argument("--dst", required=True, help="Destination dataset repo_id")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Episode indices to duplicate (default: 0 1 2 3)",
    )
    parser.add_argument("--copies", type=int, default=5, help="Number of copies per episode (default: 5)")
    parser.add_argument("--src-root", type=str, default=None, help="Root dir of source dataset (optional)")
    parser.add_argument("--dst-root", type=str, default=None, help="Root dir of destination dataset (optional)")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub after creation")
    args = parser.parse_args()

    # ── Load source dataset ──
    logging.info(f"Loading source dataset: {args.src} (episodes={args.episodes})")
    src_ds = LeRobotDataset(
        args.src,
        root=args.src_root,
        episodes=args.episodes,
    )

    logging.info(f"Source: {src_ds.meta.total_episodes} total episodes, {src_ds.meta.total_frames} total frames")
    logging.info(f"FPS: {src_ds.fps}")
    logging.info(f"Features: {list(src_ds.features.keys())}")

    # ── Create destination dataset with same features ──
    dst_root = args.dst_root or str(Path.home() / ".cache/huggingface/lerobot" / args.dst)

    # Remove existing destination if present
    dst_path = Path(dst_root)
    if dst_path.exists():
        import shutil
        logging.warning(f"Removing existing destination: {dst_path}")
        shutil.rmtree(dst_path)

    dst_ds = LeRobotDataset.create(
        repo_id=args.dst,
        fps=src_ds.fps,
        features=src_ds.features,
        root=dst_root,
        use_videos=len(src_ds.meta.video_keys) > 0,
    )

    # ── Copy episodes ──
    total_new_episodes = len(args.episodes) * args.copies
    logging.info(f"Will create {total_new_episodes} episodes ({len(args.episodes)} × {args.copies})")

    for ep_idx in args.episodes:
        # Get frame indices for this episode from the source
        ep_meta = src_ds.meta.episodes[ep_idx]
        ep_from = ep_meta["dataset_from_index"]
        ep_to = ep_meta["dataset_to_index"]
        ep_len = ep_to - ep_from

        # Get the task for this episode
        src_item_0 = src_ds[ep_from if src_ds._absolute_to_relative_idx is None
                            else src_ds._absolute_to_relative_idx[ep_from]]
        task_str = src_item_0["task"]

        logging.info(f"Episode {ep_idx}: {ep_len} frames, task='{task_str}'")

        for copy_i in range(args.copies):
            logging.info(f"  Copy {copy_i + 1}/{args.copies} → new episode {dst_ds.meta.total_episodes}")

            for frame_i in tqdm(range(ep_len), desc=f"  ep{ep_idx} copy{copy_i}", leave=False):
                abs_idx = ep_from + frame_i

                # Map to relative index if episodes are filtered
                if src_ds._absolute_to_relative_idx is not None:
                    rel_idx = src_ds._absolute_to_relative_idx[abs_idx]
                else:
                    rel_idx = abs_idx

                src_item = src_ds[rel_idx]

                frame = {"task": task_str}

                for key, ft in src_ds.features.items():
                    if key in ["index", "episode_index", "frame_index", "timestamp", "task_index"]:
                        continue

                    val = src_item[key]
                    if isinstance(val, torch.Tensor):
                        val = val.numpy()

                    # For video/image features, convert from CHW float [0,1] → HWC uint8
                    if ft["dtype"] in ["video", "image"]:
                        if val.ndim == 3 and val.shape[0] in [1, 3]:
                            # CHW → HWC
                            val = np.transpose(val, (1, 2, 0))
                        if val.dtype in [np.float32, np.float64]:
                            val = (val * 255).clip(0, 255).astype(np.uint8)

                    frame[key] = val

                dst_ds.add_frame(frame)

            dst_ds.save_episode()

    # ── Finalize ──
    logging.info(f"\nDone! Created dataset at: {dst_ds.root}")
    logging.info(f"  Episodes: {dst_ds.meta.total_episodes}")
    logging.info(f"  Frames: {dst_ds.meta.total_frames}")

    if args.push:
        logging.info("Pushing to Hub...")
        dst_ds.push_to_hub()
        logging.info("Pushed!")


if __name__ == "__main__":
    main()
