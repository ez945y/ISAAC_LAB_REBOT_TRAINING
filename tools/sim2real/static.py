#!/usr/bin/env python
"""
Check and fix stats.json for LeRobot datasets.
Identifies which datasets have wrong image stats shapes and fixes them.

Usage:
    python sim2real/fix_stats.py \
        --roots /path/to/dataset1 /path/to/dataset2
"""

import argparse
import json
import logging
from pathlib import Path

import av
import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import (
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
    DEFAULT_QUANTILES,
)
from lerobot.datasets.utils import write_stats

logging.basicConfig(level=logging.INFO)


def check_stats(root: Path) -> dict:
    """Check stats.json and return problematic keys."""
    stats_path = root / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    problems = {}
    for key, s in stats.items():
        if "image" not in key:
            continue
        for stat_name in ["min", "max", "mean", "std"]:
            shape = np.array(s[stat_name]).shape
            if shape != (3, 1, 1):
                problems[key] = shape
                break

    return problems


def recompute_image_stats_from_video(root: Path, key: str) -> dict:
    """Decode videos and compute proper (3,1,1) stats for an image feature."""
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    video_path_template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )

    # Load episodes metadata
    episodes_dir = root / "meta" / "episodes"
    ep_parquets = sorted(episodes_dir.rglob("*.parquet"))
    ep_df = pd.concat([pd.read_parquet(p) for p in ep_parquets], ignore_index=True)

    # Decode frames from all episodes
    all_frames = []
    for _, ep in ep_df.iterrows():
        chunk_idx = int(ep[f"videos/{key}/chunk_index"])
        file_idx = int(ep[f"videos/{key}/file_index"])
        from_ts = float(ep[f"videos/{key}/from_timestamp"])
        to_ts = float(ep[f"videos/{key}/to_timestamp"])

        vpath = str(root / video_path_template.format(
            video_key=key, chunk_index=chunk_idx, file_index=file_idx,
        ))

        with av.open(vpath) as container:
            stream = container.streams.video[0]
            for frame in container.decode(video=0):
                ts = float(frame.pts * stream.time_base)
                if ts < from_ts - 0.001:
                    continue
                if ts > to_ts + 0.001:
                    break
                img = frame.to_ndarray(format="rgb24")  # (H, W, 3) HWC
                img = img.transpose(2, 0, 1)  # (3, H, W) CHW
                all_frames.append(img)

    all_frames = np.stack(all_frames)  # (N, 3, H, W)
    logging.info(f"    Decoded {len(all_frames)} frames, shape={all_frames.shape}")

    # Sample and downsample
    sampled_idx = sample_indices(len(all_frames))
    sampled = np.stack([auto_downsample_height_width(all_frames[i]) for i in sampled_idx])
    logging.info(f"    Sampled {len(sampled)} frames, shape={sampled.shape}")

    # Compute stats: axis=(0,2,3) reduces batch,H,W → keeps C
    stats = get_feature_stats(sampled, axis=(0, 2, 3), keepdims=True, quantile_list=DEFAULT_QUANTILES)

    # Normalize to [0,1] and squeeze batch dim: (1,3,1,1) → (3,1,1)
    stats = {
        k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
        for k, v in stats.items()
    }

    logging.info(f"    Result shape: {stats['min'].shape}")
    assert stats["min"].shape == (3, 1, 1), f"Expected (3,1,1) but got {stats['min'].shape}"

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, help="Root directories of datasets")
    args = parser.parse_args()

    for root_str in args.roots:
        root = Path(root_str)
        logging.info(f"\n{'='*60}")
        logging.info(f"Checking: {root}")

        problems = check_stats(root)

        if not problems:
            logging.info("  ✅ All image stats have correct (3,1,1) shape")
            continue

        for key, shape in problems.items():
            logging.info(f"  ❌ {key}: shape={shape}, recomputing from video...")
            new_image_stats = recompute_image_stats_from_video(root, key)

            # Load existing stats, replace only the problematic key
            stats_path = root / "meta" / "stats.json"
            with open(stats_path) as f:
                all_stats = json.load(f)

            # Convert existing stats to numpy for write_stats
            full_stats = {}
            for k, s in all_stats.items():
                full_stats[k] = {sk: np.array(sv) for sk, sv in s.items()}

            # Replace the problematic key
            full_stats[key] = new_image_stats

            write_stats(full_stats, root)
            logging.info(f"  ✅ Fixed {key}")

        # Verify
        problems_after = check_stats(root)
        if not problems_after:
            logging.info(f"  ✅ All fixed!")
        else:
            logging.info(f"  ❌ Still broken: {problems_after}")


if __name__ == "__main__":
    main()
