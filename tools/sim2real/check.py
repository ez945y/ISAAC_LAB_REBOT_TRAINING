#!/usr/bin/env python
"""
Check if a LeRobot V3 dataset is healthy.
Verifies all required metadata files, structure, and consistency.

Usage:
    python sim2real/check_dataset.py --root /path/to/dataset
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def check(label, condition, fix_hint=""):
    status = "✅" if condition else "❌"
    print(f"  {status} {label}")
    if not condition and fix_hint:
        print(f"     💡 Fix: {fix_hint}")
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory of the dataset")
    args = parser.parse_args()

    root = Path(args.root)
    all_ok = True

    print(f"\n🔍 Checking dataset at: {root}\n")

    # ═══════════════════════════════════════════
    # 1. Required files
    # ═══════════════════════════════════════════
    print("── 1. Required Files ──")

    info_path = root / "meta" / "info.json"
    all_ok &= check("meta/info.json exists", info_path.exists())

    stats_path = root / "meta" / "stats.json"
    all_ok &= check("meta/stats.json exists", stats_path.exists(),
                     "Run recompute_stats.py to generate")

    tasks_path = root / "meta" / "tasks.parquet"
    all_ok &= check("meta/tasks.parquet exists", tasks_path.exists(),
                     "Need tasks.parquet (V3 format), not tasks.jsonl")

    # V3 episodes parquet
    episodes_parquet_dir = root / "meta" / "episodes"
    has_episodes_parquet = any(episodes_parquet_dir.rglob("*.parquet")) if episodes_parquet_dir.exists() else False
    all_ok &= check("meta/episodes/chunk-*/file-*.parquet exists (V3)", has_episodes_parquet,
                     "V2 uses episodes.jsonl, V3 needs meta/episodes/chunk-000/file-000.parquet")

    # V2 leftover check
    episodes_jsonl = root / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists() and not has_episodes_parquet:
        print("  ⚠️  Found episodes.jsonl (V2 format) but no V3 episodes parquet!")

    # Data parquet
    data_dir = root / "data"
    data_parquets = sorted(data_dir.glob("**/*.parquet")) if data_dir.exists() else []
    all_ok &= check(f"data/ parquet files exist ({len(data_parquets)} found)", len(data_parquets) > 0)

    if not info_path.exists():
        print("\n❌ Cannot continue without info.json")
        sys.exit(1)

    # ═══════════════════════════════════════════
    # 2. info.json content
    # ═══════════════════════════════════════════
    print("\n── 2. info.json Content ──")

    with open(info_path) as f:
        info = json.load(f)

    version = info.get("codebase_version", "missing")
    all_ok &= check(f"codebase_version = '{version}' (should be 'v3.0')",
                     version == "v3.0",
                     'Set "codebase_version": "v3.0" in info.json')

    all_ok &= check(f"total_episodes = {info.get('total_episodes', 'missing')}",
                     "total_episodes" in info)
    all_ok &= check(f"total_frames = {info.get('total_frames', 'missing')}",
                     "total_frames" in info)
    all_ok &= check(f"total_tasks = {info.get('total_tasks', 'missing')}",
                     "total_tasks" in info)
    all_ok &= check(f"chunks_size = {info.get('chunks_size', 'missing')}",
                     "chunks_size" in info)
    all_ok &= check(f"splits present", "splits" in info,
                     'Need "splits": {"train": "0:N"} where N = total_episodes')

    features = info.get("features", {})
    all_ok &= check(f"features defined ({len(features)} keys)", len(features) > 0)

    required_feature_keys = ["index", "episode_index", "frame_index", "timestamp", "task_index"]
    for k in required_feature_keys:
        all_ok &= check(f"  feature '{k}' present", k in features,
                         f"Add '{k}' to features in info.json")

    # ═══════════════════════════════════════════
    # 3. Data consistency
    # ═══════════════════════════════════════════
    print("\n── 3. Data Consistency ──")

    if data_parquets:
        df = pd.concat([pd.read_parquet(p) for p in data_parquets], ignore_index=True)
        n_frames = len(df)
        n_episodes = df["episode_index"].nunique()

        all_ok &= check(f"Frames in parquet: {n_frames} (info says {info.get('total_frames', '?')})",
                         n_frames == info.get("total_frames", -1),
                         "Update total_frames in info.json")
        all_ok &= check(f"Episodes in parquet: {n_episodes} (info says {info.get('total_episodes', '?')})",
                         n_episodes == info.get("total_episodes", -1),
                         "Update total_episodes in info.json")

        # Check required columns
        required_cols = ["index", "episode_index", "frame_index", "timestamp", "task_index"]
        for col in required_cols:
            all_ok &= check(f"  Column '{col}' in parquet", col in df.columns,
                             f"Missing column: {col}")

        # Check index is contiguous 0..N-1
        if "index" in df.columns:
            expected_idx = list(range(n_frames))
            actual_idx = df["index"].tolist()
            all_ok &= check("index is contiguous 0..N-1",
                             actual_idx == expected_idx,
                             "index column must be sequential starting from 0")

    # ═══════════════════════════════════════════
    # 4. Episodes metadata (V3)
    # ═══════════════════════════════════════════
    print("\n── 4. Episodes Metadata (V3) ──")

    if has_episodes_parquet:
        ep_parquets = sorted(episodes_parquet_dir.rglob("*.parquet"))
        ep_df = pd.concat([pd.read_parquet(p) for p in ep_parquets], ignore_index=True)
        all_ok &= check(f"Episodes parquet has {len(ep_df)} rows",
                         len(ep_df) == info.get("total_episodes", -1))

        for col in ["episode_index", "length", "dataset_from_index", "dataset_to_index"]:
            all_ok &= check(f"  Episodes column '{col}' present",
                             col in ep_df.columns)
    else:
        print("  ⏭️  Skipped (no V3 episodes parquet found)")

    # ═══════════════════════════════════════════
    # 5. Videos
    # ═══════════════════════════════════════════
    print("\n── 5. Videos ──")

    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    if video_keys:
        for vk in video_keys:
            video_dir = root / "videos" / vk
            mp4s = sorted(video_dir.rglob("*.mp4")) if video_dir.exists() else []
            all_ok &= check(f"  {vk}: {len(mp4s)} mp4 file(s)", len(mp4s) > 0,
                             f"No mp4 found at videos/{vk}/chunk-*/file-*.mp4")
    else:
        print("  ⏭️  No video features defined")

    # ═══════════════════════════════════════════
    # 6. Hub tag
    # ═══════════════════════════════════════════
    print("\n── 6. Hub Version Tag ──")
    print(f"  ℹ️  If pushing to Hub, you need a version tag:")
    print(f'     hub_api.create_tag("{info.get("repo_id", "YOUR/REPO")}", tag="{version}", repo_type="dataset")')

    # ═══════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ Dataset looks healthy!")
    else:
        print("❌ Dataset has issues. Fix the items above.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
