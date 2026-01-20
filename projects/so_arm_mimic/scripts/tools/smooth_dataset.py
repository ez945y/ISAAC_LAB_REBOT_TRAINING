#!/usr/bin/env python3
"""
Smooth Dataset Actions and Poses

This script smooths the actions and target EEF poses in an annotated HDF5 dataset
to create smoother robot trajectories for data generation.

Smoothing methods:
1. Moving Average - Simple but effective
2. Gaussian Filter - Better at preserving shape
3. Savitzky-Golay Filter - Best for preserving peaks while smoothing

Usage:
    python smooth_dataset.py \
        --input_file ./datasets/annotated_dataset.hdf5 \
        --output_file ./datasets/smoothed_dataset.hdf5 \
        --method savgol \
        --window_size 7
"""

import argparse
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import shutil
import os


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(data) < window_size:
        return data
    
    # Pad the data to handle boundaries
    pad_size = window_size // 2
    padded = np.pad(data, ((pad_size, pad_size), (0, 0)), mode='edge')
    
    result = np.zeros_like(data)
    for i in range(len(data)):
        result[i] = np.mean(padded[i:i+window_size], axis=0)
    
    return result


def gaussian_smooth(data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian filter smoothing."""
    if len(data) < 5:
        return data
    
    # Apply gaussian filter along time axis (axis 0) for each dimension
    result = np.zeros_like(data)
    for dim in range(data.shape[1]):
        result[:, dim] = gaussian_filter1d(data[:, dim], sigma=sigma, mode='nearest')
    
    return result


def savgol_smooth(data: np.ndarray, window_size: int = 7, polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay filter smoothing (best for preserving shape)."""
    if len(data) < window_size:
        return data
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Ensure polyorder < window_size
    polyorder = min(polyorder, window_size - 1)
    
    result = np.zeros_like(data)
    for dim in range(data.shape[1]):
        result[:, dim] = savgol_filter(data[:, dim], window_size, polyorder, mode='nearest')
    
    return result


def smooth_quaternion(quats: np.ndarray, method: str, window_size: int) -> np.ndarray:
    """
    Smooth quaternions while maintaining unit norm.
    
    Quaternions need special handling because they live on a hypersphere.
    We use a simple approach: smooth, then renormalize.
    """
    if method == 'moving_average':
        smoothed = moving_average(quats, window_size)
    elif method == 'gaussian':
        smoothed = gaussian_smooth(quats, sigma=window_size/3)
    elif method == 'savgol':
        smoothed = savgol_smooth(quats, window_size)
    else:
        return quats
    
    # Renormalize to unit quaternions
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    smoothed = smoothed / norms
    
    return smoothed


def smooth_pose_matrix(poses: np.ndarray, method: str, window_size: int) -> np.ndarray:
    """
    Smooth 4x4 pose matrices.
    
    Args:
        poses: Array of shape (T, 4, 4)
        method: Smoothing method
        window_size: Window size for smoothing
        
    Returns:
        Smoothed poses of shape (T, 4, 4)
    """
    if len(poses.shape) != 3 or poses.shape[1:] != (4, 4):
        print(f"  Warning: Unexpected pose shape {poses.shape}, skipping")
        return poses
    
    T = poses.shape[0]
    
    # Extract positions (xyz) from poses[:, :3, 3]
    positions = poses[:, :3, 3].copy()  # Shape (T, 3)
    
    # Smooth positions
    if method == 'moving_average':
        smoothed_positions = moving_average(positions, window_size)
    elif method == 'gaussian':
        smoothed_positions = gaussian_smooth(positions, sigma=window_size/3)
    elif method == 'savgol':
        smoothed_positions = savgol_smooth(positions, window_size)
    else:
        smoothed_positions = positions
    
    # For rotation, we'll SLERP between adjacent frames (simplified: just copy)
    # A more sophisticated approach would use quaternion SLERP smoothing
    smoothed_poses = poses.copy()
    smoothed_poses[:, :3, 3] = smoothed_positions
    
    return smoothed_poses


def smooth_actions(actions: np.ndarray, method: str, window_size: int, 
                   gripper_threshold: float = 0.5) -> np.ndarray:
    """
    Smooth action array while preserving gripper commands.
    
    Actions format: [pos(3), quat(4), gripper(1)] = 8 dims
    
    Args:
        actions: Array of shape (T, 8)
        method: Smoothing method
        window_size: Window size
        gripper_threshold: Threshold for binary gripper decision
        
    Returns:
        Smoothed actions
    """
    if actions.shape[1] != 8:
        print(f"  Warning: Unexpected action shape {actions.shape}, expected 8 dims")
        # Try to smooth anyway
        if method == 'moving_average':
            return moving_average(actions, window_size)
        elif method == 'gaussian':
            return gaussian_smooth(actions, sigma=window_size/3)
        elif method == 'savgol':
            return savgol_smooth(actions, window_size)
        return actions
    
    smoothed = actions.copy()
    
    # Smooth position (first 3 dims)
    positions = actions[:, :3]
    if method == 'moving_average':
        smoothed[:, :3] = moving_average(positions, window_size)
    elif method == 'gaussian':
        smoothed[:, :3] = gaussian_smooth(positions, sigma=window_size/3)
    elif method == 'savgol':
        smoothed[:, :3] = savgol_smooth(positions, window_size)
    
    # Smooth quaternion (dims 3-7) with renormalization
    quaternions = actions[:, 3:7]
    smoothed[:, 3:7] = smooth_quaternion(quaternions, method, window_size)
    
    # For gripper (dim 7), use a gentler smoothing to preserve open/close timing
    gripper = actions[:, 7:8]
    if method == 'moving_average':
        smoothed[:, 7:8] = moving_average(gripper, max(3, window_size // 2))
    elif method == 'gaussian':
        smoothed[:, 7:8] = gaussian_smooth(gripper, sigma=max(1, window_size / 6))
    elif method == 'savgol':
        smoothed[:, 7:8] = savgol_smooth(gripper, max(5, window_size // 2))
    
    return smoothed


def smooth_episode(episode_group: h5py.Group, method: str, window_size: int):
    """Smooth all relevant data in an episode."""
    
    # Smooth actions
    if 'actions' in episode_group:
        actions = np.array(episode_group['actions'])
        print(f"    Smoothing actions: {actions.shape}")
        smoothed_actions = smooth_actions(actions, method, window_size)
        del episode_group['actions']
        episode_group.create_dataset('actions', data=smoothed_actions)
    
    # Smooth processed_actions if present
    if 'processed_actions' in episode_group:
        proc_actions = np.array(episode_group['processed_actions'])
        print(f"    Smoothing processed_actions: {proc_actions.shape}")
        smoothed_proc = smooth_actions(proc_actions, method, window_size)
        del episode_group['processed_actions']
        episode_group.create_dataset('processed_actions', data=smoothed_proc)
    
    # Smooth obs/actions if present
    if 'obs' in episode_group and 'actions' in episode_group['obs']:
        obs_actions = np.array(episode_group['obs']['actions'])
        print(f"    Smoothing obs/actions: {obs_actions.shape}")
        smoothed_obs_actions = smooth_actions(obs_actions, method, window_size)
        del episode_group['obs']['actions']
        episode_group['obs'].create_dataset('actions', data=smoothed_obs_actions)
    
    # Smooth datagen_info poses
    if 'obs' in episode_group and 'datagen_info' in episode_group['obs']:
        datagen_info = episode_group['obs']['datagen_info']
        
        # Smooth eef_pose
        if 'eef_pose' in datagen_info:
            for eef_name in datagen_info['eef_pose'].keys():
                eef_poses = np.array(datagen_info['eef_pose'][eef_name])
                print(f"    Smoothing eef_pose/{eef_name}: {eef_poses.shape}")
                smoothed_eef = smooth_pose_matrix(eef_poses, method, window_size)
                del datagen_info['eef_pose'][eef_name]
                datagen_info['eef_pose'].create_dataset(eef_name, data=smoothed_eef)
        
        # Smooth target_eef_pose
        if 'target_eef_pose' in datagen_info:
            for eef_name in datagen_info['target_eef_pose'].keys():
                target_poses = np.array(datagen_info['target_eef_pose'][eef_name])
                print(f"    Smoothing target_eef_pose/{eef_name}: {target_poses.shape}")
                smoothed_target = smooth_pose_matrix(target_poses, method, window_size)
                del datagen_info['target_eef_pose'][eef_name]
                datagen_info['target_eef_pose'].create_dataset(eef_name, data=smoothed_target)
    
    # Smooth eef_pos and eef_quat in obs (if present)
    if 'obs' in episode_group:
        obs = episode_group['obs']
        
        if 'eef_pos' in obs:
            eef_pos = np.array(obs['eef_pos'])
            print(f"    Smoothing obs/eef_pos: {eef_pos.shape}")
            if method == 'moving_average':
                smoothed_eef_pos = moving_average(eef_pos, window_size)
            elif method == 'gaussian':
                smoothed_eef_pos = gaussian_smooth(eef_pos, sigma=window_size/3)
            elif method == 'savgol':
                smoothed_eef_pos = savgol_smooth(eef_pos, window_size)
            else:
                smoothed_eef_pos = eef_pos
            del obs['eef_pos']
            obs.create_dataset('eef_pos', data=smoothed_eef_pos)
        
        if 'eef_quat' in obs:
            eef_quat = np.array(obs['eef_quat'])
            print(f"    Smoothing obs/eef_quat: {eef_quat.shape}")
            smoothed_eef_quat = smooth_quaternion(eef_quat, method, window_size)
            del obs['eef_quat']
            obs.create_dataset('eef_quat', data=smoothed_eef_quat)


def analyze_jerkiness(actions: np.ndarray) -> dict:
    """Analyze how jerky the actions are."""
    if len(actions) < 3:
        return {"velocity_std": 0, "acceleration_std": 0, "jerk_std": 0}
    
    # Calculate velocity (first derivative)
    velocity = np.diff(actions, axis=0)
    
    # Calculate acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0)
    
    # Calculate jerk (third derivative)
    jerk = np.diff(acceleration, axis=0) if len(acceleration) > 1 else np.zeros_like(acceleration)
    
    return {
        "velocity_std": np.std(velocity),
        "acceleration_std": np.std(acceleration),
        "jerk_std": np.std(jerk),
    }


def main():
    parser = argparse.ArgumentParser(description="Smooth actions and poses in HDF5 dataset")
    parser.add_argument(
        "--input_file",
        type=str,
        default="./datasets/annotated_dataset.hdf5",
        help="Path to input dataset"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./datasets/smoothed_dataset.hdf5",
        help="Path to output dataset"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['moving_average', 'gaussian', 'savgol'],
        default='savgol',
        help="Smoothing method (default: savgol - best for preserving shape)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=7,
        help="Window size for smoothing (must be odd for savgol, default: 7)"
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only analyze jerkiness without smoothing"
    )
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  Dataset Smoother")
    print(f"{'='*60}")
    print(f"  Input:  {args.input_file}")
    print(f"  Output: {args.output_file}")
    print(f"  Method: {args.method}")
    print(f"  Window: {args.window_size}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        return
    
    # Analyze original dataset
    print("📊 Analyzing original dataset jerkiness...\n")
    with h5py.File(args.input_file, 'r') as f:
        if 'data' not in f:
            print("❌ Error: 'data' group not found in dataset")
            return
        
        for ep_name in list(f['data'].keys())[:3]:  # Analyze first 3 episodes
            episode = f['data'][ep_name]
            if 'actions' in episode:
                actions = np.array(episode['actions'])
                stats = analyze_jerkiness(actions)
                print(f"  {ep_name}:")
                print(f"    Velocity std:     {stats['velocity_std']:.6f}")
                print(f"    Acceleration std: {stats['acceleration_std']:.6f}")
                print(f"    Jerk std:         {stats['jerk_std']:.6f}")
    
    if args.analyze_only:
        return
    
    # Copy input to output
    print(f"\n📝 Copying dataset to {args.output_file}...")
    shutil.copy2(args.input_file, args.output_file)
    
    # Smooth the output dataset
    print(f"\n🔧 Smoothing with {args.method} (window={args.window_size})...\n")
    with h5py.File(args.output_file, 'r+') as f:
        for ep_name in f['data'].keys():
            print(f"  Processing {ep_name}...")
            smooth_episode(f['data'][ep_name], args.method, args.window_size)
    
    # Analyze smoothed dataset
    print("\n📊 Analyzing smoothed dataset jerkiness...\n")
    with h5py.File(args.output_file, 'r') as f:
        for ep_name in list(f['data'].keys())[:3]:
            episode = f['data'][ep_name]
            if 'actions' in episode:
                actions = np.array(episode['actions'])
                stats = analyze_jerkiness(actions)
                print(f"  {ep_name}:")
                print(f"    Velocity std:     {stats['velocity_std']:.6f}")
                print(f"    Acceleration std: {stats['acceleration_std']:.6f}")
                print(f"    Jerk std:         {stats['jerk_std']:.6f}")
    
    print(f"\n✅ Done! Smoothed dataset saved to: {args.output_file}")
    print("\nTo use the smoothed dataset for generation:")
    print(f"  python generate_dataset.py --input_file {args.output_file} ...")


if __name__ == "__main__":
    main()
