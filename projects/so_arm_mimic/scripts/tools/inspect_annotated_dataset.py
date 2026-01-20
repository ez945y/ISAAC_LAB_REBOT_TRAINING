#!/usr/bin/env python3
"""
Inspect Annotated Dataset for Isaac Lab Mimic

This script inspects an annotated HDF5 dataset to verify that:
1. All required datagen_info fields are present
2. subtask_term_signals are correctly recorded
3. object_poses match the expected cube names
4. eef_pose and target_eef_pose are properly recorded

Usage:
    python inspect_annotated_dataset.py --input_file /path/to/annotated_dataset.hdf5
"""

import argparse
import h5py
import numpy as np


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def inspect_dataset(input_file: str):
    """Inspect the annotated dataset."""
    print_section(f"Inspecting: {input_file}")
    
    with h5py.File(input_file, 'r') as f:
        # Check top-level structure
        print("\n📁 Top-level groups:")
        for key in f.keys():
            print(f"   - {key}")
        
        # Check data group
        if 'data' not in f.keys():
            print("❌ ERROR: 'data' group not found!")
            return
        
        data_group = f['data']
        episode_names = list(data_group.keys())
        print(f"\n📊 Number of episodes: {len(episode_names)}")
        
        for ep_idx, ep_name in enumerate(episode_names[:3]):  # Check first 3 episodes
            print_section(f"Episode: {ep_name}")
            episode = data_group[ep_name]
            
            # Check episode structure
            print("\n📁 Episode structure:")
            for key in episode.keys():
                if isinstance(episode[key], h5py.Group):
                    print(f"   📂 {key}/")
                    for subkey in episode[key].keys():
                        if isinstance(episode[key][subkey], h5py.Group):
                            print(f"      📂 {subkey}/")
                            for subsubkey in episode[key][subkey].keys():
                                item = episode[key][subkey][subsubkey]
                                if isinstance(item, h5py.Dataset):
                                    print(f"         📄 {subsubkey}: shape={item.shape}, dtype={item.dtype}")
                                else:
                                    print(f"         📂 {subsubkey}/")
                        else:
                            item = episode[key][subkey]
                            if isinstance(item, h5py.Dataset):
                                print(f"      📄 {subkey}: shape={item.shape}, dtype={item.dtype}")
                else:
                    item = episode[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"   📄 {key}: shape={item.shape}, dtype={item.dtype}")
            
            # Check obs/datagen_info
            if 'obs' in episode and 'datagen_info' in episode['obs']:
                datagen_info = episode['obs']['datagen_info']
                print("\n✅ datagen_info found!")
                
                # Check object_pose
                if 'object_pose' in datagen_info:
                    object_pose = datagen_info['object_pose']
                    print(f"\n📦 object_pose objects:")
                    for obj_name in object_pose.keys():
                        obj_data = object_pose[obj_name]
                        print(f"   - {obj_name}: shape={obj_data.shape}")
                        # Print first pose
                        if obj_data.shape[0] > 0:
                            first_pose = np.array(obj_data[0])
                            print(f"     First pose position: {first_pose[:3, 3] if first_pose.ndim == 2 else 'N/A'}")
                else:
                    print("❌ object_pose not found in datagen_info!")
                
                # Check eef_pose
                if 'eef_pose' in datagen_info:
                    eef_pose = datagen_info['eef_pose']
                    print(f"\n🤖 eef_pose:")
                    for eef_name in eef_pose.keys():
                        eef_data = eef_pose[eef_name]
                        print(f"   - {eef_name}: shape={eef_data.shape}")
                else:
                    print("❌ eef_pose not found in datagen_info!")
                
                # Check target_eef_pose
                if 'target_eef_pose' in datagen_info:
                    target_eef_pose = datagen_info['target_eef_pose']
                    print(f"\n🎯 target_eef_pose:")
                    for eef_name in target_eef_pose.keys():
                        target_data = target_eef_pose[eef_name]
                        print(f"   - {eef_name}: shape={target_data.shape}")
                else:
                    print("❌ target_eef_pose not found in datagen_info!")
                
                # Check subtask_term_signals
                if 'subtask_term_signals' in datagen_info:
                    subtask_signals = datagen_info['subtask_term_signals']
                    print(f"\n🏁 subtask_term_signals:")
                    for signal_name in subtask_signals.keys():
                        signal_data = np.array(subtask_signals[signal_name])
                        num_true = np.sum(signal_data)
                        first_true_idx = np.argmax(signal_data) if num_true > 0 else -1
                        print(f"   - {signal_name}: shape={signal_data.shape}, first_true_idx={first_true_idx}, num_true={num_true}")
                else:
                    print("❌ subtask_term_signals not found in datagen_info!")
            else:
                print("❌ ERROR: obs/datagen_info not found!")
            
            # Check actions
            if 'actions' in episode:
                actions = np.array(episode['actions'])
                print(f"\n🎮 actions: shape={actions.shape}")
                print(f"   First action: {actions[0]}")
                print(f"   Last action: {actions[-1]}")
            
            print("")
        
        if len(episode_names) > 3:
            print(f"\n... and {len(episode_names) - 3} more episodes")


def main():
    parser = argparse.ArgumentParser(description="Inspect annotated dataset for Isaac Lab Mimic")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="./datasets/annotated_dataset.hdf5",
        help="Path to the annotated dataset file"
    )
    args = parser.parse_args()
    
    inspect_dataset(args.input_file)


if __name__ == "__main__":
    main()
