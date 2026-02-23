import argparse
import h5py
import numpy as np

###
# python projects/so_arm_mimic/scripts/isaaclab_mimic/remove_first_n_frames.py --input ./datasets/move_generated.hdf5 --output ./datasets/move_generated_truncated.hdf5 --n_frames 10
##

def is_dataset(obj):
    return isinstance(obj, h5py.Dataset)

def copy_and_truncate(src_group, dest_group, n_frames, orig_num_samples):
    """
    Recursively iterate through groups and datasets.
    If it is a time-series dataset (matching orig_num_samples), slice off the first n_frames.
    Otherwise, copy it directly (e.g., initial_state).
    """
    for key, item in src_group.items():
        if is_dataset(item):
            # Read data
            data = item[()]
            
            # Sub-check: only truncate if it's a time-series array 
            # (i.e., its first dimension matches orig_num_samples).
            if data.shape and data.shape[0] == orig_num_samples:
                if orig_num_samples >= n_frames:
                    truncated_data = data[n_frames:]
                else:
                    print(f"  Warning: Dataset {item.name} has fewer frames ({data.shape[0]}) than {n_frames}, setting to empty.")
                    truncated_data = data[0:0]
            else:
                # E.g. scalars, empty arrays, or static data like `initial_state` with shape [1, ...]
                truncated_data = data
            
            # Create new dataset in dest
            try:
                dest_group.create_dataset(key, data=truncated_data)
                # Copy attributes
                for attr_name, attr_value in item.attrs.items():
                    dest_group[key].attrs[attr_name] = attr_value
            except Exception as e:
                print(f"Error processing dataset {item.name}: {e}")
                
        else:
            # It's a group
            new_group = dest_group.create_group(key)
            # Copy attributes
            for attr_name, attr_value in item.attrs.items():
                new_group.attrs[attr_name] = attr_value
            # Recursively copy
            copy_and_truncate(item, new_group, n_frames, orig_num_samples)

def main():
    parser = argparse.ArgumentParser(description="Remove first N frames from all demonstrations in an HDF5 dataset.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input HDF5 file path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output HDF5 file path")
    parser.add_argument("--n_frames", "-n", type=int, required=True, help="Number of frames to remove from the start")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    n_frames = args.n_frames

    print(f"Reading {input_file} and removing first {n_frames} frames...")

    with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dest:
        # 1. Copy top-level attributes (like env_args, total_frames, metadata)
        for attr_name, attr_value in src.attrs.items():
            dest.attrs[attr_name] = attr_value

        # 2. Process data group
        if 'data' in src:
            data_group_dest = dest.create_group('data')
            for attr_name, attr_value in src['data'].attrs.items():
                data_group_dest.attrs[attr_name] = attr_value

            demo_keys = [k for k in src['data'].keys() if k.startswith('demo_')]
            # sort chronologically by numerical ID
            demo_keys = sorted(demo_keys, key=lambda x: int(x.split('_')[1]))
            
            total_frames = 0
            
            for demo_key in demo_keys:
                src_demo = src['data'][demo_key]
                dest_demo = data_group_dest.create_group(demo_key)
                
                # Copy demo attributes, update num_samples
                orig_num_samples = src_demo.attrs.get('num_samples', 0)
                new_num_samples = max(0, orig_num_samples - n_frames)
                
                for attr_name, attr_value in src_demo.attrs.items():
                    if attr_name == 'num_samples':
                        dest_demo.attrs['num_samples'] = new_num_samples
                    else:
                        dest_demo.attrs[attr_name] = attr_value
                        
                total_frames += new_num_samples

                copy_and_truncate(src_demo, dest_demo, n_frames, orig_num_samples)
                print(f"Processed {demo_key}: {orig_num_samples} -> {new_num_samples} frames")

            # Update total_frames at root data group
            if 'total' in data_group_dest.attrs:
                data_group_dest.attrs['total'] = total_frames
        
        # 3. Copy any other top-level groups (just in case they exist)
        for key, item in src.items():
            if key != 'data':
                src.copy(key, dest)

    print(f"Successfully exported truncated dataset to {output_file}")

if __name__ == "__main__":
    main()
