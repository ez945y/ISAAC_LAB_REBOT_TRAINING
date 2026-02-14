"""
檢查 LeRobot 資料集中各關節在指定 episode 的上下限（degree + radian）
"""
import os, math, argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MikeChenYZ/soarm-fmb-v2")
parser.add_argument("--episode", type=int, default=0, help="episode index (-1 = all episodes)")
args = parser.parse_args()

dataset = LeRobotDataset(args.dataset)

actions = []   # degree
obs = []       # degree

for i in range(len(dataset)):
    sample = dataset[i]
    ep = sample["episode_index"].item()
    if args.episode >= 0:
        if ep < args.episode:
            continue
        if ep > args.episode:
            break
    actions.append(sample["action"].tolist())
    obs.append(sample["observation.state"].tolist())

import torch
actions_t = torch.tensor(actions)  # [N, 6] degree
obs_t = torch.tensor(obs)          # [N, 6] degree

print(f"\nDataset: {args.dataset}")
print(f"Episode: {'ALL' if args.episode < 0 else args.episode}")
print(f"Frames:  {len(actions_t)}\n")

header = f"{'Joint':<16} | {'action min':>12} {'action max':>12} | {'obs min':>12} {'obs max':>12} |  {'act min(rad)':>12} {'act max(rad)':>12} | {'obs min(rad)':>12} {'obs max(rad)':>12}"
print(header)
print("-" * len(header))

for j, name in enumerate(JOINT_NAMES):
    a_min = actions_t[:, j].min().item()
    a_max = actions_t[:, j].max().item()
    o_min = obs_t[:, j].min().item()
    o_max = obs_t[:, j].max().item()
    print(
        f"{name:<16} | "
        f"{a_min:>10.4f}° {a_max:>10.4f}° | "
        f"{o_min:>10.4f}° {o_max:>10.4f}° |  "
        f"{math.radians(a_min):>10.4f}  {math.radians(a_max):>10.4f}  | "
        f"{math.radians(o_min):>10.4f}  {math.radians(o_max):>10.4f}"
    )
