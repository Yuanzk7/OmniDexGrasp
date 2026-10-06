"""T_cam_hand = T_cam_obj (pose_est.json "scene") @ T_obj_hand (robo.json wrist: xyz + axis-angle).

    python -m scripts.robo_to_camera --task toilet_paper_1   # -> out/<task>/robo_cam.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--hand", default="inspire")
p.add_argument("--output", default="../out")
args = p.parse_args()
task_out = Path(args.output) / args.task

pose = json.load(open(task_out / "robo.json"))[args.hand]["final"]   # [tx ty tz, rx ry rz, joints...]
T_cam_obj = np.array(json.load(open(task_out / "pose_est.json"))["scene"]["pose"])

T_obj_hand = np.eye(4)
T_obj_hand[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
T_obj_hand[:3, 3] = pose[:3]
T_cam_hand = T_cam_obj @ T_obj_hand

(task_out / "robo_cam.json").write_text(json.dumps({
    "hand_type": args.hand,
    "T_cam_hand": T_cam_hand.tolist(),   # hand URDF root link in the RealSense color frame (OpenCV axes, m)
    "joints": pose[6:],
}, indent=2))
print("hand root (cam):", np.round(T_cam_hand[:3, 3], 3))
