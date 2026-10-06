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
p.add_argument("--calib", default="../calibration/eye_to_hand.json", help="T_base_camera json (optional)")
args = p.parse_args()
task_out = Path(args.output) / args.task

pose = json.load(open(task_out / "robo.json"))[args.hand]["final"]   # [tx ty tz, rx ry rz, joints...]
T_cam_obj = np.array(json.load(open(task_out / "pose_est.json"))["scene"]["pose"])

T_obj_hand = np.eye(4)
T_obj_hand[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
T_obj_hand[:3, 3] = pose[:3]
T_cam_hand = T_cam_obj @ T_obj_hand

result = {
    "hand_type": args.hand,
    "T_cam_hand": T_cam_hand.tolist(),   # hand URDF root link in the RealSense color frame (OpenCV axes, m)
    "joints": pose[6:],
}
print("hand root (cam):", np.round(T_cam_hand[:3, 3], 3))
if Path(args.calib).exists():
    T_base_cam = np.array(json.load(open(args.calib))["T_base_camera"])
    result["T_base_hand"] = (T_base_cam @ T_cam_hand).tolist()   # hand root in the robot base frame (m)
    print("hand root (base):", np.round(result["T_base_hand"], 3)[:3, 3])
    mount = Path(args.calib).with_name("hand_mount.json")
    if mount.exists():                                   # hand root -> xArm TCP target
        T_tcp_hand = np.array(json.load(open(mount))["T_tcp_hand"])
        T_base_tcp = np.array(result["T_base_hand"]) @ np.linalg.inv(T_tcp_hand)
        result["T_base_tcp"] = T_base_tcp.tolist()
        result["xarm_tcp_aa_mm_rad"] = [*(T_base_tcp[:3, 3] * 1000), *Rotation.from_matrix(T_base_tcp[:3, :3]).as_rotvec()]
        print("xArm TCP target (mm, rad):", np.round(result["xarm_tcp_aa_mm_rad"], 3))
(task_out / "robo_cam.json").write_text(json.dumps(result, indent=2))
