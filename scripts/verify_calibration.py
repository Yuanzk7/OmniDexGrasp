#!/usr/bin/env python3
"""Check T_base_camera against a reconstructed object: print the object center in the robot base frame
and, with --move, bring the xArm TCP to a point straight above it (orientation unchanged).

    conda activate xarm-teleop
    python scripts/verify_calibration.py --task toilet_paper_2            # numbers only
    python scripts/verify_calibration.py --task toilet_paper_2 --move     # also move TCP above the object
"""
import argparse
import json

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--calib", default="calibration/eye_to_hand.json")
p.add_argument("--ip", default="192.168.1.216")
p.add_argument("--height", type=float, default=0.15, help="TCP height above the object center (m)")
p.add_argument("--move", action="store_true")
args = p.parse_args()

T_cam_obj = np.array(json.load(open(f"out/{args.task}/pose_est.json"))["scene"]["pose"])
T_base_cam = np.array(json.load(open(args.calib))["T_base_camera"])
obj_base = (T_base_cam @ T_cam_obj)[:3, 3]
print(f"object center  cam  (m): {np.round(T_cam_obj[:3, 3], 3)}")
print(f"object center  base (m): {np.round(obj_base, 3)}")
target = obj_base + [0, 0, args.height]
print(f"TCP target     base (m): {np.round(target, 3)}  ({args.height * 100:.0f} cm above the object center)")
if not args.move:
    raise SystemExit("add --move to send the TCP there")

from xarm.wrapper import XArmAPI
arm = XArmAPI(args.ip)
code, pose = arm.get_position_aa(is_radian=True)           # [x y z mm, rx ry rz rad]
print(f"TCP now        base (m): {np.round(np.array(pose[:3]) / 1000, 3)}")
input("Enter to move (keeps current orientation, 30 mm/s; hand on E-stop)")
arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)
goal = [target[0] * 1000, target[1] * 1000, target[2] * 1000, *pose[3:6]]
code = arm.set_position_aa(goal, speed=30, mvacc=100, is_radian=True, wait=True)
print("move result code:", code)
arm.disconnect()
