#!/usr/bin/env python3
"""Execute the OmniDexGrasp result on xArm7 + Inspire RH56F1, one confirmed step at a time.

Reads out/<task>/robo_cam.json (from scripts.robo_to_camera) and robo.json:
  1. open hand, move TCP to a pre-grasp pose (backed off along the finger direction, raised)
  2. move TCP to the grasp pose (optionally lifted by --lift)
  3. close the fingers to the retargeted joints (x --close-scale)
  4. lift straight up, then release on request
Every move waits for Enter. Keep a hand on the E-stop.

    conda activate xarm-teleop
    python scripts/execute_grasp.py --task toilet_paper_2 --lift 0.03
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from grasp_from_robo import CMD_OPEN, robo_to_cmd, ramp, write_angles  # noqa: E402


def to_aa(T):
    """4x4 (m) -> xArm pose [x y z mm, rx ry rz rad]."""
    return [*(T[:3, 3] * 1000), *Rotation.from_matrix(T[:3, :3]).as_rotvec()]


def plan(task_out, lift, back, up):
    rc = json.load(open(task_out / "robo_cam.json"))
    T_grasp = np.array(rc["T_base_tcp"]); T_grasp[2, 3] += lift
    finger_dir = -np.array(rc["T_base_hand"])[:3, 2]                    # hand URDF -z = finger direction
    T_pre = T_grasp.copy(); T_pre[:3, 3] += -back * finger_dir + [0, 0, up]
    T_up = T_grasp.copy(); T_up[2, 3] += 0.05
    return T_pre, T_grasp, T_up


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--output", default="out")
    p.add_argument("--lift", type=float, default=0.03, help="raise the whole grasp pose by this much (m)")
    p.add_argument("--back", type=float, default=0.0, help="pre-grasp offset against the finger direction (m); 0 = straight above")
    p.add_argument("--up", type=float, default=0.08, help="pre-grasp extra height (m)")
    p.add_argument("--close-scale", type=float, default=1.0)
    p.add_argument("--speed", type=float, default=30, help="TCP speed mm/s")
    p.add_argument("--ip", default="192.168.1.216")
    p.add_argument("--port", default="/dev/ttyUSB0")
    args = p.parse_args()
    task_out = Path(args.output) / args.task

    T_pre, T_grasp, T_up = plan(task_out, args.lift, args.back, args.up)
    pose = json.load(open(task_out / "robo.json"))["inspire"]["final"]
    _, ratio, hand_cmd = robo_to_cmd(pose, args.close_scale)
    for name, T in (("pre-grasp", T_pre), ("grasp", T_grasp), ("lift", T_up)):
        print(f"{name:10} TCP (mm): {np.round(T[:3, 3] * 1000, 1)}")
    print("hand close ratios:", np.round(ratio, 2))

    import serial
    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.ip)
    for name, T in (("pre-grasp", T_pre), ("grasp", T_grasp), ("lift", T_up)):
        code, _ = arm.get_inverse_kinematics(to_aa(T), input_is_radian=True)
        if code != 0:
            arm.disconnect(); raise SystemExit(f"no IK solution for the {name} pose (code {code})")
    print("IK ok for all poses")

    def move(T, label):
        input(f"Enter -> move to {label}")
        code = arm.set_position_aa(to_aa(T), speed=args.speed, mvacc=100, is_radian=True, wait=True)
        print(f"  {label}: code {code}")
        if code != 0:
            raise SystemExit("move failed; stopping")

    with serial.Serial(args.port, 115200, timeout=0.2) as ser:
        write_angles(ser, CMD_OPEN); time.sleep(1.0)                     # start open
        arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)
        try:
            move(T_pre, "pre-grasp")
            move(T_grasp, "grasp pose")
            input("Enter -> close fingers")
            ramp(ser, CMD_OPEN, hand_cmd)
            move(T_up, "lift")
            input("Enter -> release (open hand)")
            write_angles(ser, CMD_OPEN); time.sleep(1.0)
            move(T_pre, "pre-grasp (retreat)")
        finally:
            arm.disconnect()


if __name__ == "__main__":
    main()
