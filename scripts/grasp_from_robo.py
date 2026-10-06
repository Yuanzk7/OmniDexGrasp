#!/usr/bin/env python3
"""Drive the Inspire RH56F1 to the grasp in out/<task>/robo.json (12 URDF joints -> 6 motors).

Standalone: talks Modbus RTU over RS485 directly (115200 8N1, slave id 1, ANGLE_SET register 1040).

    python scripts/grasp_from_robo.py out/toilet_paper_1/robo.json   # prints the 6 motor values, Enter moves the hand
"""
import argparse
import json
import time

import numpy as np

# robo.json joints (after the 6 wrist values) follow the URDF order:
#   0 thumb_1(rot) 1 thumb_2(bend) 2 thumb_3* 3 thumb_4* 4 index_1 5 index_2* 6 middle_1 7 middle_2*
#   8 ring_1 9 ring_2* 10 little_1 11 little_2*          (* = mimic joint, no motor)
# RH56 motor order: little, ring, middle, index, thumb_bend, thumb_rot
NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
JOINT_IDX = [10, 8, 6, 4, 1, 0]
JOINT_UPPER = np.array([1.60, 1.60, 1.60, 1.60, 0.55, 1.15])        # URDF upper limits (rad)
CMD_OPEN = np.array([1740, 1740, 1740, 1740, 1350, 1500])            # device units, verified on this hand
CMD_CLOSED = np.array([900, 900, 900, 900, 900, 1650])
REG_ANGLE_SET, HAND_ID = 1040, 1


def crc16(data: bytes) -> bytes:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc.to_bytes(2, "little")


def write_angles(ser, values) -> None:
    """Modbus function 0x10: write 6 holding registers (big-endian int16)."""
    data = b"".join(int(v).to_bytes(2, "big", signed=True) for v in values)
    body = bytes([HAND_ID, 0x10]) + REG_ANGLE_SET.to_bytes(2, "big") + (6).to_bytes(2, "big") + bytes([12]) + data
    ser.reset_input_buffer()
    ser.write(body + crc16(body))
    ser.read(8)  # ack


def robo_to_cmd(pose, close_scale=1.0):
    """robo.json 'final' (18) -> 6 RH56 device commands; close_scale > 1 closes the fingers further."""
    rad = np.array(pose[6:])[JOINT_IDX]
    ratio = np.clip(rad / JOINT_UPPER * close_scale, 0.0, 1.0)            # 0 = open, 1 = fully closed
    return rad, ratio, np.rint(CMD_OPEN + ratio * (CMD_CLOSED - CMD_OPEN)).astype(int)


def ramp(ser, start, target, steps=20):
    for w in np.linspace(0, 1, steps + 1):
        write_angles(ser, np.rint(start + w * (target - start)))
        time.sleep(0.1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("robo_json")
    p.add_argument("--hand-type", default="inspire")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--steps", type=int, default=20, help="interpolation steps from open to grasp")
    p.add_argument("--close-scale", type=float, default=1.0, help="multiply closing ratios (e.g. 1.3 for a firmer grip)")
    args = p.parse_args()

    pose = json.load(open(args.robo_json))[args.hand_type]["final"]
    rad, ratio, target = robo_to_cmd(pose, args.close_scale)
    for n, r, v, c in zip(NAMES, rad, ratio, target):
        print(f"{n:11} {r:6.3f} rad  ratio {v:.2f}  cmd {c}")
    input("press Enter to move the hand (Ctrl-C to abort)")

    import serial
    with serial.Serial(args.port, 115200, timeout=0.2) as ser:
        ramp(ser, CMD_OPEN, target, args.steps)
        input("grasp pose reached; press Enter to open")
        write_angles(ser, CMD_OPEN)
