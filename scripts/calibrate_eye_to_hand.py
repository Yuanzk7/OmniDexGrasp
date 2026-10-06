#!/usr/bin/env python3
"""Eye-to-hand calibration: fixed RealSense + ChArUco board on the xArm7 hand -> T_base_camera.

Moves the arm through wrist-rotation (J5/J6/J7) variations of a base pose, records
(T_base_tcp, T_cam_board) at each, and solves T_base_camera (Park & Martin). Keep a hand on the E-stop.

    conda activate xarm-teleop
    python scripts/calibrate_eye_to_hand.py --base -8.1 -25.2 7.8 21.2 4.9 46.3 53.2
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

# Physical board: DICT_4X4_50, 4x5 squares, 30 mm squares, 22 mm markers
BOARD = cv2.aruco.CharucoBoard((4, 5), 0.030, 0.022, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
# (dJ5, dJ6, dJ7) degrees. J6 limited to [-25, +6]: more + tilts the fingertips toward the table.
DELTAS = [
    (0, 0, 0), (0, 0, 20), (0, 0, -20), (0, 0, 35), (0, 0, -35),
    (0, 6, 0), (0, -12, 0), (0, -25, 0),
    (15, 0, 0), (-15, 0, 0), (25, 0, 0), (-25, 0, 0),
    (15, 5, 20), (-15, 5, -20), (15, -15, -20), (-15, -15, 20),
    (20, -8, 35), (-20, -8, -35), (0, -20, 30), (10, 6, -30),
]
SPEED = 15  # deg/s


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.ravel(t); return M


def board_pose(image, K):
    """T_cam_board from a ChArUco detection (>= 6 corners), else None."""
    corners, ids, _, _ = cv2.aruco.CharucoDetector(BOARD).detectBoard(image)
    if ids is None or len(ids) < 6:
        return None
    obj, img = BOARD.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    return T(cv2.Rodrigues(rvec)[0], tvec) if ok else None


def hand_eye(G, C):
    """Solve G_i X C_i = const for X (Park & Martin): pairs give A X = X B with
    A = inv(G_j) G_i, B = C_j inv(C_i); rotation from log-maps, translation by least squares."""
    A = [np.linalg.inv(G[j]) @ G[i] for i in range(len(G)) for j in range(i + 1, len(G))]
    B = [C[j] @ np.linalg.inv(C[i]) for i in range(len(C)) for j in range(i + 1, len(C))]
    log = lambda R: cv2.Rodrigues(R)[0].ravel()
    M = sum(np.outer(log(b[:3, :3]), log(a[:3, :3])) for a, b in zip(A, B))
    w, V = np.linalg.eigh(M.T @ M)
    R = (V @ np.diag(w ** -0.5) @ V.T) @ M.T
    lhs = np.vstack([a[:3, :3] - np.eye(3) for a in A])
    rhs = np.concatenate([R @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    return T(R, np.linalg.lstsq(lhs, rhs, rcond=None)[0])


def solve(T_base_tcp, T_cam_board):
    """Eye-to-hand: the board is rigid on the TCP, so inv(T_base_tcp) · T_base_cam · T_cam_board is constant."""
    T_base_cam = hand_eye([np.linalg.inv(M) for M in T_base_tcp], T_cam_board)
    tcp_board = [np.linalg.inv(a) @ T_base_cam @ b for a, b in zip(T_base_tcp, T_cam_board)]
    return T_base_cam, float(np.linalg.norm(np.std([M[:3, 3] for M in tcp_board], axis=0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=float, nargs=7, required=True, metavar="J", help="J1..J7 in degrees")
    p.add_argument("--ip", default="192.168.1.216")
    p.add_argument("--out", default="calibration/eye_to_hand.json")
    args = p.parse_args()
    base = np.array(args.base)

    import pyrealsense2 as rs
    from xarm.wrapper import XArmAPI

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    intr = pipe.start(cfg).get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    arm = XArmAPI(args.ip)
    arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)
    input(f"{len(DELTAS)} poses at {SPEED} deg/s. Hand on E-stop, press Enter to start")

    T_base_tcp, T_cam_board = [], []
    try:
        for i, (d5, d6, d7) in enumerate(DELTAS):
            q = base + np.array([0, 0, 0, 0, d5, d6, d7])
            if arm.set_servo_angle(angle=q.tolist(), speed=SPEED, mvacc=50, wait=True) != 0:
                print(f"[{i}] move failed; stopping"); break
            time.sleep(1.0)
            for _ in range(5):                                   # drop buffered frames
                frame = pipe.wait_for_frames().get_color_frame()
            Tcb = board_pose(np.asanyarray(frame.get_data()), K)
            if Tcb is None:
                print(f"[{i}] board not detected, skipped"); continue
            code, pose = arm.get_position_aa(is_radian=True)      # [x y z mm, rx ry rz rad]
            if code != 0:
                print(f"[{i}] get_position_aa failed; stopping"); break
            T_base_tcp.append(T(cv2.Rodrigues(np.array(pose[3:6]))[0], np.array(pose[:3]) / 1000.0))
            T_cam_board.append(Tcb)
            print(f"[{i}] recorded")
        arm.set_servo_angle(angle=base.tolist(), speed=SPEED, mvacc=50, wait=True)
    finally:
        pipe.stop(); arm.disconnect()

    if len(T_base_tcp) < 5:
        raise SystemExit(f"need at least 5 samples, got {len(T_base_tcp)}")
    T_base_cam, err = solve(T_base_tcp, T_cam_board)
    print("T_base_camera =\n", np.round(T_base_cam, 4))
    print(f"camera position in base (m): {np.round(T_base_cam[:3, 3], 4)}  |  consistency std: {err * 1000:.1f} mm")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"convention": "p_base = T_base_camera @ p_camera (RealSense color frame, m)",
                               "T_base_camera": T_base_cam.tolist(), "samples": len(T_base_tcp),
                               "consistency_std_m": err}, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
