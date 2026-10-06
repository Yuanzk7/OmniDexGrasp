#!/usr/bin/env python3
"""📷 Capture scene_image.png / depth.png / camera.yaml from a RealSense into datasets/{task}/.

    python scripts/capture_realsense.py --task RedMug_4 --obj "red mug"

Preview keys: [space] capture, [q] quit.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

W, H, FPS = 640, 480, 30


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="task folder name, e.g. RedMug_4")
    p.add_argument("--obj", required=True, help="obj_description for GSAM, e.g. 'red mug'")
    p.add_argument("--datasets", type=Path, default=Path(__file__).resolve().parents[1] / "datasets")
    args = p.parse_args()

    # Color + depth streams; depth is aligned to color so it shares the color intrinsics.
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = pipeline.start(config)
    depth_unit_mm = profile.get_device().first_depth_sensor().get_depth_scale() * 1000
    align = rs.align(rs.stream.color)

    try:
        for _ in range(30):  # let auto-exposure settle
            pipeline.wait_for_frames()
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            color = np.asanyarray(color_frame.get_data())
            depth_mm = (np.asanyarray(depth_frame.get_data()) * depth_unit_mm).round().astype(np.uint16)

            cv2.imshow("capture", color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
            if key == ord(" "):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
    task_dir = args.datasets / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(task_dir / "scene_image.png"), color)
    cv2.imwrite(str(task_dir / "depth.png"), depth_mm)
    with open(task_dir / "camera.yaml", "w") as f:
        yaml.safe_dump({
            "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
            "width": intr.width, "height": intr.height, "coeffs": list(intr.coeffs),
            "obj_description": args.obj,
        }, f)
    print(f"💾 saved to {task_dir}")


if __name__ == "__main__":
    main()
