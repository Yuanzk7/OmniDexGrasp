#!/usr/bin/env python3
"""SAM 3D Objects mesh -> Text2HOI-ready object mesh, sized from the captured depth.

Text2HOI was trained on H2O meshes: a single outer surface, longest axis along z,
bounding-box center at the origin, meters. SAM 3D gives a hollow double-walled shell
in an arbitrary scale. This script makes the four properties hold:
  1. keep the largest connected piece (drops floating debris)
  2. outer surface only: convex hull (correct for cups, bottles, cans; NOT for
     concave shapes such as mugs with handles -> use --no-hull and clean manually)
  3. rotate so the longest bounding-box axis is z; --flip-z turns it upside down
  4. scale so the z extent equals the object's real longest extent, center at the bbox center
The result has >= 1024 vertices (Text2HOI samples 1024 points from the vertices).

The real size is estimated from the capture itself, the same way Stage 1 does it:
datasets/<task>/mask.png (the SAM 3 mask that was used for SAM 3D Objects) selects the
object's pixels in datasets/<task>/depth.png, those are unprojected with camera.yaml,
statistically denoised, and the longest 1-99 percentile extent of the minimal oriented
bounding box is taken. No ruler is needed.

    conda activate omnidexgrasp
    python scripts/sam3d_to_text2hoi_mesh.py --task <task>            # datasets/<task>/sam3d_raw.obj -> datasets/<task>/mesh/text2hoi.obj
    python scripts/sam3d_to_text2hoi_mesh.py --task <task> --src /tmp/raw.obj
Then in Text2HOI: obj_path=<dst> obj_scale=1.0
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "omnidexgrasp"))  # utils.* live in the omnidexgrasp package root
from utils.camera import CameraIntrinsics  # noqa: E402
from utils.pointcloud import denoise_pointcloud, depth_to_pointcloud  # noqa: E402

# Same depth / denoise settings as omnidexgrasp/cfg/recons.yaml (scale: block)
DEPTH_SCALE = 0.001
MAX_DEPTH_M = 3.0
EDGE_ERODE_PX = 3
STAT_NB_NEIGHBORS = 80
STAT_STD_RATIO = 2.0
MIN_POINTS = 200


def estimate_longest_extent(task_dir: Path) -> tuple[float, int]:
    """Longest real-world extent (m) of the masked object from depth.png, and the point count used."""
    for name in ("mask.png", "depth.png", "camera.yaml"):
        if not (task_dir / name).exists():
            raise SystemExit(f"missing {task_dir / name} (mask.png = the SAM 3 mask saved by scripts/sam3_mask.py)")
    cam = yaml.safe_load(open(task_dir / "camera.yaml"))
    camera = CameraIntrinsics(fx=cam["fx"], fy=cam["fy"], ppx=cam["ppx"], ppy=cam["ppy"],
                              width=cam["width"], height=cam["height"])
    mask = cv2.imread(str(task_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)
    if mask.sum() == 0:
        raise SystemExit(f"{task_dir / 'mask.png'} is empty")

    pts = depth_to_pointcloud(task_dir / "depth.png", mask, camera,
                              depth_scale=DEPTH_SCALE, max_depth_m=MAX_DEPTH_M, edge_erode_px=EDGE_ERODE_PX)
    pts = denoise_pointcloud(pts, STAT_NB_NEIGHBORS, STAT_STD_RATIO)
    if pts.shape[0] < MIN_POINTS:
        raise SystemExit(f"only {pts.shape[0]} depth points inside the mask; check mask.png / depth.png")

    # Minimal OBB axes as in Stage 1, but a 1-99 percentile span instead of min/max so a few
    # flying pixels at the silhouette cannot stretch the estimate.
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    obb = pcd.get_minimal_oriented_bounding_box()
    proj = (pts - np.asarray(obb.center)) @ np.asarray(obb.R)
    span = np.percentile(proj, 99, axis=0) - np.percentile(proj, 1, axis=0)
    return float(span.max()), int(pts.shape[0])


p = argparse.ArgumentParser()
p.add_argument("--task", required=True, help="task folder name under --datasets")
p.add_argument("--datasets", default=REPO / "datasets", type=Path)
p.add_argument("--src", default=None, help="SAM 3D Objects mesh (default datasets/<task>/sam3d_raw.obj)")
p.add_argument("--dst", default=None, help="output mesh (default datasets/<task>/mesh/text2hoi.obj)")
p.add_argument("--no-hull", action="store_true", help="keep the full mesh instead of the convex hull")
p.add_argument("--flip-z", action="store_true", help="rotate 180 deg about x after aligning (if it ends upside down)")
args = p.parse_args()

task_dir = args.datasets / args.task
src = Path(args.src) if args.src else task_dir / "sam3d_raw.obj"
dst = Path(args.dst) if args.dst else task_dir / "mesh" / "text2hoi.obj"
dst.parent.mkdir(parents=True, exist_ok=True)

height, n_pts = estimate_longest_extent(task_dir)
print(f"object longest extent from depth: {height * 1000:.1f} mm  ({n_pts} points inside the SAM 3 mask)")

m = trimesh.load(str(src), force="mesh", process=False)
parts = m.split(only_watertight=False)
if len(parts) > 1:
    m = max(parts, key=lambda q: len(q.faces))
if not args.no_hull:
    m = m.convex_hull

# longest axis -> z (rotations only, no reflection)
longest = int(m.extents.argmax())
if longest == 1:
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))   # y -> z
elif longest == 0:
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))    # x -> z
if args.flip_z:
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

m.apply_scale(height / m.extents[2])
m.vertices -= m.bounding_box.centroid

# densify: the hull's own vertices all sit on silhouette edges (e.g. a cup's rims), so the
# side walls would never be sampled. Always subdivide to ~1/50 of the object size (2 mm for a
# 10 cm cup); Text2HOI then picks 1024 of these vertices by farthest-point sampling.
v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=m.extents.max() / 50)
out = trimesh.Trimesh(v, f, process=False)
out.export(str(dst))

c = out.bounding_box.centroid
outward = float((np.einsum("ij,ij->i", out.vertex_normals, out.vertices - c) > 0).mean())
print(f"wrote {dst}: V={len(out.vertices)} F={len(out.faces)} extents(m)={out.extents.round(4)} "
      f"longest={'xyz'[int(out.extents.argmax())]} outward_normals={outward:.2f} watertight={out.is_watertight}")
print("check the up/down direction (e.g. cup opening at +z); rerun with --flip-z if needed")
print("if the printed extent looks wrong, the mask is leaking (compare against the object by eye)")
