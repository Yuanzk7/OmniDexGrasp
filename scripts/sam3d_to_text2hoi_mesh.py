#!/usr/bin/env python3
"""SAM 3D Objects mesh -> Text2HOI-ready object mesh.

Text2HOI was trained on H2O meshes: a single outer surface, longest axis along z,
bounding-box center at the origin, meters. SAM 3D gives a hollow double-walled shell
in an arbitrary scale. This script makes the four properties hold:
  1. keep the largest connected piece (drops floating debris)
  2. outer surface only: convex hull (correct for cups, bottles, cans; NOT for
     concave shapes such as mugs with handles -> use --no-hull and clean manually)
  3. rotate so the longest bounding-box axis is z; --flip-z turns it upside down
  4. scale so the z extent equals --height (meters), center at the bbox center
The result has >= 1024 vertices (Text2HOI samples 1024 points from the vertices).

    conda activate omnidexgrasp
    python scripts/sam3d_to_text2hoi_mesh.py datasets/<task>/sam3d_raw.obj datasets/<task>/mesh/text2hoi.obj --height 0.10
Then in Text2HOI: obj_path=<that file> obj_scale=1.0
"""
import argparse
import numpy as np
import trimesh

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--height", type=float, required=True, help="real object size along its longest axis (m)")
p.add_argument("--no-hull", action="store_true", help="keep the full mesh instead of the convex hull")
p.add_argument("--flip-z", action="store_true", help="rotate 180 deg about x after aligning (if it ends upside down)")
args = p.parse_args()

m = trimesh.load(args.src, force="mesh", process=False)
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

m.apply_scale(args.height / m.extents[2])
m.vertices -= m.bounding_box.centroid

# densify: the hull's own vertices all sit on silhouette edges (e.g. a cup's rims), so the
# side walls would never be sampled. Always subdivide to ~1/50 of the object size (2 mm for a
# 10 cm cup); Text2HOI then picks 1024 of these vertices by farthest-point sampling.
v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=m.extents.max() / 50)
out = trimesh.Trimesh(v, f, process=False)
out.export(args.dst)

c = out.bounding_box.centroid
outward = float((np.einsum("ij,ij->i", out.vertex_normals, out.vertices - c) > 0).mean())
print(f"wrote {args.dst}: V={len(out.vertices)} F={len(out.faces)} extents(m)={out.extents.round(4)} "
      f"longest={'xyz'[int(out.extents.argmax())]} outward_normals={outward:.2f} watertight={out.is_watertight}")
print("check the up/down direction (e.g. cup opening at +z); rerun with --flip-z if needed")
