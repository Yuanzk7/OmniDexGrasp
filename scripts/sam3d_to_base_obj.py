"""SAM 3D Objects raw mesh -> OmniDexGrasp base.obj / material.mtl / shaded.png (+ check renders).

usage: python postprocess_mesh.py <raw.obj> <task_dir> <render_dir> [target_faces=60000]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

raw_path, task_dir, render_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
target_faces = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
render_dir.mkdir(parents=True, exist_ok=True)

# ── load + keep the largest connected piece (drops floating debris) ─────────
raw = trimesh.load(str(raw_path), force="mesh", process=False)
parts = raw.split(only_watertight=False)
keep = [p for p in parts if len(p.faces) >= 0.05 * len(raw.faces)]   # drop debris < 5% of faces
mesh = trimesh.util.concatenate(keep) if len(keep) > 1 else (keep[0] if keep else raw)
print(f"raw: V={len(raw.vertices)} F={len(raw.faces)} components={len(parts)} "
      f"sizes={sorted((len(p.faces) for p in parts), reverse=True)[:6]} -> kept {len(keep)} comps, F={len(mesh.faces)}")

# ── decimate to ~target faces with open3d quadric decimation ────────────────
o3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                               o3d.utility.Vector3iVector(mesh.faces))
o3 = o3.simplify_quadric_decimation(target_number_of_triangles=target_faces)
o3.remove_degenerate_triangles(); o3.remove_duplicated_vertices(); o3.remove_unreferenced_vertices()
mesh = trimesh.Trimesh(np.asarray(o3.vertices), np.asarray(o3.triangles), process=False)
mesh.vertices -= mesh.bounding_box.centroid
print(f"decimated: V={len(mesh.vertices)} F={len(mesh.faces)} extents={mesh.extents.round(3)} "
      f"watertight={mesh.is_watertight}")

# ── write base.obj referencing material.mtl / shaded.png like the shipped samples ──
task_dir.mkdir(parents=True, exist_ok=True)
uv = np.full((len(mesh.vertices), 2), 0.5)              # flat UV: every vertex samples the same texel
with open(task_dir / "base.obj", "w") as f:
    f.write("mtllib material.mtl\nusemtl material_0\no output\n")
    for v in mesh.vertices:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for t in uv:
        f.write(f"vt {t[0]:.4f} {t[1]:.4f}\n")
    for a, b, c in mesh.faces + 1:
        f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
(task_dir / "material.mtl").write_text(
    "newmtl material_0\nKa 0.1 0.1 0.1\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\nNs 10\nd 1.0\nillum 2\nmap_Kd shaded.png\n")
Image.new("RGB", (64, 64), (235, 235, 235)).save(task_dir / "shaded.png")   # plain white roll
print("wrote:", [p.name for p in sorted(task_dir.glob("base.obj")) + sorted(task_dir.glob("material.mtl")) + sorted(task_dir.glob("shaded.png"))])

# ── sanity: the pipeline's own loader path ─────────────────────────────────
chk = trimesh.load(str(task_dir / "base.obj"), force="mesh")
print(f"reload via trimesh: F={len(chk.faces)} visual={type(chk.visual).__name__}")

# ── render 4 views (normals-style shading) for visual inspection ───────────
import pyrender
scene_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
r = pyrender.OffscreenRenderer(480, 480)
cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(40))
dist = mesh.extents.max() * 2.2
views = {"front": (0, 0), "side": (0, 90), "top": (80, 0), "back_below": (-30, 180)}
for name, (elev, azim) in views.items():
    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.3, 0.3, 0.3])
    scene.add(scene_mesh)
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    eye = dist * np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    fwd = -eye / np.linalg.norm(eye)
    up = np.array([0, 1, 0]) if abs(elev) < 85 else np.array([0, 0, -1])
    right = np.cross(fwd, up); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    pose = np.eye(4); pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -fwd, eye
    scene.add(cam, pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
    color, _ = r.render(scene)
    Image.fromarray(color).save(render_dir / f"{name}.png")
r.delete()
print("renders:", sorted(p.name for p in render_dir.glob("*.png")))
