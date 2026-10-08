#!/usr/bin/env python3
"""Text2HOI export NPZ -> OmniDexGrasp Stage 3 input (out/<task>_f###/optim_res.json + scaled_mesh.obj).

What changes (and only this):
  rotation  : Text2HOI 6D (16 x 6)            -> MANO axis-angle fullpose (48)
  frame     : Text2HOI world (object moves)   -> object mesh-local (object fixed)
  translation: world hand trans               -> R_obj^T (J0 + trans - t_obj) - J0
                                                 (MANO rotates about the wrist joint J0, so the
                                                  translation must be corrected by J0)
Everything else is written so that OmniDexGrasp's loader transform is the identity:
  T = diag(1,-1,-1,1) (cancels the loader's HaMeR y/z flip), scale = 1, transl = 0, betas = 0.

    conda activate omnidexgrasp
    python scripts/text2hoi_to_omnidex.py --npz <export>.npz --task cappuccino --frame 30
    python scripts/text2hoi_to_omnidex.py --npz <export>.npz --task cappuccino --frame all
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]


def rot6d_to_matrix(x):
    """Identical to Text2HOI lib/utils/rot.py:rot6d_to_rotmat (note the (3, 2) interleaved reshape)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3, 2)
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)                      # (N, 3, 3), columns b1 b2 b3


def object_rotation(x_obj, dataset):
    """R_obj such that v_world = R_obj @ v_mesh + t_obj, per Text2HOI lib/utils/data.py:process_obj_result."""
    R = rot6d_to_matrix(x_obj[:, 3:9])
    if dataset == "grab":                                        # grab: einsum('tij,ki->tkj') == R^T @ v
        R = R.transpose(0, 2, 1)
    return R, x_obj[:, :3].astype(np.float64)


def mano_wrist_offset(mano_root):
    """J0: wrist joint of the zero-pose, zero-shape MANO (same regressor in smplx MANO and manotorch)."""
    from manotorch.manolayer import ManoLayer
    layer = ManoLayer(side="right", mano_assets_root=str(mano_root), use_pca=False)
    with torch.no_grad():
        return layer(torch.zeros(1, 48), torch.zeros(1, 10)).joints[0, 0].double().numpy()


def convert(x_hand, x_obj, dataset, J0):
    """(T, 99), (T, 9|10) world-frame -> fullpose (T, 48) and trans (T, 3) in object mesh-local frame."""
    T = x_hand.shape[0]
    R_hand = rot6d_to_matrix(x_hand[:, 3:].reshape(-1, 6)).reshape(T, 16, 3, 3)
    R_obj, t_obj = object_rotation(x_obj, dataset)
    R_obj_T = R_obj.transpose(0, 2, 1)

    R_global = R_obj_T @ R_hand[:, 0]                            # R_obj^T @ R_hand
    fullpose = np.concatenate([
        Rotation.from_matrix(R_global).as_rotvec()[:, None, :],
        Rotation.from_matrix(R_hand[:, 1:].reshape(-1, 3, 3)).as_rotvec().reshape(T, 15, 3),
    ], axis=1).reshape(T, 48)
    trans = np.einsum("tij,tj->ti", R_obj_T, J0 + x_hand[:, :3] - t_obj) - J0
    return fullpose, trans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, type=Path)
    p.add_argument("--task", required=True, help="task name prefix; folders are <task>_f###")
    p.add_argument("--frame", default="all", help="frame index or 'all' (all valid right-hand frames)")
    p.add_argument("--out", default=REPO / "out", type=Path)
    p.add_argument("--mano-root", default=REPO / "assets" / "mano", type=Path)
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    dataset = str(d["dataset_name"])
    if dataset == "arctic":
        raise SystemExit("arctic (articulated object) is not supported")
    if not bool(d["is_rhand"]):
        raise SystemExit("export has no right hand; OmniDexGrasp Stage 3 retargets the right hand only")

    valid = np.flatnonzero(d["valid_mask_right"])
    frames = valid.tolist() if args.frame == "all" else [int(args.frame)]
    bad = [f for f in frames if f not in valid]
    if bad:
        raise SystemExit(f"frames {bad} are outside the valid range {valid[0]}..{valid[-1]}")

    J0 = mano_wrist_offset(args.mano_root)
    fullpose, trans = convert(d["right_hand"].astype(np.float64), d["object_pose"].astype(np.float64), dataset, J0)
    mesh = trimesh.Trimesh(d["object_vertices"], d["object_faces"], process=False)

    for f in frames:
        task_dir = args.out / f"{args.task}_f{f:03d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        mesh.export(task_dir / "scaled_mesh.obj")
        (task_dir / "optim_res.json").write_text(json.dumps({
            "fullpose": [fullpose[f].tolist()],
            "betas": [[0.0] * 10],
            "cam_transl": [trans[f].tolist()],
            "is_right": True,
            "T": [[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]],
            "hand_params": {"scale": [1.0], "transl": [0.0, 0.0, 0.0]},
            "source": {"npz": str(args.npz), "frame": f, "dataset": dataset, "text": str(d["text"])},
        }, indent=2))
    print(f"wrote {len(frames)} task folder(s) under {args.out}: {args.task}_f{frames[0]:03d}"
          + (f" .. {args.task}_f{frames[-1]:03d}" if len(frames) > 1 else ""))


if __name__ == "__main__":
    main()
