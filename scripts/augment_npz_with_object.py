#!/usr/bin/env python3
"""Add object reference trajectories to AnyBody motion .npz files.

Why this exists
---------------
``csv_to_npz.py`` / ``batch_csv_to_npz.py`` produce motion files holding only the
robot's state (``joint_pos``, ``body_pos_w``, ...). The OmniRetarget dataset, however,
also records the pose of the manipulated object in ``qpos[:, 36:43]``, and the CSV
interchange format used by AnyBody has no column for it. This script re-attaches that
object trajectory to an already-converted motion file so the training environment can
use it as an object-tracking reference.

Frame alignment
---------------
The object trajectory must land on exactly the same time base as the robot motion,
otherwise the box and the robot drift apart by a frame or more. ``MotionLoader``
(scripts/motion_csv_loader.py) resamples with:

    duration = (input_frames - 1) * (1 / input_fps)
    times    = arange(0, duration, 1 / output_fps)
    phase    = times / duration
    index_0  = floor(phase * (input_frames - 1))
    index_1  = min(index_0 + 1, input_frames - 1)
    blend    = phase * (input_frames - 1) - index_0
    value    = lerp/slerp(v[index_0], v[index_1], blend)

We reproduce it exactly, but take the output frame count from the already-converted
robot npz rather than recomputing ``arange`` — float32 vs float64 accumulation can
differ by one frame at the boundary, and the robot file is the ground truth.

Quaternions are (w, x, y, z) in both OmniRetarget and Isaac Lab, and Isaac Lab's
``quat_slerp`` takes the shortest path, so we do the same and perform no reordering.

Output keys added
-----------------
``object_pos_w``   (T, 3) float32  world position, metres
``object_quat_w``  (T, 4) float32  world orientation, (w, x, y, z)
``object_name``    ()     str      asset key, e.g. "largebox" / "chair_scaled_1.0"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

N_JOINTS = 29
BASE_DIM = 7                      # root quat(4) + root pos(3)
OBJ_SLICE = slice(BASE_DIM + N_JOINTS, BASE_DIM + N_JOINTS + 7)   # qpos[36:43]
CONVERTED_PREFIX = "omniret_g1_"  # batch_csv_to_npz --output_prefix

# Assets we are prepared to simulate. OmniRetarget_Dataset/models/ ships meshes for
# largebox and chair only, but the chair motions (robot-object-terrain/scene_*) store an
# object quaternion whose norm is a constant 0.7416 instead of 1.0 -- dividing it out
# does yield a valid unit quaternion, but the meaning of that constant factor is not
# understood, so those 138 motions are deliberately out of scope rather than guessed at.
# largebox covers 1796 / 2089 motions (86%) with exactly unit-norm quaternions.
KNOWN_ASSETS = {"largebox"}
UNVERIFIED_ASSETS = {"chair"}


def detect_object(name: str) -> str | None:
    """Map a motion name onto an asset key in OmniRetarget_Dataset/models/."""
    # sub12_largebox_086_original, sub10_largebox_000_trans_2, ...
    m = re.search(r"sub\d+_([a-z]+)_", name)
    if m:
        return m.group(1)
    # scene_00_chair_scaled_0.9, scene_00_chair_scaled_1.0_z_scale_1.1, ...
    m = re.search(r"(chair(?:_scaled_\d+\.\d+)?)", name)
    if m:
        return m.group(1)
    return None


def asset_base(object_name: str) -> str:
    """Strip the _scaled_x.y suffix to get the mesh directory name."""
    return object_name.split("_scaled_")[0]


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Batched shortest-path SLERP on (w, x, y, z) quaternions.

    Mirrors isaaclab.utils.math.quat_slerp (which negates q2 when the dot product is
    negative and falls back to q1 for near-degenerate pairs).
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64).copy()
    dot = np.sum(q0 * q1, axis=1)
    flip = dot < 0.0
    q1[flip] *= -1.0
    dot = np.abs(dot)
    np.clip(dot, -1.0, 1.0, out=dot)
    angle = np.arccos(dot)
    sin_a = np.sin(angle)
    degenerate = sin_a < 1e-8          # parallel (or antiparallel-after-flip) inputs
    safe_sin = np.where(degenerate, 1.0, sin_a)
    w0 = np.where(degenerate, 1.0 - t, np.sin((1.0 - t) * angle) / safe_sin)
    w1 = np.where(degenerate, t, np.sin(t * angle) / safe_sin)
    out = w0[:, None] * q0 + w1[:, None] * q1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(n == 0.0, 1.0, n)).astype(np.float32)


def upright_yaw_only(quat_wxyz: np.ndarray) -> np.ndarray:
    """Replace the recorded orientation with 'upright, same yaw'.

    OmniRetarget's largebox asset has its own local z axis pointing DOWN: across whole clips
    the world-frame components of its local z average about (-0.16, 0.02, -0.99). Handing that
    quaternion straight to an Isaac ``CuboidCfg`` -- whose local z is up by construction --
    spawns the box nearly upside down, so it is balanced on an edge and topples the instant
    physics starts. That toppling, not the robot, was producing ~0.45 m of box displacement
    on every episode regardless of mass, friction, collision size or reward weight.

    For a box sliding flat the only meaningful degree of freedom is yaw, and it is well
    behaved: taking the asset's local x axis, projecting it onto the world XY plane and
    reading atan2 gives a yaw that moves at most 1.4-1.9 deg per frame. So we keep that yaw
    and drop the rest, which also removes the ~9 deg of residual wobble in the recording.
    """
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    # First column of the rotation matrix = the asset's local x axis in world coordinates.
    ax_x = 1.0 - 2.0 * (y * y + z * z)
    ax_y = 2.0 * (x * y + w * z)
    yaw = np.arctan2(ax_y, ax_x)
    half = 0.5 * yaw
    out = np.zeros_like(quat_wxyz)
    out[:, 0] = np.cos(half)
    out[:, 3] = np.sin(half)
    return out.astype(np.float32)


def resample(quat_in: np.ndarray, pos_in: np.ndarray, input_fps: int,
             output_fps: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample an object trajectory onto MotionLoader's output time base."""
    n_in = pos_in.shape[0]
    if n_in < 2:
        raise ValueError(f"need >= 2 input frames, got {n_in}")
    duration = (n_in - 1) / float(input_fps)
    # Same samples as torch.arange(0, duration, 1/output_fps) but with the count
    # pinned to the robot file, which is authoritative.
    times = np.arange(n_out, dtype=np.float64) / float(output_fps)
    if times[-1] > duration + 1e-9:
        raise ValueError(
            f"robot npz has {n_out} frames ({times[-1]:.4f}s) but object trajectory "
            f"only covers {duration:.4f}s"
        )
    phase = times / duration
    scaled = phase * (n_in - 1)
    i0 = np.floor(scaled).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    blend = scaled - i0
    pos = (pos_in[i0] * (1.0 - blend[:, None]) + pos_in[i1] * blend[:, None]).astype(np.float32)
    quat = slerp(quat_in[i0], quat_in[i1], blend)
    return quat, pos


def augment(conv_path: Path, raw_root: Path, dry_run: bool = False) -> dict:
    """Attach object data to one converted motion file. Returns a status dict."""
    stem = conv_path.stem
    name = stem[len(CONVERTED_PREFIX):] if stem.startswith(CONVERTED_PREFIX) else stem

    obj = detect_object(name)
    if obj is None:
        return {"name": name, "status": "no-object", "reason": "name has no object token"}
    base = asset_base(obj)
    if base in UNVERIFIED_ASSETS:
        return {"name": name, "status": "unverified-asset",
                "reason": f"{base!r}: object quat norm != 1 in source data, layout not understood"}
    if base not in KNOWN_ASSETS:
        return {"name": name, "status": "no-mesh", "reason": f"no mesh for {base!r}"}

    matches = list(raw_root.rglob(f"{name}.npz"))
    if not matches:
        return {"name": name, "status": "raw-missing", "reason": "original OmniRetarget npz not found"}
    raw_path = matches[0]

    with np.load(raw_path, allow_pickle=True) as raw:
        if "qpos" not in raw:
            return {"name": name, "status": "bad-raw", "reason": "no qpos"}
        qpos = np.asarray(raw["qpos"], dtype=np.float64)
        in_fps = int(raw["fps"])
    if qpos.shape[1] != BASE_DIM + N_JOINTS + 7:
        return {"name": name, "status": "no-object",
                "reason": f"qpos has D={qpos.shape[1]} (no object block)"}

    with np.load(conv_path, allow_pickle=True) as conv:
        data = {k: conv[k] for k in conv.files}
    n_out = int(data["joint_pos"].shape[0])
    out_fps = int(np.asarray(data["fps"]).reshape(-1)[0])

    obj_block = qpos[:, OBJ_SLICE]
    quat_in, pos_in = obj_block[:, 0:4], obj_block[:, 4:7]
    norms = np.linalg.norm(quat_in, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        return {"name": name, "status": "bad-raw",
                "reason": f"object quat norm {norms.min():.4f}..{norms.max():.4f}"}

    quat, pos = resample(quat_in, pos_in, in_fps, out_fps, n_out)
    quat = upright_yaw_only(quat)

    if not dry_run:
        data["object_pos_w"] = pos
        data["object_quat_w"] = quat
        data["object_name"] = np.array(obj)
        # np.savez appends ".npz" when the path does not already end in it, so the temp
        # name must itself end in .npz or the rename below looks for the wrong file.
        tmp = conv_path.with_name(conv_path.stem + ".tmp.npz")
        np.savez(tmp, **data)
        tmp.replace(conv_path)

    return {"name": name, "status": "ok", "object": obj, "frames": n_out,
            "in_frames": qpos.shape[0], "in_fps": in_fps, "out_fps": out_fps,
            "travel_m": float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))),
            "z_mean": float(pos[:, 2].mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--converted_dir", required=True,
                    help="Directory of npz produced by batch_csv_to_npz.py")
    ap.add_argument("--raw_root", required=True,
                    help="Root of OmniRetarget_Dataset (searched recursively)")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    files = sorted(Path(a.converted_dir).glob(a.pattern))
    if a.limit:
        files = files[: a.limit]
    raw_root = Path(a.raw_root)

    counts: dict[str, int] = {}
    objects: dict[str, int] = {}
    examples: list[dict] = []
    for f in files:
        r = augment(f, raw_root, dry_run=a.dry_run)
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] == "ok":
            objects[r["object"]] = objects.get(r["object"], 0) + 1
            if len(examples) < 5:
                examples.append(r)
        elif counts[r["status"]] <= 3:
            print(f"  [{r['status']}] {r['name'][:60]}: {r['reason']}", file=sys.stderr)

    print(f"\n处理 {len(files)} 个文件{'（dry-run，未写入）' if a.dry_run else ''}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14} {v}")
    if objects:
        print("  物体分布:", ", ".join(f"{k}={v}" for k, v in sorted(objects.items(), key=lambda kv: -kv[1])))
    for e in examples:
        print(f"  样本 {e['name'][:44]:<44} {e['object']:<20} "
              f"{e['in_frames']}@{e['in_fps']} -> {e['frames']}@{e['out_fps']}  "
              f"移动 {e['travel_m']:.2f}m  z均值 {e['z_mean']:.3f}m")
    return 0 if counts.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
