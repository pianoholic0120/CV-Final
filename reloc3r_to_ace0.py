import argparse
import math
import os
from pathlib import Path
import numpy as np


# ---------- quaternion utilities ---------- #
def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    (w, x, y, z)  → 3×3 rotation matrix, right-handed, row major
    """
    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm == 0:
        raise ValueError("Quaternion has zero norm.")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    r00 = 1 - 2 * (qy*qy + qz*qz)
    r01 = 2 * (qx*qy - qz*qw)
    r02 = 2 * (qx*qz + qy*qw)

    r10 = 2 * (qx*qy + qz*qw)
    r11 = 1 - 2 * (qx*qx + qz*qz)
    r12 = 2 * (qy*qz - qx*qw)

    r20 = 2 * (qx*qz - qy*qw)
    r21 = 2 * (qy*qz + qx*qw)
    r22 = 1 - 2 * (qx*qx + qy*qy)

    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]], dtype=np.float64)


# ---------- main routine ---------- #
def convert_file(src_path: Path, dst_root: Path):
    name = src_path.stem           # poses_final-chess-seq-03
    if not name.startswith("poses_final-") or "-seq-" not in name:
        raise RuntimeError(f"Unexpected file name pattern: {src_path.name}")
    scene_and_seq = name[len("poses_final-"):]  # chess-seq-03
    scene, seq = scene_and_seq.split("-seq-")   # chess, 03

    out_dir = dst_root / scene / "test" / f"seq-{seq}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with src_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            tokens = line.strip().split()
            if len(tokens) != 9:
                raise RuntimeError(
                    f"{src_path.name}: line {line_num} has {len(tokens)} tokens (expect 9)"
                )

            img_name, qw, qx, qy, qz, tx, ty, tz, _ = tokens
            qw, qx, qy, qz = map(float, (qw, qx, qy, qz))
            tx, ty, tz = map(float, (tx, ty, tz))

            R = quat_to_rotmat(qw, qx, qy, qz)
            T = np.array([[tx], [ty], [tz]], dtype=np.float64)

            pose_4x4 = np.block([
                [R, T],
                [np.zeros((1, 3), dtype=np.float64),
                 np.ones((1, 1), dtype=np.float64)]
            ])

            pose_name = img_name.replace(".color.png", ".pose.txt")
            (out_dir / pose_name).write_text(
                "\n".join(" ".join(f"{v:.6f}" for v in row) for row in pose_4x4),
                encoding="utf-8"
            )


def cli():
    p = argparse.ArgumentParser(
        description="Convert Reloc3r poses (stdloc format) to ACE0 7-Scenes format."
    )
    p.add_argument("--poses_dir", required=True,
                   help="folder containing poses_final-*.txt files")
    p.add_argument("--output_root", default="/home/arthur/storage/7scenes",
                   help="7-Scenes root folder to write converted poses")
    args = p.parse_args()

    src_dir = Path(args.poses_dir).expanduser().resolve()
    dst_root = Path(args.output_root).expanduser().resolve()

    txt_files = sorted(src_dir.glob("poses_final-*-seq-*.txt"))
    if not txt_files:
        raise RuntimeError(f"No pose files found in {src_dir}")

    print(f"Found {len(txt_files)} pose files, converting…")
    for fp in txt_files:
        print(f"  • {fp.name}")
        convert_file(fp, dst_root)

    print("Done!  Converted poses are under", dst_root)


if __name__ == "__main__":
    cli()
