import re
import numpy as np
from pathlib import Path

FOCAL_LEN = 525.0                  
FMT_FLOAT = "{:.10f}"              

FRAME_RE = re.compile(r"frame-(\d+)\.color\.png")

# ---------- R → q ---------- #
def rot_to_quat(R: np.ndarray):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m21 - m12) * s
        qy = (m02 - m20) * s
        qz = (m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            qw = (m21 - m12) / s
            qx = 0.25 * s
            qy = (m01 + m10) / s
            qz = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            qw = (m02 - m20) / s
            qx = (m01 + m10) / s
            qy = 0.25 * s
            qz = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            qw = (m10 - m01) / s
            qx = (m02 + m20) / s
            qy = (m12 + m21) / s
            qz = 0.25 * s
    return qw, qx, qy, qz


def convert_poses(input_txt: str, output_txt: str):
    records = []

    with open(input_txt, "r") as fin:
        for ln, line in enumerate(fin, 1):
            parts = line.strip().split()
            if not parts:
                continue

            n = len(parts)
            path = parts[0]
            basename = Path(path).name

            m = FRAME_RE.match(basename)
            if not m:
                print(f"[Skip line {ln}] Unable to parse frame number: {basename}")
                continue
            frame_idx = int(m.group(1))

            if n == 17:
                nums = list(map(float, parts[5:]))  
                R = np.array(
                    [
                        nums[0:3],          # r11 r12 r13
                        nums[4:7],          # r21 r22 r23
                        nums[8:11],         # r31 r32 r33
                    ],
                    dtype=np.float64,
                )
                t = np.array([nums[3], nums[7], nums[11]], dtype=np.float64)
                qw, qx, qy, qz = rot_to_quat(R)

            elif n in (9, 10):
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                t = np.array([tx, ty, tz], dtype=np.float64)
            else:
                print(f"[Skip line {ln}] Length {n} not supported")
                continue

            records.append(
                (frame_idx, basename, qw, qx, qy, qz, *t.tolist())
            )

    records.sort(key=lambda x: x[0])

    with open(output_txt, "w") as fout:
        for _, fname, qw, qx, qy, qz, tx, ty, tz in records:
            fout.write(
                f"{fname} "
                f"{FMT_FLOAT.format(qw)} {FMT_FLOAT.format(qx)} "
                f"{FMT_FLOAT.format(qy)} {FMT_FLOAT.format(qz)} "
                f"{FMT_FLOAT.format(tx)} {FMT_FLOAT.format(ty)} {FMT_FLOAT.format(tz)} "
                f"{FOCAL_LEN:.6f}\n"
            )


if __name__ == "__main__":
    convert_poses(
        input_txt="/home/arthur/cv/reloc3r_ace0_refined_poses/poses_final-chess-seq-03.txt",
        output_txt="/home/arthur/cv/reloc3r_ace0_refined_poses/chess-seq-01.txt"
    )
    print("Generated poses_final-*.txt (sorted)")

