import numpy as np

ACE0_FOCAL = None  

INTRINSIC_TUP = (525.0, 525.0, 320.0, 240.0)  # fx, fy, cx, cy

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)]
    ], dtype=np.float32)

    return R

def convert_ace0_to_poses(input_txt, output_txt):
    """
      <filename> fx fy cx cy  r11 r12 r13 t1  r21 r22 r23 t2  r31 r32 r33 t3
    """
    fx, fy, cx, cy = INTRINSIC_TUP

    with open(input_txt, 'r') as fin, open(output_txt, 'w') as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 8:
                continue  
            filename = parts[0]
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            # focal_length, confidence = map(float, parts[8:10])  

            # quaternion -> rotation matrix
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

            vals = [filename,
                    f"{fx:.6f}", f"{fy:.6f}", f"{cx:.6f}", f"{cy:.6f}"]
            for i in range(3):
                for j in range(3):
                    vals.append(f"{R[i, j]:.6f}")
                vals.append(f"{[tx, ty, tz][i]:.6f}")

            fout.write(" ".join(vals) + "\n")

if __name__ == "__main__":
    convert_ace0_to_poses(
        input_txt="/home/arthur/storage/data/stairs-seq-01/stairs-seq-01-poses.txt",
        output_txt="/home/arthur/storage/data/stairs-seq-01/poses_final.txt"
    )
    print("Generated poses_final.txt")
