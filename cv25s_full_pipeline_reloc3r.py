import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Convert one sequence (Reloc3r poses_final) into a clean, down-sampled point cloud")
    p.add_argument("--seq_dir", required=True, help=".../scene/seq-XX/")
    p.add_argument("--output",  required=True, help="output .ply")
    p.add_argument("--depth_suffix", default=".depth.proj.png")
    p.add_argument("--color_suffix", default=".color.png")
    p.add_argument("--depth_scale",  type=float, default=1e-3, help="depth units ➜ metres")
    # depth / bbox
    p.add_argument("--depth_max", type=float, default=4.0, help="truncate depths > depth_max (m)")
    p.add_argument("--bbox_half_extent", type=float, default=3.0, help="+/-m around sequence barycentre")
    # statistical outlier removal
    p.add_argument("--sor_k",   type=int,   default=20)
    p.add_argument("--sor_std", type=float, default=2.0)
    # radius outlier removal
    p.add_argument("--r_radius", type=float, default=0.05)
    p.add_argument("--r_min",    type=int,   default=8)
    # final target pts
    p.add_argument("--max_points", type=int, default=300000)
    return p.parse_args()


# ---------- helpers ----------
def load_poses(pose_file: Path):
    poses, focals = {}, {}
    for ln in pose_file.open():
        tok = ln.split()
        if len(tok) < 8:
            continue
        key = tok[0].replace(".color.png", "")
        qw, qx, qy, qz = map(float, tok[1:5])
        tx, ty, tz     = map(float, tok[5:8])
        f              = float(tok[8]) if len(tok) >= 9 else None
        Twc = np.eye(4, dtype=np.float32)
        Twc[:3,:3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        Twc[:3, 3] = [tx, ty, tz]
        poses[key]  = Twc
        if f is not None:
            focals[key] = f
    return poses, focals


def backproject(depth, K):
    H, W = depth.shape
    v, u = np.indices((H, W))
    z    = depth.flatten()
    valid = (z > 0) & np.isfinite(z)
    u, v, z = u.flatten()[valid], v.flatten()[valid], z[valid]
    x = (u - K["cx"]) * z / K["f"]
    y = (v - K["cy"]) * z / K["f"]
    return np.stack((x, y, z), 1), valid


# ---------- main ----------
def main():
    args = parse_args()
    seq_dir = Path(args.seq_dir)
    pose_file = seq_dir / "poses_final.txt"
    assert pose_file.exists(), "missing poses_final.txt"
    poses, focals = load_poses(pose_file)

    # assume all frames same size
    samp = next(iter(poses))
    depth0 = cv2.imread(str(seq_dir / f"{samp}{args.depth_suffix}"), cv2.IMREAD_UNCHANGED)
    H, W = depth0.shape

    # accumulate
    pts_all, col_all, centres = [], [], []

    for key, Twc in poses.items():
        d_path = seq_dir / f"{key}{args.depth_suffix}"
        c_path = seq_dir / f"{key}{args.color_suffix}"
        if not d_path.exists() or not c_path.exists():
            continue

        depth = cv2.imread(str(d_path), cv2.IMREAD_UNCHANGED).astype(np.float32) * args.depth_scale
        depth[depth > args.depth_max] = 0  # truncate far
        color = cv2.cvtColor(cv2.imread(str(c_path)), cv2.COLOR_BGR2RGB)

        f = focals.get(key)
        if f is None:
            continue
        K = {"f": f, "cx": W/2.0, "cy": H/2.0}

        cam_pts, valid = backproject(depth, K)
        cols = color.reshape(-1, 3)[valid] / 255.0

        # to world
        pts_w = (Twc[:3, :3] @ cam_pts.T + Twc[:3, 3:4]).T
        pts_all.append(pts_w)
        col_all.append(cols)
        centres.append(Twc[:3, 3])

    if not pts_all:
        print("no points!")
        return

    pts  = np.vstack(pts_all)
    cols = np.vstack(col_all)

    # bbox crop (3m around barycentre)
    ctr = np.mean(np.vstack(centres), 0)
    mask = np.all(np.abs(pts - ctr) < args.bbox_half_extent, 1)
    pts, cols = pts[mask], cols[mask]

    # open3d cloud
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(cols)

    pcd = pcd.voxel_down_sample(0.005)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=args.sor_k,
                                            std_ratio=args.sor_std)
    pcd, _ = pcd.remove_radius_outlier(nb_points=args.r_min,
                                       radius=args.r_radius)

    n = len(pcd.points)
    if n > args.max_points:
        pcd = pcd.random_down_sample(float(args.max_points)/n)

    o3d.io.write_point_cloud(args.output, pcd)
    print(f"Saved {len(pcd.points)} pts ➜ {args.output}")


if __name__ == "__main__":
    main()
