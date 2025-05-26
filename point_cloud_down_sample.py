#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import open3d as o3d
import numpy as np

# ---------- CLI ----------
def get_args():
    p = argparse.ArgumentParser(
        description="Batch-down-sample every .ply in a folder and save to target folder")
    p.add_argument("--src_dir", required=True, help="folder containing *.ply")
    p.add_argument("--out_dir", required=True, help="where to write down-sampled ply")
    p.add_argument("--target_pts", type=int, default=300_000,
                   help="final number of points you roughly want")
    # voxel + denoise
    p.add_argument("--voxel", type=float, default=0.004,
                   help="voxel size for first down-sample (m)")
    p.add_argument("--sor_k", type=int, default=20,
                   help="statistical-outlier nb_neighbors (0 to skip)")
    p.add_argument("--sor_std", type=float, default=2.0,
                   help="statistical-outlier std_ratio")
    p.add_argument("--r_radius", type=float, default=0.05,
                   help="radius-outlier radius (0 to skip)")
    p.add_argument("--r_min", type=int, default=8,
                   help="radius-outlier min_pts")
    return p.parse_args()


# ---------- utilities ----------
def denoise(pcd: o3d.geometry.PointCloud, k: int, std: float,
            r_radius: float, r_min: int) -> o3d.geometry.PointCloud:
    if k > 0:
        pcd, _ = pcd.remove_statistical_outlier(k, std)
    if r_radius > 0:
        pcd, _ = pcd.remove_radius_outlier(nb_points=r_min, radius=r_radius)
    return pcd


def downsample_to_target(pcd: o3d.geometry.PointCloud,
                         target: int) -> o3d.geometry.PointCloud:
    n = np.asarray(pcd.points).shape[0]
    if n <= target:
        return pcd
    prob = target / float(n)
    return pcd.random_down_sample(prob)


# ---------- main ----------
def main():
    args = get_args()
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_files = list(src_dir.glob("*.ply"))
    if not ply_files:
        print(f"No .ply found in {src_dir}")
        return

    for ply_path in ply_files:
        print(f"[{ply_path.name}] loading …", end="", flush=True)
        pcd = o3d.io.read_point_cloud(str(ply_path))
        n0 = len(pcd.points)

        # 1) voxel down-sample
        if args.voxel > 0:
            pcd = pcd.voxel_down_sample(args.voxel)

        # 2) denoise
        pcd = denoise(pcd, args.sor_k, args.sor_std,
                      args.r_radius, args.r_min)

        # 3) uniform random down-sample to target
        pcd = downsample_to_target(pcd, args.target_pts)
        n1 = len(pcd.points)

        # write
        out_path = out_dir / ply_path.name
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f" done. {n0:,} → {n1:,} pts ➜ {out_path}")

    print("All files processed.")

if __name__ == "__main__":
    main()
