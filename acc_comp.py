import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def compute_metrics(pred_pcd, gt_pcd):
    pred_pts = np.asarray(pred_pcd.points)
    gt_pts   = np.asarray(gt_pcd.points)
    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    # Accuracy: predicted → GT
    d_pred_gt, _ = tree_gt.query(pred_pts, k=1)
    # Completeness: GT → predicted
    d_gt_pred, _ = tree_pred.query(gt_pts, k=1)
    return np.median(d_pred_gt), np.median(d_gt_pred)

def main(pred_folder, gt_folder):
    files = sorted(f for f in os.listdir(pred_folder) if f.endswith('.ply'))
    acc_list, comp_list = [], []
    
    for fname in files:
        # parse parameters from filename: base-v{voxel_mm}-k{kf}.ply
        if '-v' not in fname or '-k' not in fname:
            print(f"Skipping {fname}: invalid name")
            continue
        base, params = fname[:-4].rsplit('-v', 1)
        voxel_tag, kf_tag = params.split('-k')
        gt_name = base + '.ply'
        pred_path = os.path.join(pred_folder, fname)
        gt_path   = os.path.join(gt_folder,   gt_name)
        if not os.path.isfile(gt_path):
            print(f"Skipping {fname}: no GT {gt_name}")
            continue
        
        pred_pcd = o3d.io.read_point_cloud(pred_path)
        gt_pcd   = o3d.io.read_point_cloud(gt_path)
        acc, comp = compute_metrics(pred_pcd, gt_pcd)
        
        print(f"{fname} → Acc: {acc:.4f} m, Comp: {comp:.4f} m")
        acc_list.append(acc)
        comp_list.append(comp)
    
    if acc_list:
        avg_acc = float(np.mean(acc_list))
        avg_comp = float(np.mean(comp_list))
        print(f"\nFinal Average Accuracy: {avg_acc:.4f} m")
        print(f"Final Average Completeness: {avg_comp:.4f} m")
    else:
        print("No valid files to compute metrics.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute average Accuracy & Completeness")
    parser.add_argument('--pred_folder', required=True)
    parser.add_argument('--gt_folder',   required=True)
    args = parser.parse_args()
    main(args.pred_folder, args.gt_folder)

