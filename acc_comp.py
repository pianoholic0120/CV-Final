import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def compute_metrics(pred_pcd, gt_pcd):
    """
    Calculate the Accuracy (pred → gt) and Completeness (gt → pred) for a single pair.
    Return: (accuracy_med, completeness_med), both are median distances.
    """
    pred_pts = np.asarray(pred_pcd.points)
    gt_pts   = np.asarray(gt_pcd.points)
    # Build KD-tree
    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    # Accuracy: For each pred_point, find the nearest GT
    d_pred_gt, _ = tree_gt.query(pred_pts, k=1)
    # Completeness: For each gt_point, find the nearest pred
    d_gt_pred, _ = tree_pred.query(gt_pts, k=1)
    # Return median
    return np.median(d_pred_gt), np.median(d_gt_pred)

def main(pred_folder, gt_folder):
    # 1. Collect all .ply files in both folders (only consider filenames, not paths)
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.lower().endswith('.ply')])
    gt_files   = sorted([f for f in os.listdir(gt_folder)   if f.lower().endswith('.ply')])

    # 2. Check if the number of files is consistent
    if len(pred_files) != len(gt_files):
        print(f"Error: There are {len(pred_files)} .ply files in the prediction folder ({pred_folder}), "
              f"and {len(gt_files)} .ply files in the GT folder ({gt_folder}), the counts do not match!")
        return

    # 3. Check if the filename sets are identical
    set_pred = set(pred_files)
    set_gt   = set(gt_files)
    if set_pred != set_gt:
        missing_in_gt   = sorted(list(set_pred - set_gt))
        missing_in_pred = sorted(list(set_gt - set_pred))
        if missing_in_gt:
            print("The following prediction files could not be found in the GT folder:")
            for fn in missing_in_gt:
                print("  -", fn)
        if missing_in_pred:
            print("The following GT files could not be found in the prediction folder:")
            for fn in missing_in_pred:
                print("  -", fn)
        return

    # 4. Read files one by one, calculate metrics, and collect them in a list
    acc_list, comp_list = [], []
    for fname in pred_files:
        pred_path = os.path.join(pred_folder, fname)
        gt_path   = os.path.join(gt_folder,   fname)

        # Read point cloud
        try:
            pred_pcd = o3d.io.read_point_cloud(pred_path)
            gt_pcd   = o3d.io.read_point_cloud(gt_path)
        except Exception as e:
            print(f"Failed to read point cloud: {fname}, skipping. Error message: {e}")
            continue

        # Calculate Accuracy & Completeness
        acc, comp = compute_metrics(pred_pcd, gt_pcd)
        print(f"{fname} → Accuracy: {acc:.4f} m, Completeness: {comp:.4f} m")

        acc_list.append(acc)
        comp_list.append(comp)

    # 5. If there are any valid files, calculate and display the average
    if acc_list:
        avg_acc  = float(np.mean(acc_list))
        avg_comp = float(np.mean(comp_list))
        print("\n--- Final Average Scores ---")
        print(f"Average Accuracy     : {avg_acc:.4f} m")
        print(f"Average Completeness : {avg_comp:.4f} m")
    else:
        print("No valid files to calculate.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate Accuracy & Completeness for corresponding .ply files in two folders and display average scores in the terminal."
    )
    parser.add_argument('--pred_folder', required=True, help="Path to the folder containing predicted .ply files")
    parser.add_argument('--gt_folder',   required=True, help="Path to the folder containing Ground-Truth .ply files")
    args = parser.parse_args()
    main(args.pred_folder, args.gt_folder)
