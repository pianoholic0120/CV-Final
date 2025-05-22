# CV-Final

## Environment

## Requirements & Installation

## File Structure

## Ablation

| ID | Method | Pre-processing | Post-processing | Focal Length | Depth |Property | Accuracy | Completeness | 
| :-----| ----: | :----: | :-----| ----: | :----: |:----: | :----: |:----: |
| 291723 | GT | ✅  | ✅ | ✅ |✅ | -  |  0.0 | 0.51 |  
| 295646 | GT | ❌  | ❌ | ✅ |✅ | -  | 0.0  | 0.51  |
| 286791 | ACE0 | ❌ | ❌ | ❌ |❌ | Sparse  | 0.22  | 0.31  |
| 287379 | ACE0 | ❌  | ❌ | ✅ | ❌| Dense  |  0.53 | 0.18  |
| 287534 | ACE0 | ❌  | ❌| ✅ |❌ | Sparse  |  0.27 |  0.31 |
| 291376 | COLMAP | ❌  | ❌ | ❌ |❌| Sparse  |  4.28 | 0.4  |
| 291970 | ACE0 | ❌  | ❌ | ❌ | ✅| Sparse  |  0.2 |  0.34 |
| 292226 | ACE0 | ❌  | ✅ (Adaptive)| ❌ | ✅| Sparse  |  0.27 |  0.35 |
| 292839 | ACE0 | ❌  | ✅ (7.5e-3) | ❌ | ✅| Sparse  | 0.23  | 0.34  |
| 295372 | ACE0 | ✅ (20)  | ❌ | ❌ | ❌| Sparse  | 0.24  |  0.38 |
| 295908 | ACE0 | ✅ (Q-align)  | ❌ | ❌ | ❌| Sparse  |  0.26 |  0.34 |
| 296158 | cv25s + ACE0 | ✅ (15)  | ✅ (2.5e-3) | ✅ | ✅| Sparse  | 0.25  |  0.19 |
|  | cv25s + Droid-SLAM | ✅  | ✅ | ✅ | ✅| Sparse  |   |   |


* Pre-processing: kf_every, Q-align
* Post-processing: voxel_grid_size (down-sample)