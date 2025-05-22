# CV-Final

## Requirements & Installation

## File Structure

## How to run (examples)

First, you have to generate poses for desired image sequence. 

> To sweep parameters (voxels, kf) of reconstruction

    python cv25s_full_pipeline.py --data_root /path/to/your_file/with/7scenes/ --output_dir /path/to/save/your_results/ --voxels 0.0025,0.003 --kf 1,5

> To compute accuracy w.r.t GT
    
    python acc_comp.py --pred_folder /path/to/your_file/with/point_clouds/ --gt_folder /path/to/your_file/with/gt7scenes/


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
| 296420 | Combined | ✅  | ✅ | ✅ | ✅| Sparse  | 0.16  |  0.16 |
|  | cv25s + Droid-SLAM | ✅  | ✅ | ✅ | ✅| Sparse  |   |   |


* Pre-processing: kf_every, Q-align
* Post-processing: voxel_grid_size (down-sample)

## External Dependences
1. ACE0
    https://github.com/nianticlabs/acezero
2. Droid-SLAM
    https://github.com/princeton-vl/DROID-SLAM
3. Q-Align
    https://github.com/Q-Future/Q-Align
4. COLMAP SfM
    https://github.com/colmap/colmap


## Cited
ACE0:

    @article{bhat2023zoedepth,
    title={Zoe{D}epth: Zero-shot transfer by combining relative and metric depth},
    author={Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and M{\"u}ller, Matthias},
    journal={arXiv},
    year={2023}
    }
Droid-SLAM:

    @article{teed2021droid,
    title={{DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras}},
    author={Teed, Zachary and Deng, Jia},
    journal={Advances in neural information processing systems},
    year={2021}
    }

Q-Align:

    @article{wu2023qalign,
    title={Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels},
    author={Wu, Haoning and Zhang, Zicheng and Zhang, Weixia and Chen, Chaofeng and Li, Chunyi and Liao, Liang and Wang, Annan and Zhang, Erli and Sun, Wenxiu and Yan, Qiong and Min, Xiongkuo and Zhai, Guangtai and Lin, Weisi},
    journal={arXiv preprint arXiv:2312.17090},
    year={2023},
    institution={Nanyang Technological University and Shanghai Jiao Tong University and Sensetime Research},
    note={Equal Contribution by Wu, Haoning and Zhang, Zicheng. Project Lead by Wu, Haoning. Corresponding Authors: Zhai, Guangtai and Lin, Weisi.}
    }

COLMAP:

    @inproceedings{schoenberger2016sfm,
        author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
        title={Structure-from-Motion Revisited},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }