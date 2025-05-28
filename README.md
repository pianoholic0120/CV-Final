# CV-Final

## Requirements & Installation

## File Structure

## How to run (examples)

First, you have to generate poses for desired image sequence. 

> To sweep parameters (voxels, kf) of reconstruction based on ace0-generated poses:

    python cv25s_full_pipeline.py --data_root /path/to/your_file/with/7scenes/ --output_dir /path/to/save/your_results/ --voxels 0.0025,0.003 --kf 1,5

> To compute accuracy w.r.t GT:
    
    python acc_comp.py --pred_folder /path/to/your_file/with/point_clouds/ --gt_folder /path/to/your_file/with/gt7scenes/

> To run reloc3r to generate poses

Specify train/test split under ./reloc3r/datasets/sevenscenes_retrieval.py

Generate desired dataset under ./data/7scenes/{scene}/ with all seq-{seq_index}/ with *.color.png, *.depth.png, and *.poses.txt

- Testing sequences doesn't need pose.txt of each images.

Run reloc3r pipeline to generate poses_final.py

    python eval_visloc.py --model "Reloc3rRelpose(img_size=512)" --dataset_db "SevenScenesRetrieval(scene='{}', split='train')" --dataset_q "ne='{}', split='train')" --dataset_q "SevenScenesRetrieval(scene='{}', split='test')" --dataset_relpose "SevenScenesRelpose(scene='{}', pair_id-topk 10nesRetrieval(scene='{}', split={}, resolution={})" --scene "{scene}" --topk 10

Put the generated poses_final.py into the folder of testing sequence to be evaluated, then run:

    python cv25s_full_pipeline_reloc3r.py --seq_dir /path/to/the_folder/7scenes/{scenes}/test/seq-{sequence_index}/ --output /path/to/the_file/of/{scenes}-seq-{sequence_index}.ply



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
| 297182 | PCN Refined | ❌  | ✅ | ❌ | ✅| Sparse  | 0.20  |  0.34 |
| 297998 | cv25s + Reloc3r | ❌  | ✅(5e-3)| ✅ | ✅| - (300000) | 0.02  |  0.01 |
| 298700 | spann3r | ❌  | ✅(4e-3)| ❌ | ❌| - (300000) | 0.21  |  0.53 |
| 299961 | cv25s + Reloc3r | ❌  | ✅(2.5e-3)| ✅ | ✅| - (300000) | 0.01  |  0.01 |
|  | cv25s + Droid-SLAM | ✅  | ✅ | ✅ | ✅| Sparse  |   |   |

* Pre-processing: kf_every, Q-align
* Post-processing: voxel_grid_size (down-sample), PCN Refinement


### parameters markdown

| ID | Sor_k | Sor_std | r_radius | r_min | voxel_down_sample | # of points | Accuracy | Completeness | 
| :-----| ----: | :----: | :-----| ----: | :----: |:----: | :----: |:----: |
| 297998 | 20  | 2.0| 0.05 | 8| 0.005 |300000 | 0.02  |  0.01 |
| 299961 | 20  | 0.8| 0.03 | 12| 0.0025 |300000 | 0.01  |  0.01 |


## External Dependences
1. ACE0
    https://github.com/nianticlabs/acezero
2. Droid-SLAM
    https://github.com/princeton-vl/DROID-SLAM
3. Q-Align
    https://github.com/Q-Future/Q-Align
4. COLMAP SfM
    https://github.com/colmap/colmap
5. Reloc3r
    https://github.com/ffrivera0/reloc3r
6. PCN (Point Completion Network)
    https://github.com/wentaoyuan/pcn?tab=readme-ov-file
7. Spann3r
    https://github.com/HengyiWang/spann3r





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

Reloc3r:

    @article{reloc3r,
    title={Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization},
    author={Dong, Siyan and Wang, Shuzhe and Liu, Shaohui and Cai, Lulu and Fan, Qingnan and Kannala, Juho and Yang, Yanchao},
    journal={arXiv preprint arXiv:2412.08376},
    year={2024}
    }

PCN:

    @inProceedings{yuan2018pcn,
    title     = {PCN: Point Completion Network},
    author    = {Yuan, Wentao and Khot, Tejas and Held, David and Mertz, Christoph and Hebert, Martial},
    booktitle = {3D Vision (3DV), 2018 International Conference on},
    year      = {2018}
    }

Spann3r:

    @article{wang20243d,
    title={3D Reconstruction with Spatial Memory},
    author={Wang, Hengyi and Agapito, Lourdes},
    journal={arXiv preprint arXiv:2408.16061},
    year={2024}
    }