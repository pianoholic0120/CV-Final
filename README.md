# CV-Final

![Pipeline](./images/Screenshot%202025-05-30%20at%2014.39.23.png)

## Requirements & Installation

You should prepare the following two environments to run our code.

### Reloc3r - generate poses
 - You can directly follow the desciption of the original [reloc3r github codebase](https://github.com/ffrivera0/reloc3r) for detailed installation setup. 
 
 - However, we made slight modification to the file: 
 eval_visloc_pose.py, ./reloc3r/datasets/sevenscenes_retrieval.py, and ./reloc3r/datasets/sevenscenes.py

#### Create the environment using conda

    conda create -n reloc3r python=3.11 cmake=3.14.0
    conda activate reloc3r 
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system

    pip install -r requirements.txt
    # optional: you can also install additional packages to:
    # - add support for HEIC images
    pip install -r requirements_optional.txt    

#### Compile cuda kernels for RoPE

    cd croco/models/curope/
    python setup.py build_ext --inplace
    cd ../../../

- The pre-trained model weights will automatically download when running the evaluation and demo code below.
- Note that the pre-trained weights does not include 7scenes, which is crucial for valid comparison and testing.

### Main body - 2D images & poses to 3D point cloud

#### Create the environment using conda

    conda env create -f environment.yml -n {scenes310}
    conda activate {scenes310}
    pip install -r requirements.txt


## How to run (examples)

### To run reloc3r to generate poses

First change directory to ./reloc3r/

    cd ./reloc3r/

Specify train/test split under ./reloc3r/datasets/sevenscenes_retrieval.py

Generate desired dataset under ./data/7scenes/{scene}/ with all seq-{seq_index}/ with *.color.png, *.depth.png, and *.poses.txt

- Testing sequences doesn't need pose.txt of each images.

Run reloc3r pipeline to generate poses_final.py

    python eval_visloc.py --model "Reloc3rRelpose(img_size=512)" --dataset_db "SevenScenesRetrieval(scene='{}', split='train')" --dataset_q "ne='{}', split='train')" --dataset_q "SevenScenesRetrieval(scene='{}', split='test')" --dataset_relpose "SevenScenesRelpose(scene='{}', pair_id-topk 10nesRetrieval(scene='{}', split={}, resolution={})" --scene "{scene}" --topk 10

Put the generated poses_final.py into the folder of testing sequence to be evaluated

    cd ..

### To reconstruct based on reloc3r-generated poses:
run:

    python cv25s_full_pipeline_reloc3r.py --seq_dir /path/to/the_folder/7scenes/{scenes}/test/seq-{sequence_index}/ --output /path/to/the_file/of/{scenes}-seq-{sequence_index}.ply

### To sweep parameters (voxels, kf) of reconstruction based on ace0-generated poses:

    python cv25s_full_pipeline.py --data_root /path/to/your_file/with/7scenes/ --output_dir /path/to/save/your_results/ --voxels 0.0025,0.003 --kf 1,5

### To compute accuracy w.r.t GT:
    
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
| 297182 | PCN Refined | ❌  | ✅ | ❌ | ✅| Sparse  | 0.20  |  0.34 |
| 297998 | cv25s + Reloc3r | ❌  | ✅(5e-3)| ✅ | ✅| - (300000) | 0.02  |  0.01 |
| 298700 | spann3r | ❌  | ✅(4e-3)| ❌ | ❌| - (300000) | 0.21  |  0.53 |
| 299961 | cv25s + Reloc3r | ❌  | ✅(2.5e-3)| ✅ | ✅| - (300000) | 0.01  |  0.01 |
| 300466 | cv25s + Reloc3r | ❌  | ✅(2e-3)| ✅ | ✅| - (300000) | 0.01  |  0.01 |
| 300486 | combine 300466 & 291723 | ❌  | ✅(2e-3)| ✅ | ✅| - (300000) | 0.01  |  0.01 |

* Pre-processing: kf_every, Q-align
* Post-processing: voxel_grid_size (down-sample), PCN Refinement


### parameters markdown

| ID | Sor_k | Sor_std | r_radius | r_min | voxel_down_sample | # of points | Accuracy | Completeness | 
| :-----| ----: | :----: | :-----| ----: | :----: |:----: | :----: |:----: |
| 297998 | 20  | 2.0| 0.05 | 8| 0.005 |300000 | 0.02  |  0.01 |
| 299961 | 20  | 0.8| 0.03 | 12| 0.0025 |300000 | 0.01  |  0.01 |
| 300466 | 25  | 0.6| 0.025 | 16| 0.002 |300000 | 0.01  |  0.01 |


## External Dependences
1. ACE0
    https://github.com/nianticlabs/acezero
2. Q-Align
    https://github.com/Q-Future/Q-Align
3. COLMAP SfM
    https://github.com/colmap/colmap
4. Reloc3r
    https://github.com/ffrivera0/reloc3r
5. PCN (Point Completion Network)
    https://github.com/wentaoyuan/pcn?tab=readme-ov-file
6. Spann3r
    https://github.com/HengyiWang/spann3r



## Cited
ACE0:

    @article{bhat2023zoedepth,
    title={Zoe{D}epth: Zero-shot transfer by combining relative and metric depth},
    author={Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and M{\"u}ller, Matthias},
    journal={arXiv},
    year={2023}
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