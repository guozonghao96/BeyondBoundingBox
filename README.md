# Beyond Bounding-Box: Convex-hull Feature Adaptation for Oriented and Densely Packed Object Detection.(CVPR2021)

## Introduction
Detecting oriented and densely packed objects remains challenging for spatial feature aliasing caused by the intersection of reception fields between objects. In this paper, we propose a convex-hull feature adaptation (CFA) approach for configuring convolutional features in accordance with oriented and densely packed object layouts. CFA is rooted in convex-hull feature representation, which defines a set ofdynamically predicted feature points guided by the convex intersection over union (CIoU) to bound the extent of objects. CFA pursues optimal feature assignment by constructing convex-hull sets and dynamically splitting positive or negative convex-hulls. By simultaneously considering overlapping convex-hulls and objects and penalizing convex-hulls shared by multiple objects, CFA alleviates spatial feature aliasing towards optimal feature adaptation. Experiments on DOTA and SKU110KR datasets show that CFA significantly outperforms the baseline approach, achieving new state-of-the-art detec-
tion performance.

![Motivation](./docs/motivation.pdf)

The framework of the CFA is shown as following:

![Framework](./docs/framework.pdf)

# Thanks mmlab for re-implementating my code in [mmrotate](https://github.com/open-mmlab/mmrotate). Welcome to use it and cite my paper.

## Installation

My official implementation is as following:

Detection framework is based on the [MMDetection v1.1.0](https://github.com/open-mmlab/mmdetection/tree/v1.1.0).

Please refer to the **Installation** of MMDetection to complete the code environment.

## Dataset
Please refer to [DOTA](https://captain-whu.github.io/DOTA/index.html) to get the training, validation and test set.

Before training, the image-splitting process must be carried out. Check the [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit).

## Visualization Demo
We upload some validation images in demo/demo_datasets for visualization.

The detection results for these images is saved in demo/bbox_predict.pkl.

Use *Jupyter Notebook* to run demo/demo.ipynb to visualize the results.

## Training and Inference 

Create a training and inference shell script contains following command.

```
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
GPUS=1

DIR=path_to_save_model
CONFIG=dota_configs/beyond_bounding_boxes_demo.py
./tools/dist_train.sh ${CONFIG} ${GPUS} --work_dir ${DIR} --gpus ${GPUS} --autoscale
./tools/dist_test.sh ${CONFIG} ${DIR}/latest.pth ${GPUS} --out ${DIR}/bbox_predict.pkl
```

## Evaluation
[DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) supplies the evalution details.  


## Citing

```
@inproceedings{Guo_2021CVPR_CFA,
  author    = {Zonghao Guo, Chang Liu, Xiaosong Zhang, Jianbin Jiao, Xiangyang Ji and Qixiang Ye},
  title     = {Beyond Bounding-Box: Convex-hull Feature Adaptation for Oriented and Densely Packed Object Detection},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2021}
}
```
