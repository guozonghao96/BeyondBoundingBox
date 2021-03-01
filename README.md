# CVPR2021-1616

## Introduction
The code includes training and inference procedures for **Beyond Bounding-Box: Convex-hull Feature Adaptation for Oriented and Densely Packed Object Detection**.

## Installation
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
