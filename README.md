# SCKD :  Self-Cross Similarity Knowledge Distillation for Light Camera Pose Regressor
The similarity based knowledge distillation mehod to compress the 6-dof Pose Regressor model so that it can show high performance on 5W low-power environment

## Architecture 

<p align="center"><img src="https://github.com/e-LENS/SCKD/assets/108324590/92b748dc-2ec6-42f4-98fc-bab1fef07d37" width="300" height="300"></p>


## Requirements
* LINUX
* Colab
  * Python 3
  * CPU : Intel Xeon CPU 2.3 GHHz(Dual-Core)
  * GPU : Nvidia Tesla T4
  * GPU merory : 8GB

## Installation
```
git clone https://github.com/e-LENS/SCKD.git
cd SCKD
pip install -r requirements.txt
```

## Dataset
* Outdoor dataset

  Download [King's College dataset](https://www.repository.cam.ac.uk/handle/1810/251342) under `datasets` folder
  
* Indoor dataset

  Download [7 Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) under `datasets` folder

```
datasets/KingsCollege
datasets/Chess
datasets/Fire
datasets/Heads
datasets/Office
datasets/Pumpkin
datasets/RedKitchen
datasets/Stairs
```

* Compute the mean image for each dataset

```
python util/compute_image_mean.py --dataroot datasets/[Dataset_name] --height 256 --width 455 --save_resized_imgs
```

* 6DoF 전처리

> 7Scenes dataset은 Position & Orientation label을 4x4 matrix 로 제공하나 
> 위 PoseNet은 position(X,Y,Z) & Orientation quaternion(W,P,Q,R) 의 7dimension vector label을 사용해 label값을 변환하는 dataset preprocessing이 필요하다.

해당 github의
`
posenet-pytorch/7scenes_preprocessing.py
`
파일을 이용해 

4x4 => position(X,Y,Z) & Orientation quaternion(W,P,Q,R) 로 label을 변환한다. 


## SCKD

Train & Test the PoseNet model on each dataset


### Pretrained model
    'resnet18im': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34im': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50im': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101im': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152im': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',


### Train Teacher Model 
```
!python train.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name [model]/[Dataset]/[beta500_ex] --beta 500 --gpu 0 --niter 500 --batchSize32 --lr 0.001
```

### Test Teacher Model
```
!python test.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name [model]/[Dataset]/[beta500_ex] --gpu 0
```

### Train Student Model
```
!python KD_train.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name [student model]/[Dataset]/[beta500_bt_lr_m#1_m#2_scaling] --beta 500 --gpu 0 --niter 500 --T_model [ resnet34 | resnet50 | resnet101]  --T_path [TeacherModel_Path] --save_epoch_freq 5 --SCmodule [ 0 | 1 | 2 | 3 | 4 | 5 ] --hintmodule [ 0 | 1 | 2 | 3 | 4 | 5 ] [--SCKL]
```
- `SCmodule` 과 `hintmodule` option layer는 list type 으로 선택가능

### Test Student Model
```
!python KD_test.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name [student model]/[Dataset]/[beta500_bt_lr_m#1_m#2_scaling] --beta 500 --gpu 0 
```

* 학습된 모델 및 학습 결과는 `./checkpoints/[name]`에 저장됨
* 학습된 모델의 테스트 결과는 `./results/[name]`에 저장됨



---
### reference

참고한 repo
- https://github.com/hazirbas/poselstm-pytorch
- https://github.com/HobbitLong/RepDistiller

## Result 

__** ResNet Model Size  **__

|   Backbone Models   | ResNet50(Teacher) | ResNet34(Teacher) | ResNet18(Student) | 
| :----------------:  | :---------------: | :---------------: | :---------------: |
|       Size(MB)      |       102.45      |       87.27       |       __46__      |
| Number of Parameter |     25,613,383    |    21,817,159     |   __11,708,999__  |


__** Inference time at Jetson Nano 2GB developer kit  **__

|   Backbone Models   | ResNet50(Teacher) | ResNet34(Teacher) | ResNet18(Student) | 
| :----------------:  | :---------------: | :---------------: | :---------------: |
| Median Inference Time (clock) |      125      |       90       |     __50__     |


__** Comparision with other KD method for Regression problems  **__

|   Model   | Shop Facade |Stairs | 
| :----------------:  | :---------------: | :---------------: | 
|       Vanilla KD      |     8.25m /9.17   |       0.39m /13.08     |  
| M.U's KD |     1.45m / 7.58    |    0.39m / 15.27    | 
| Self-Similarity KD |     0.94m / 6.56     |  0.37m / 13.63    | 
| __Ours__ |     __0.93m / 5.96__   |    __0.33m / 13.02__     | 
