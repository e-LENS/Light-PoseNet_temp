# Light-PoseNet



### Train

```
!python train.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name [모델명]/[Dataset]/[beta500_ex] --beta 500 --gpu 0 --niter 500
```

### Test

```
!python test.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name [모델명]/[Dataset]/[beta500_ex] --gpu 0
```

### KD_train

```
!python KD_train.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name  Student/[Dataset]/[beta500_LossFunction] --LossF [Cross-Similarity] --beta 500 --gpu 0 --niter 300 --T_path [TeacherModel_Path] --save_epoch_freq 5
```

### KD_test

```
!python KD_test.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name Student/[Dataset]/[beta500_LossFunction] --beta 500 --gpu 0 
```
