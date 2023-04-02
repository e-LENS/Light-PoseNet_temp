import time
from options.KD_train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import KD_create_Tmodel, KD_create_Smodel
from util.visualizer import Visualizer
import copy
import util.util as util
from PIL import Image

opt = TrainOptions().parse()
## SEEDING
import torch
import numpy as np
import random

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
## SEEDING

# import wandb
# wandb.init(project="LightPoseNet_asanSquare_486_122", entity="e-lens-", name="cls2_KD_MSE_asanSquare_0.7_300")

# Set Dataloader
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

visualizer = Visualizer(opt)
total_steps = 0

# Set Teacher Model Parameter
opt_T = copy.deepcopy(opt)
opt_T.isTrain = False
teacher = KD_create_Tmodel(opt_T)
teacher.netG.load_state_dict(torch.load(opt_T.T_path))

# Set Student Model
student = KD_create_Smodel(opt)  # Criterion : KDLoss, Optimizer : Adam

# RofR map for full dataset
gt_pose = data_loader.dataset.A_poses
gt_path = data_loader.dataset.A_paths
gt_image = []

for i, path in enumerate(gt_path):
    A_img = Image.open(path).convert('RGB')
    A_img = np.array(A_img).astype(np.float32)
    A_img = torch.tensor(A_img)

    gt_image.append(A_img)

gt_img = torch.stack(gt_image, dim=0)

gt_pose = torch.tensor(gt_pose)

gt_RofR = util.getRofR(gt_img,gt_pose)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        teacher.set_input(data)
        teacher.test()
        pred_T, feature_T = teacher.pred_B

        student.set_input(data)
        student.optimize_parameters(dataset_size, gt_RofR, feature_T)

        if total_steps % opt.print_freq == 0:
            errors = student.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

            # for k, v in errors.items():
            # wandb.log({'%s'%k :v})

            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        student.save('latest')
        student.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    student.update_learning_rate()














