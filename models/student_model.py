import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pickle
import numpy

from options.KD_train_options import TrainOptions

opt = TrainOptions().parse()

# SoftLabel
oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0 * np.pi)  # normalization factor for Gaussians


def gaussian_distribution(y, mu):
    sigma = opt.sigma
    sigma = torch.tensor(sigma, dtype=torch.int8)
    sigma.expand_as(mu)
    # sigma = torch.exp(sigma)
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI


def get_CS(X, Y):
    SS_X = util.getSelfSimilarity(X)
    SS_Y = util.getSelfSimilarity(Y)

    CS = util.getSelfCrossSimilarity(SS_X, SS_Y)

    return CS


def featureLoss(feature_guided, feature_hint):
    return torch.dist(feature_guided, feature_hint)


def CSLoss(CS_A, CS_B):
    MSELoss = nn.MSELoss()
    CSLoss = MSELoss(CS_A, CS_B)

    return CSLoss


class simLoss(nn.Module):
    def __init__(self):
        super(simLoss, self).__init__()

    def forward(self, CS_1_3, pred_S, feature_S):
        if opt.LossF == "RofR":
            pred_S[0] = torch.tensor(pred_S[0], requires_grad=True)
            pred_S[1] = torch.tensor(pred_S[1], requires_grad=True)
            pred_S = torch.cat(pred_S, dim=1)  # [32, 7]

            CS_2_5 = get_CS(feature_S, pred_S)
            loss_cs = CSLoss(CS_1_3, CS_2_5)

        return loss_cs


# Knowledge Distillation loss
alpha = opt.alpha


class StudentModel(BaseModel):
    def name(self):
        return 'StudentModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.isKD = opt.isKD
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.groundtruth = self.Tensor(opt.batchSize, opt.output_nc)

        # load/define networks
        googlenet_weights = None
        if self.isTrain and opt.init_weights != '':
            googlenet_file = open(opt.init_weights, "rb")
            googlenet_weights = pickle.load(googlenet_file, encoding="bytes")
            googlenet_file.close()
            print('initializing the weights from ' + opt.init_weights)
        self.mean_image = np.load(os.path.join(opt.dataroot, 'mean_image.npy'))

        self.netG = networks.define_network(opt.input_nc, None, opt.model,
                                            init_from=googlenet_weights, isTest=not self.isTrain,
                                            isKD=self.isKD,
                                            gpu_ids=self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            # self.criterion = torch.nn.MSELoss()
            # self.simLoss = simLoss()
            self.MSELoss = nn.MSELoss()
            self.simLoss = simLoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, eps=1,
                                                weight_decay=0.0625,
                                                betas=(self.opt.adambeta1, self.opt.adambeta2))
            self.optimizers.append(self.optimizer_G)
            # for optimizer in self.optimizers:
            #     self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')

    def set_input(self, input):
        input_A = input['A']
        groundtruth = input['B']
        self.image_paths = input['A_paths']

        # self.batch_index = input['index_N']

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.groundtruth.resize_(groundtruth.size()).copy_(groundtruth)

    def forward(self):
        self.pred_B = self.netG(self.input_A)

    # no backprop gradients
    def test(self):
        self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward(self, CS_1_3, feature_hint):
        self.loss_G = 0
        self.loss_pos = 0
        self.loss_ori = 0

        pred_S, feature_guided, feature_S = self.pred_B
        # feature_S = feature_S.detach()

        pos_gt = self.groundtruth[:, 0:3]
        ori_gt = F.normalize(self.groundtruth[:, 3:], p=2, dim=1)

        self.loss_pos += self.MSELoss(pred_S[0], pos_gt)
        self.loss_ori += self.MSELoss(pred_S[1], ori_gt)

        # 2. feature loss
        loss_feature = torch.dist(feature_guided, feature_hint)

        # 3. similarity loss

        loss_sim = self.simLoss(CS_1_3, pred_S, feature_S)

        # Loss = response loss + feature loss + similarity loss
        self.loss_G += self.loss_pos + self.loss_ori * self.opt.beta + loss_feature + loss_sim * opt.sigma
        self.loss_G.backward()

    def optimize_parameters(self, CS_1_3, feature_hint):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward(CS_1_3, feature_hint)
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('pos_err', self.loss_pos),
                                ('ori_err', self.loss_ori),
                                ])

        pos_err = torch.dist(self.pred_B[0], self.groundtruth[:, 0:3])
        ori_gt = F.normalize(self.groundtruth[:, 3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(self.pred_B[1])).sum())
        ori_err = 2 * 180 / numpy.pi * torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]

    def get_current_pose(self):
        return numpy.concatenate((self.pred_B[0].data[0].cpu().numpy(),
                                  self.pred_B[1].data[0].cpu().numpy()))

    def get_current_visuals(self):
        input_A = util.tensor2im(self.input_A.data)
        # pred_B = util.tensor2im(self.pred_B.data)
        # groundtruth = util.tensor2im(self.groundtruth.data)
        return OrderedDict([('input_A', input_A)])

    def save(self, label):  # label : latest or epoch#
        self.save_network(self.netG, 'G', label, self.gpu_ids)
