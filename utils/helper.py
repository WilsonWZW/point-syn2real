import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.nn.functional as F

def save_confusion_matrix(y_true, y_pred, disp_labels, save_path='confusion_matrix.png'):
    """ Save confusion matrix to an image for paper """

    idx_labels = [x for x in range(len(disp_labels))]

    cm = confusion_matrix(y_true, y_pred, labels=idx_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)

    disp.plot(cmap='GnBu')
    plt.xticks(rotation=30)
    plt.savefig(save_path)
    plt.close(disp.figure_)


def stats_print(dataset):
    """ Print percent of classes in the given dataset """
    data_label = dataset.data.y.tolist()
    data_cat = dataset.categories
    data_count = Counter(data_label)

    print ('Total number of items: %d' % len(data_label))
    percent_list = []
    for idx, cat in enumerate(data_cat):
        percent = data_count[idx]/len(data_label)
        percent_list.append(percent)
        print ('%10s: %4d - %2.2f%%' % (cat, data_count[idx], 100*percent))

    return percent_list

def get_class_weight(train_percent, test_percent, flag=None):
    """ Reciprocal of category percentage and normalise it """

    if flag == 'source_set':
        reci_percent = np.reciprocal(train_percent)
        norm = np.linalg.norm(reci_percent)
        loss_class_weights = reci_percent/norm
    elif flag == 'target_set':
        train_reci = np.reciprocal(train_percent)
        reci_percent = train_reci * test_percent
        norm = np.linalg.norm(reci_percent)
        loss_class_weights = reci_percent/norm
    else:
        loss_class_weights = [1] * len(train_percent)

    return loss_class_weights

class RandomJitter(object):
    """ This class generates and adds random jitter.

    Attributes:
        sigma (float): The sigma variable controls the noise level. Default: 0.01.
        clip (float): The clip variable limits the noise. Default: 0.02.
    """
    def __init__(self, sigma=0.01, clip=0.02):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        """ 
        The function to generate and add jitter.
        
        Parameters:
            data (array): size [N x C]

        Returns:
            data (array): the tensor after applying jitter noise.
        """

        N, C = data.pos.shape[0], data.pos.shape[1]
        noise = np.clip(self.sigma * np.random.randn(N, C), -1*self.clip, self.clip)
        data.pos = data.pos + np.float32(noise)

        return data

    def __repr__(self):
        return '{}(sigma={}, clip={})'.format(self.__class__.__name__, self.sigma, self.clip)


class MMD_loss(nn.Module):
    """ 
    Compute Maximum Mean Discrepancy (MMD) loss in PyTorch.
    Source: https://github.com/jindongwang/transferlearning/blob/master/code/deep/DaNN/mmd.py
    """
    
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    # source: [B x feat_dim]
    # target: [B x feat_dim]
    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class EntropyLoss(nn.Module):
    """
    Compute Entropy Loss in PyTorch.

    """
    def __init__(self, reduction = 'mean'):
        super(EntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        """ 
        Parameters:
            x: Logits output. Softmax has not been applied on x.

        Return:
            loss: Entropy loss.
        """

        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(1)

        if self.reduction == 'mean':
            loss = b.mean()
        elif self.reduction == 'sum':
            loss = b.sum()
        else:
            loss = b

        return loss