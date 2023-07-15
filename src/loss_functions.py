'''
PyTorch Loss functions for semantic segmentation
1. Focal Loss
2. Weighted Focal Loss
3. Tversky Loss
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F # For Tversky Loss

class FocalLoss2(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=0, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction=reduction)
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_index=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        
        target = target * ((target != self.ignore_index) & (target != 15)).long()

        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, class_weights=None, ignore_index=0):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, input, target):
        target = target * ((target != self.ignore_index) & (target != 15)).long()

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        # print("DEBUGGING: logpt value1: ", logpt.shape)
        logpt = logpt.gather(1, target)
        # print("DEBUGGING: logpt value2: ", logpt.shape)
        logpt = logpt.view(-1)
        # print("DEBUGGING: logpt value3: ", logpt.shape)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        if self.class_weights is not None:
            class_weights = self.class_weights.type_as(input)
            weights = class_weights.gather(0, target.view(-1))
            logpt = logpt * weights

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-7, n_classes=None, ignore_index=0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.n_classes = n_classes

    def forward(self, input, target):
        target = target * ((target != self.ignore_index) & (target != 15)).long()
        target = F.one_hot(target, self.n_classes+1).permute(0, 3, 1, 2).float()
        input = F.softmax(input, dim=1)

        tp = torch.sum(input * target, dim=(2, 3))
        fp = torch.sum(input * (1 - target), dim=(2, 3))
        fn = torch.sum((1 - input) * target, dim=(2, 3))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1 - tversky.mean()

        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0, smooth=1e-7, n_classes=None, ignore_index=0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.n_classes=n_classes

    def forward(self, input, target):
        target = target * ((target != self.ignore_index) & (target != 15)).long()
        target = F.one_hot(target, self.n_classes+1).permute(0, 3, 1, 2).float()
        input = F.softmax(input, dim=1)

        tp = torch.sum(input * target, dim=(2, 3))
        fp = torch.sum(input * (1 - target), dim=(2, 3))
        fn = torch.sum((1 - input) * target, dim=(2, 3))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        loss = focal_tversky.mean()

        return loss
