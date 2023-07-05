import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch.nn as nn
from torch.autograd import Variable
# HD loss and boundary loss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg


def _one_hot_encoder(input_tensor,n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(dim=1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def per_structure_dice(logits, labels, epsilon=1e-10, sum_over_batches=False, use_hard_pred=True):
    '''
    Dice coefficient per subject per label
    :param logits: network output
    :param labels: groundtruth labels
    :param epsilon: for numerical stability
    :param sum_over_batches: Calculate intersection and union over whole batch rather than single images
    :return: tensor shaped (tf.shape(logits)[0], tf.shape(logits)[-1]) (except when sum_over_batches is on)
    '''

    ndims = len(logits.shape)
    nclasses = logits.size(1)
    prediction = F.softmax(logits,dim=1)
    labels = _one_hot_encoder(labels,nclasses)
    if use_hard_pred:
        # This casts the predictions to binary 0 or 1
        hard_pred = torch.argmax(prediction,dim=1)
        prediction = _one_hot_encoder(hard_pred,nclasses)

    intersection = prediction * labels

    if ndims == 5:
        reduction_axes = [2,3,4]
    else:
        reduction_axes = [2,3]

    if sum_over_batches:
        reduction_axes = [0] + reduction_axes

    # Reduce the maps over all dimensions except the batch and the label index
    i = torch.sum(intersection, dim=reduction_axes)
    l = torch.sum(prediction, dim=reduction_axes)
    r = torch.sum(labels, dim=reduction_axes)

    dice_per_img_per_lab = 2 * i / (l + r + epsilon)

    return dice_per_img_per_lab


def dice_loss_foreground(logits, labels, epsilon=1e-10, only_foreground=False, sum_over_batches=True):
    '''
    Calculate a dice loss defined as `1-foreround_dice`. Default mode assumes that the 0 label
     denotes background and the remaining labels are foreground. Note that the dice loss is computed
     on the softmax output directly (i.e. (0,1)) rather than the hard labels (i.e. {0,1}). This provides
     better gradients and facilitates training.
    :param logits: Network output before softmax
    :param labels: ground truth label masks
    :param epsilon: A small constant to avoid division by 0
    :param only_foreground: Exclude label 0 from evaluation
    :param sum_over_batches: calculate the intersection and union of the whole batch instead of individual images
    :return: Dice loss
    '''

    dice_per_img_per_lab = per_structure_dice(logits=logits,
                                              labels=labels,
                                              epsilon=epsilon,
                                              sum_over_batches=sum_over_batches,
                                              use_hard_pred=False)

    if only_foreground:
        if sum_over_batches:
            loss = 1 - torch.mean(dice_per_img_per_lab[1:])
        else:
            loss = 1 - torch.mean(dice_per_img_per_lab[:, 1:])
    else:
        loss = 1 - torch.mean(dice_per_img_per_lab)

    return loss


def compute_sdf1_1(img_gt, out_shape):
    """
    compute the normalized signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1, 1]
    """

    img_gt = img_gt.astype(np.uint8)

    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):
        # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                        np.max(posdis) - np.min(posdis))
                sdf[boundary == 1] = 0
                normalized_sdf[b][c] = sdf
    return normalized_sdf


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.reshape(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    pc = outputs_soft[:, 1, ...]
    dc = gt_sdf[:, 1, ...]
    multipled = torch.mul(pc, dc)
    bd_loss = multipled.mean()
    return bd_loss


class InstanceWeightedBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        unweightedloss = F.binary_cross_entropy(input, target, reduction="none")
        instance_weight = target.clone()
        #         print(instance_weight.size())
        #         print(instance_weight[0,0,245:248,245:248])
        instance_weight[instance_weight == 0] = 0.3
        instance_weight[instance_weight == 1] = 0.7
        weightedloss = unweightedloss * instance_weight
        loss = torch.mean(weightedloss)
        return loss


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss_w(score, target, w):
    score = score[w != 0]
    target = target[w != 0]
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    ## p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True) / torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss

    # def mse_loss(input_logits, target_logits):
    #     """Takes softmax on both sides and returns MSE loss
    #
    #     Note:
    #     - Returns the sum over all examples. Divide by the batch size afterwards
    #       if you want the mean.
    #     - Sends gradients to inputs but not the targets.
    #     """
    #     assert input_logits.size() == target_logits.size()
    #     input_softmax = F.softmax(input_logits, dim=1)
    #     target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def sigmoid_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()

    mse_loss = (input_logits - target_logits) ** 2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2) ** 2)


def weighted_celoss(output, target, mask_fill):
    # input:[1,3,256,256]
    # target:[1,256,256] in {0,1,2}
    # mask_fill:[1,256,256]
    # 这一步不用改，因为本来就是用的PyTorch的内置方法
    ls = torch.nn.LogSoftmax(dim=1)
    log_softmax = ls(output)
    bsize, h, w = target.shape[0], target.shape[1], target.shape[2]
    loss = 0
    # 由于batchsize一般都不会很大，因此该for循环花费时间很少
    for b in range(bsize):
        # 下面是本次更改的部分
        # 获取每个像素点的真实类别标签
        ind = target[b, ...].type(torch.int64).unsqueeze(0)
        # print('ind:',ind.shape)#torch.Size([1, 256, 256])
        # 获取预测得到的每个像素点的类别取值分布（3代表类别）
        pred_3channels = log_softmax[b, ...]
        # print('pred_3channels:',pred_3channels.shape)#torch.Size([3, 256, 256])
        # 使用gather，在第0个维度（类别所在维度）上用ind进行索引得到每个像素点的value
        pred = -pred_3channels.gather(0, ind)
        # print('pred:',pred.shape)#torch.Size([1, 256, 256])
        # 添加了这句代码，通过两者的点乘实现了对每个像素点的加权
        pred = pred * mask_fill
        # 求这些像素点value的平均值，并累加到总的loss中
        current_loss = torch.mean(pred)
        loss += current_loss
    return loss / bsize


def PPC(contrast_logits, contrast_target):
    loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long())

    return loss_ppc


def PPD(contrast_logits, contrast_target):
    logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
    loss_ppd = (1 - logits).pow(2).mean()

    return loss_ppd


# class PixelPrototypeCELoss(nn.Module):
#     def __init__(self):
#         super(PixelPrototypeCELoss, self).__init__()
#         self.loss_ppc_weight = 0.01
#         self.loss_ppd_weight = 0.01
#
#     def forward(self, proto_seg, contrast_logits, contrast_target, target):
#         loss_ppc = PPC(contrast_logits, contrast_target)
#         loss_ppd = PPD(contrast_logits, contrast_target)
#         outputs_soft = F.softmax(proto_seg, dim=1)
#         loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :], target == 1)
#         loss = (F.cross_entropy(proto_seg, target) + loss_seg_dice) * 0.5
#         return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd

class PixelPrototypeCELoss(nn.Module):
    def __init__(self):
        super(PixelPrototypeCELoss, self).__init__()
        self.loss_ppc_weight = 0.01
        self.loss_ppd_weight = 0.01

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = PPC(contrast_logits, contrast_target)
        loss_ppd = PPD(contrast_logits, contrast_target)
        return  self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd


class DiceLoss(nn.Module):
    def __init__(self, n_classes,weights=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weights = weights

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(dim=1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if self.weights is None:
            self.weights = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * self.weights[i]
        return loss / self.n_classes


def CEDice(pred, target,nclass,weights=None):
    if weights != None:
        weights = torch.from_numpy(np.array(weights)).float().cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    dice_loss = DiceLoss(nclass,weights=weights)
    loss_cls_ce = criterion(pred, target)
    outputs_soft = F.softmax(pred, dim=1)
    loss_seg_dice = dice_loss(outputs_soft, target)
    loss = 0.5 * (loss_cls_ce + loss_seg_dice)
    return loss


import cv2 as cv
import numpy as np

import torch
from torch import nn

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
"""


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
            self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
                pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        return loss


################ rebuttal ############################333

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)  # N*H*W,1

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1) # N*H*W
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class AsymFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True,dataset='acdc'):
        """

        :param dataset: ACDC:[0,0,1,0] MMWHS: [0,1,1,0,0,0,0,1]
        :param gamma:
        :param alpha:
        :param size_average:
        """
        super(AsymFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        if dataset == 'acdc':
            self.rare = [0,1,1,1]
        else:
            self.rare = [0,1,1,1,1,1,1,1]

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)  # N*H*W,1

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1) # N*H*W
        pt = Variable(logpt.data.exp())
        for i,item in enumerate(self.rare):
            if item == 1:
                pt[target[:,0]==i]=0
            elif item == 0.5:
                pt[target[:, 0] == i] = 0.5

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()