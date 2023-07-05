# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from PIL import Image, ImageColor
import networks
import shutil
def findindex(indices):

    indices_list= []
    for i in range(len(indices)):
        a = indices[i]
        a = a.tolist()
        indices_list.append(a)
    indices_list = list(zip(indices_list[0], indices_list[1], indices_list[2], indices_list[3], indices_list[4]))
    return indices_list

def aggreate(ori_img):

    b,w,h,d = ori_img.shape
    img = ori_img[:,:, :, 0] | ori_img[:,:, :, 1]
    for i in range(2,d):
        img = img | ori_img[:,:, :,i]
    return img
def aggreate_pred(ori_img):

    b,w,h,d = ori_img.shape
    img = torch.add(ori_img[:,:, :, 0], ori_img[:,:, :, 1])
    for i in range(2,d):
        img = torch.add(img,ori_img[:,:, :,i])
    img = img / d
    return img

def load_model(path):
    """Loads Ourmodel and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the Ourmodel
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

from datetime import datetime
currentDateAndTime = datetime.now()
currentTime = currentDateAndTime.strftime("%H%M%S")

def paintAndSave(ori_img, path, index):
    img_slice = ori_img[index:index+1, :,:]
    img_rgb = img_slice.repeat(3, axis=0)
    img_rgb = img_rgb.transpose(1,2,0)
    w = img_rgb.shape[0]
    h = img_rgb.shape[1]
    for i in range(0, w):
        for j in range(0, h):
            if img_rgb[i, j, 0] == 1:
                img_rgb[i, j] = ImageColor.getrgb('hotpink')
            elif img_rgb[i, j, 0] == 2:
                img_rgb[i, j] = ImageColor.getrgb('blue')
            elif img_rgb[i, j, 0] == 3:
                img_rgb[i, j] = ImageColor.getrgb('darkviolet')
            elif img_rgb[i, j, 0] == 4:
                img_rgb[i, j] = ImageColor.getrgb('skyblue')
            elif img_rgb[i, j, 0] == 5:
                img_rgb[i, j] = ImageColor.getrgb('green')
            elif img_rgb[i, j, 0] == 6:
                img_rgb[i, j] = ImageColor.getrgb('orange')
            elif img_rgb[i, j, 0] == 7:
                img_rgb[i, j] = ImageColor.getrgb('red')
    img = Image.fromarray(np.uint8(img_rgb))
    img.save(path)



def visulize(index, iternum, **kwargs):
    path = "../testimages/refinelabel/"
    # shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)
    if 'img' in kwargs:
        print("done")
        img = kwargs['img']
        img=img.detach().clone().cpu().numpy()
        img = img[index][0]
        imgs_normalized = ((img - np.min(img)) / (
                np.max(img) - np.min(img)))
        img = imgs_normalized*255.
        img = Image.fromarray(img.astype(np.uint8))
        img.save(path +"img_" + str(iternum) + '_' + currentTime + ".png")
    if 'label' in kwargs:
        label = kwargs['label']
        label = label.detach().clone().cpu().numpy()
        label_path = path + "label_" + str(iternum) + '_' + currentTime + ".png"
        paintAndSave(label, label_path, index)
    if 'coarse' in kwargs:
        coarse = kwargs['coarse']
        coarse = coarse.detach().clone().cpu().numpy()
        coarse_path = path + "coarse_" + str(iternum) + '_' + currentTime + ".png"
        paintAndSave(coarse, coarse_path, index)
    if 'coarsep' in kwargs:
        coarsep = kwargs['coarsep']
        coarsep = coarsep.detach().clone().cpu().numpy()
        coarse_path = path + "coarsep_" + str(iternum) + '_' + currentTime + ".png"
        paintAndSave(coarsep, coarse_path, index)
    if 'refinel' in kwargs:
        refinel = kwargs['refinel']
        refinel = refinel.detach().clone().cpu().numpy()
        refine_path = path + "refinel_" + str(iternum) + '_' + currentTime + ".png"
        paintAndSave(refinel,refine_path,index)
    if 'refineh' in kwargs:
        refineh = kwargs['refineh']
        refineh = refineh.detach().clone().cpu().numpy()
        refine_path = path + "refineh_" + str(iternum) + '_' + currentTime + ".png"
        paintAndSave(refineh,refine_path,index)
    if 'refine' in kwargs:
        refine = kwargs['refine']
        refine = refine.detach().clone().cpu().numpy()
        refine_path = path + "refine_" + str(iternum) + '_' + currentTime + ".png"
        paintAndSave(refine,refine_path,index)
    if 'refinelh' in kwargs:
        refine = kwargs['refinelh']
        refine = refine.detach().clone().cpu().numpy()
        refine_path = path + "refinelh_" + str(iternum) + '_' + currentTime + ".png"
        paintAndSave(refine,refine_path,index)
    if 'refinehl' in kwargs:
        refine = kwargs['refinehl']
        refine = refine.detach().clone().cpu().numpy()
        refine_path = path + "refinehl_" + str(iternum) + '_' + currentTime + ".png"
        paintAndSave(refine,refine_path,index)




def converToSlice(input):

    D = input.size(-1)
    input2d = input[..., 0]
    for i in range(1, D):
        input2dtmp = input[..., i]
        input2d = torch.cat((input2d, input2dtmp), dim=0)

    return input2d


def converToThreeSlice(input):
    if len(input.shape) == 5:
        B, C, H, W, D = input.size()
        input2d = input[:, :, :, :, 0:2]
        single = input[:, :, :, :, 0:1]
        input2d = torch.cat((input2d, single), dim=4)
        for i in range(D - 2):
            input2dtmp = input[:, :, :, :, i:i + 3]
            input2d = torch.cat((input2d, input2dtmp), dim=0)
        f1 = input[:, :, :, :, D - 2: D]
        f2 = input[:, :, :, :, D - 1: D]
        ff = torch.cat((f1, f2), dim=4)
        input2d = torch.cat((input2d, ff), dim=0)
        input2d = input2d[:, 0, ...]  # squeeze the last channel C (BxD,H,W,3)
        input2d = input2d.permute(0, 3, 1, 2)  # BxD,3,H,W
    else:
        B, H, W, D = input.size()
        input2d = input[..., 0]
        for i in range(1, D):
            input2dtmp = input[..., i]
            input2d = torch.cat((input2d, input2dtmp), dim=0)
    return input2d

def converToVolumn(input):
    """

    :param input: d,2,h,w
    :return: 1,2,h,w,d
    """
    if len(input.shape)==4:
        input = torch.unsqueeze(input=input,dim=-1)
        D, C, H, W, _ = input.size()

        input2d = input[0:1, :, :, :,:]
        for i in range(1, D):
            input2dtmp = input[i:i+1, :, :, :,:]
            input2d = torch.cat((input2d, input2dtmp), dim=-1)
    return input2d

from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list