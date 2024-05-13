import os
import argparse
import logging
from networks.unet_proto_hybrid import UNetProto
import h5py
import math
import nibabel as nib
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from test_util import test_all_case
import sys
import numpy as np
from skimage import measure
from utils.util import converToSlice
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../Ourmodel/contrast/"+FLAGS.model+"/"
test_save_path = "../Ourmodel/prediction/inferTime/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 8
with open(FLAGS.root_path + '/train_c.txt', 'r') as f:
    alllist = f.readlines()
image_list = [FLAGS.root_path + "/cardiac_smooth_h5py/" + item.replace('\n', '') for item in alllist[-4:]]


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in [1, 2, 3, 4, 5, 6, 7]:

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img

def test_all_case(net, image_list, num_classes, patch_size=(192, 192, 96), stride_xy=18, stride_z=4, save_result=False, test_save_path=None, do_postprocessing=False):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if do_postprocessing:
            print("postprocessing......")
            prediction = keep_largest_connected_components(prediction)
        if np.sum(prediction)==0:
            single_metric = (0,0,0)
        else:
            single_metric = calculate_metric_percase_avg(prediction, label[:],8)
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test(image,xs,ys,zs,patch_size,net,num_classes):

    test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
    test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
    test_patch = torch.from_numpy(test_patch).cuda() # h,w,d
    test_2d = converToSlice(test_patch)
    outputs=net(x_2d=test_2d,
                          x_3d =test_patch,
                          label=None,
                          use_prototype=False) # d,2,h,w
    output3d = outputs["cls_seg_3d"]
    result = F.softmax(output3d, dim=1)
    return result

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape
    print(w,h,d)
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape
    print(ww,hh,dd)

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                result = test(image,xs,ys,zs,patch_size,net,num_classes)
                result = result.cpu().data.numpy()
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + result[0, :, :, :, :]
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, hd, asd

def binary_hd3d(s,g):
    """
    dice score of 3d binary volumes
    inputs:
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert(len(s.shape)==3)
    w,h,d = s.shape
    hd_list = []
    for i in range (d):
        if np.sum(g[:,:,i])==0 or np.sum(s[:,:,i]) == 0:
            continue
        else:
            dice = metric.binary.hd95(s[:,:,i],g[:,:,i])
        hd_list.append(dice)
    return np.mean(hd_list)


def calculate_metric_percase_avg(pred, gt, nclasses):
    """

    :param pred: w,h,d
    :param gt: w,h,d
    :param nclasses:
    :return:
    """

    dice_list = []
    hd_list = []
    assd_list = []
    for c in range(1, nclasses):
        pred_single = pred.copy()
        pred_single[pred_single != c] = 0
        pred_single[pred_single != 0] = 1
        gt_single = gt.copy()
        gt_single[gt_single!=c]=0
        gt_single[gt_single != 0] = 1


        dice_list.append(metric.binary.dc(pred_single, gt_single))
        hd_list.append(metric.binary.hd95(pred_single, gt_single))
        assd_list.append(metric.binary.assd(pred_single, gt_single))


    dice_mean = np.mean(dice_list)
    hd_mean = np.mean(hd_list)
    assd_mean = np.mean(assd_list)
    """
           myocardium of the left ventricle (MYO):205 --> 1
           left atrium blood cavity (LA): 420 --> 2
           left ventricle blood cavity (LV): 500 --> 3
           right atrium blood cavity (RA): 550 --> 4
           right ventricle blood cavity (RV): 600 --> 5
           ascending aorta (AA): 820 --> 6
           pulmonary artery (PA): 850 --> 7
           """
    print('Metrics:')
    print('MYO :%.1f    %.1f    %.1f' % (dice_list[0],hd_list[0],assd_list[0]))
    print('LA :%.1f     %.1f    %.1f' % (dice_list[1],hd_list[1],assd_list[1]))
    print('LV :%.1f     %.1f    %.1f' % (dice_list[2],hd_list[2],assd_list[2]))
    print('RA :%.1f     %.1f    %.1f' % (dice_list[3],hd_list[3],assd_list[3]))
    print('RV :%.1f     %.1f    %.1f' % (dice_list[4],hd_list[4],assd_list[4]))
    print('AA :%.1f     %.1f    %.1f' % (dice_list[5],hd_list[5],assd_list[5]))
    print('PA :%.1f     %.1f    %.1f' % (dice_list[6],hd_list[6],assd_list[6]))
    print('dice mean :%.1f  %.1f    %.1f' % (dice_mean,hd_mean,assd_mean))
    return dice_mean,hd_mean,assd_mean

    print('Dice:')
    print('MYO :%.1f' % (dice_list[0]))
    print('LA :%.1f' % (dice_list[1]))
    print('LV :%.1f' % (dice_list[2]))
    print('RA :%.1f' % (dice_list[3]))
    print('RV :%.1f' % (dice_list[4]))
    print('AA :%.1f' % (dice_list[5]))
    print('PA :%.1f' % (dice_list[6]))
    print('dice mean :%.1f' % (dice_mean))

    return dice_mean



def test_calculate_metric(epoch_num):
    net = UNetProto(
        inchannel=1,
        nclasses=num_classes,
        proj_dim=64,
        projection="v1",
        l2_norm=True,
        proto_mom=0.999,
        sub_proto_size=2,
    ).cuda()

    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    checkponit = torch.load(save_mode_path,map_location="cuda:0")

    net.load_state_dict(checkponit)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(80,80,80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    logging.basicConfig(filename=test_save_path + 'nopo_'+'iter_' + str(epoch_num) + "log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("average metric is {}".format(avg_metric))
    return avg_metric








