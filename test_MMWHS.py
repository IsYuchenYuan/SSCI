import os
import argparse
import time
import logging
import h5py
import math
import nibabel as nib
import sys
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage import measure
from einops import rearrange

from networks.unet_proto_hybrid import UNetProto
from utils.util import converToSlice

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='', help='data path')
parser.add_argument('--model', type=str,  default='', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--our', type=bool,  default=True, help='GPU to use')
FLAGS = parser.parse_args()
ours = FLAGS.our
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

snapshot_path = ""
test_save_path = ""

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 8
with open(FLAGS.root_path + '/train_c.txt', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/cardiac_smooth_h5py/" + item.replace('\n', '') for item in image_list[-4:]]

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

def test_all_case(net, image_list, num_classes, patch_size=(192, 192, 96), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, do_postprocessing=True):
    dice_list = []
    assd_list = []
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        time1 = time.time()
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        time2 = time.time()
        print('inference cost {}'.format(time2-time1))
        if do_postprocessing:
            print("postprocessing......")
            prediction = keep_largest_connected_components(prediction)

        if np.sum(prediction)==0:
            single_metric = (0,0,0)
        else:

            for c in range(1, num_classes):
                pred_single = prediction.copy()
                pred_single[pred_single != c] = 0
                pred_single[pred_single != 0] = 1
                gt_single = label.copy()
                gt_single[gt_single != c] = 0
                gt_single[gt_single != 0] = 1
                if np.sum(pred_single)==0:
                    assd_list.append(100)
                else:
                    assd_list.append(metric.binary.assd(pred_single, gt_single))
                dice_list.append(metric.binary.dc(pred_single, gt_single))


        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")

    image_num = len(dice_list) // (num_classes - 1)
    dice_arr = np.reshape(dice_list, [image_num, -1]).transpose()

    dice_mean = np.mean(dice_arr, axis=1)
    dice_std = np.std(dice_arr, axis=1)
    logging.info('Dice:')
    logging.info('MYO :%.4f(%.4f)' % (dice_mean[0], dice_std[0]))
    logging.info('LA :%.4f(%.4f)' % (dice_mean[1], dice_std[1]))
    logging.info('LV :%.4f(%.4f)' % (dice_mean[2], dice_std[2]))
    logging.info('RA :%.4f(%.4f)' % (dice_mean[3], dice_std[3]))
    logging.info('RV :%.4f(%.4f)' % (dice_mean[4], dice_std[4]))
    logging.info('AA :%.4f(%.4f)' % (dice_mean[5], dice_std[5]))
    logging.info('PA :%.4f(%.4f)' % (dice_mean[6], dice_std[6]))
    logging.info('dice mean :%.4f' % np.mean(dice_mean))

    assd_arr = np.reshape(assd_list, [image_num, -1]).transpose()

    assd_mean = np.mean(assd_arr, axis=1)
    assd_std = np.std(assd_arr, axis=1)
    logging.info('Assd:')
    logging.info('MYO :%.4f(%.4f)' % (assd_mean[0], assd_std[0]))
    logging.info('LA :%.4f(%.4f)' % (assd_mean[1], assd_std[1]))
    logging.info('LV :%.4f(%.4f)' % (assd_mean[2], assd_std[2]))
    logging.info('RA :%.4f(%.4f)' % (assd_mean[3], assd_std[3]))
    logging.info('RV :%.4f(%.4f)' % (assd_mean[4], assd_std[4]))
    logging.info('AA :%.4f(%.4f)' % (assd_mean[5], assd_std[5]))
    logging.info('PA :%.4f(%.4f)' % (assd_mean[6], assd_std[6]))
    logging.info('assd mean :%.4f' % np.mean(assd_mean))


    return np.mean(dice_mean)


def test(image,xs,ys,zs,patch_size,net,num_classes):

    test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
    test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
    test_patch = torch.from_numpy(test_patch).cuda() # h,w,d
    input2d = converToSlice(test_patch)
    outputs = net(x_2d=input2d,
                  x_3d=test_patch,
                  label=None,
                  use_prototype=False)
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
    checkpoint_state = torch.load(save_mode_path,map_location="cuda:0")
    if ours:
        checkpoint_state = checkpoint_state

    net.load_state_dict(checkpoint_state)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(80,80,80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    logging.info("average metric is {}".format(avg_metric))
    return avg_metric


if __name__ == '__main__':

    iteration = 6000
    logging.basicConfig(filename=test_save_path + 'iter_' + str(iteration) + "log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    metric = test_calculate_metric(iteration)
