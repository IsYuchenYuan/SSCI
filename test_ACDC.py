import os
import argparse
from networks.unet_proto import UNetProto
import torch
import torch.nn.functional as F
import numpy as np
import medpy.metric.binary as mmb
import nibabel as nib
from collections import defaultdict
from skimage import transform,measure
import logging
import sys
from scipy.ndimage import zoom

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='', help='data path')
parser.add_argument('--model', type=str,  default='', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--our', type=bool,  default=True, help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

snapshot_path = ""
test_save_path = ""

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 4
batch_size = 1
pathsize = 256
data_dir= FLAGS.root_path
with open('../data/ACDC/' + 'train.txt', 'r') as f:
    lists = f.readlines()
image_list = lists[80:]
image_list = [item.replace('\n', '') for item in image_list]


def crop_ROI(image,gt):
    # crop the ROI
    w, h, d = gt.shape
    tempL = np.nonzero(gt)
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])
    minz, maxz = np.min(tempL[2]), np.max(tempL[2])

    minx = max(minx - 10, 0)
    maxx = min(maxx + 10, w)
    miny = max(miny - 10, 0)
    maxy = min(maxy + 10, h)
    minz = max(minz - 10, 0)
    maxz = min(maxz + 10, d)

    image = image[minx:maxx, miny:maxy, minz:maxz]
    gt = gt[minx:maxx, miny:maxy, minz:maxz]
    return image, gt

def CropAndPad(slice,x,y, nx, ny,value):

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.full((nx, ny),fill_value=value)
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def back_CropAndPad(x,y,nx,ny,prediction_cropped):
    # ASSEMBLE BACK THE SLICES
    slice_predictions = np.zeros((x, y))
    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2
    # insert cropped region into original image again
    if x > nx and y > ny:
        slice_predictions[x_s:x_s + nx, y_s:y_s + ny] = prediction_cropped
    else:
        if x <= nx and y > ny:
            slice_predictions[:, y_s:y_s + ny] = prediction_cropped[x_c:x_c + x, :]
        elif x > nx and y <= ny:
            slice_predictions[x_s:x_s + nx, :] = prediction_cropped[:, y_c:y_c + y]
        else:
            slice_predictions[:, :] = prediction_cropped[x_c:x_c + x, y_c:y_c + y]
    return slice_predictions


def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def read_info(path):
    f = open(path, 'r')
    Lines = f.readlines()
    info = defaultdict(list)
    for line in Lines:
        res = dict(map(lambda x: x.split(': '), line.rstrip("\n").split(",")))
        info.update(res)
    return info

def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in [1, 2, 3]:

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


def test_eval(net,image_list,save_result=False,do_postprocessing=True,log=True,crop=False):

    dice_list = []
    assd_list = []
    for i, patient in enumerate(image_list):
        for phase in ['ED','ES']:
            info = read_info(data_dir + patient + '/Info.cfg')
            ed = data_dir + patient + '/' + patient + '_frame' + str(
                info[phase]).zfill(2)
            img_dat = load_nii(ed + '.nii.gz')
            pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
            ed_img = img_dat[0]
            ed_gt = nib.load(ed + '_gt.nii.gz').get_fdata()
            if crop:
                ed_img, ed_gt = crop_ROI(ed_img, ed_gt)
            ed_img = (ed_img - np.mean(ed_img)) / np.std(ed_img)

            ####  resample ##########
            scale_vector = (pixel_size[0] / 1,
                            pixel_size[1] / 1)

            tmp_pred = np.zeros([ed_gt.shape[0], ed_gt.shape[1],ed_gt.shape[2]])

            for ii in range(int(np.floor(ed_img.shape[2] // batch_size))):
                vol = np.zeros([batch_size, 1, pathsize, pathsize])
                for idx, jj in enumerate(range(ii * batch_size, (ii + 1) * batch_size)):
                    image = ed_img[..., jj] # HxW
                    image = transform.rescale(image,
                                                   scale_vector,
                                                   order=1,
                                                   preserve_range=True,
                                                   anti_aliasing=True,
                                                   mode='constant')
                    x, y = image.shape
                    image = CropAndPad(image, x, y, pathsize, pathsize,np.min(image))
                    vol[idx, ...] = image[np.newaxis,...].copy()
                vol = torch.from_numpy(vol).float().cuda()
                outputs = net(x_2d=vol,
                              label=None,
                              use_prototype=False)
                pred = outputs["proto_seg"]  # b,c,h,w
                if isinstance(pred,tuple):
                    pred = pred[1]
                pred = F.softmax(pred, dim=1)
                pred = pred.detach().cpu().numpy()
                pred = pred[0]
                newpred = np.zeros((pred.shape[0],ed_gt.shape[0], ed_gt.shape[1]))
                for i in range(pred.shape[0]):
                    slicepred = back_CropAndPad(x, y, pathsize, pathsize, pred[i])
                    newpred[i]= transform.resize(slicepred,
                                            (ed_gt.shape[0], ed_gt.shape[1]),
                                            order=1,
                                            preserve_range=True,
                                            mode='constant')
                pred = np.argmax(newpred[np.newaxis,...],axis=1)
                for idx, jj in enumerate(range(ii * batch_size, (ii + 1) * batch_size)):
                    tmp_pred[..., jj] = pred[idx, ...].copy()

            if do_postprocessing:
                tmp_pred = keep_largest_connected_components(tmp_pred)
            for c in range(1, num_classes):

                pred_test_data_tr = tmp_pred.copy()
                pred_test_data_tr[pred_test_data_tr != c] = 0

                pred_gt_data_tr = ed_gt.copy()
                pred_gt_data_tr[pred_gt_data_tr != c] = 0

                dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
                assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))


            if save_result:

                nib.save(nib.Nifti1Image(tmp_pred.astype(np.float32), np.eye(4)), test_save_path + patient + phase + "_pred.nii.gz")
                nib.save(nib.Nifti1Image(ed_img[:].astype(np.float32), np.eye(4)), test_save_path + patient + phase + "_img.nii.gz")
                nib.save(nib.Nifti1Image(ed_gt[:].astype(np.float32), np.eye(4)), test_save_path + patient + phase + "_gt.nii.gz")


    dice_arr = 100 * np.reshape(dice_list, [-1, 3])
    dice_mean = np.mean(dice_arr, axis=0)
    dice_std = np.std(dice_arr, axis=0)

    if log:
        logging.info('Dice:')
        logging.info('RV :%.3f(%.3f)' % (dice_mean[0], dice_std[0]))
        logging.info('MYO :%.3f(%.3f)' % (dice_mean[1], dice_std[1]))
        logging.info('LV :%.3f(%.3f)' % (dice_mean[2], dice_std[2]))
        logging.info('dice mean :%.3f' % (np.mean(dice_mean)))

        assd_arr = np.reshape(assd_list, [-1, 3])

        assd_mean = np.mean(assd_arr, axis=0)
        assd_std = np.std(assd_arr, axis=0)

        logging.info('assd:')
        logging.info('RV :%.3f(%.3f)' % (assd_mean[0], assd_std[0]))
        logging.info('MYO :%.3f(%.3f)' % (assd_mean[1], assd_std[1]))
        logging.info('LV :%.3f(%.3f)' % (assd_mean[2], assd_std[2]))
        logging.info('assd mean :%.3f' % (np.mean(assd_mean)))

    return np.mean(dice_mean),dice_mean

if __name__ == '__main__':
    epoch_num = 15800
    net = UNetProto(
        inchannel=1,
        nclasses=num_classes,
        proj_dim=64,
        projection=None,
        l2_norm=True,
        proto_mom=0.999,
        sub_proto_size=2,
    ).cuda()
    logging.basicConfig(filename=test_save_path + 'iter_' + str(epoch_num) + "log.txt", level=logging.INFO,
                       format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    checkponit = torch.load(save_mode_path, map_location="cuda:0")
    net.load_state_dict(checkponit)
    print("init weight from {}".format(save_mode_path))
    net.eval()
    test_eval(net,image_list=image_list)

