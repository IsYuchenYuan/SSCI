import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from networks.UNet_3d import unet_3D
from dataloaders import utils
from utils import ramps, losses, util
from dataloaders.brainvessel import BrainVessel, RandomCrop, CenterCrop, RandomRotFlip, ToTensor,TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='dualstudent_wbce', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0,1', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--stabilization', type=float,  default=1, help='stabilization')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--label', type=str, default='unlabel',  help='whether use unlabel data')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../Ourmodel/" + args.exp + "/"
print()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 1
patch_size = (112, 112, 80)
# patch_size = (128, 128, 96)
device_ids=[0,1]

### supervised loss
wbce = losses.InstanceWeightedBCELoss()
wbce.cuda()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_current_stabilization_weight(epoch):

    return args.stabilization * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = unet_3D(n_classes=num_classes)
        model = net.cuda()
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model_1 = create_model()
    model_2 = create_model()

    db_train = BrainVessel(base_dir=train_data_path,
                       data=args.label,
                       fold='5',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labeled_idxs = list(range(34))
    unlabeled_idxs = list(range(34, 101))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)

    model_1.train()
    model_1.train()
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    if args.consistency_type == 'mse':
        if num_classes ==1:
            consistency_criterion = losses.sigmoid_mse_loss
            stabilization_criterion = losses.sigmoid_mse_loss
        else:
            consistency_criterion = losses.softmax_mse_loss
            stabilization_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
        stabilization_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            noise_inputs = volume_batch + noise
            # forward
            l_model_out = model_1(volume_batch)
            r_model_out = model_2(volume_batch)
            le_model_out = model_1(noise_inputs)
            re_model_out = model_2(noise_inputs)

            ## classification loss
            if num_classes ==  1:
                l_class_loss = wbce(l_model_out[:labeled_bs], label_batch[:labeled_bs].unsqueeze(dim=1).float())
                r_class_loss = wbce(r_model_out[:labeled_bs], label_batch[:labeled_bs].unsqueeze(dim=1).float())

                l_loss_seg_dice = losses.dice_loss(l_model_out[:labeled_bs, 0, :, :, :],
                                                   label_batch[:labeled_bs] == 1)
                r_loss_seg_dice = losses.dice_loss(r_model_out[:labeled_bs, 0, :, :, :],
                                                   label_batch[:labeled_bs] == 1)
            else:
                l_class_loss = F.cross_entropy(l_model_out[:labeled_bs], label_batch[:labeled_bs])
                r_class_loss = F.cross_entropy(r_model_out[:labeled_bs], label_batch[:labeled_bs])

                l_outputs_soft = F.softmax(l_model_out, dim=1)
                r_outputs_soft = F.softmax(r_model_out, dim=1)
                l_loss_seg_dice = losses.dice_loss(l_outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                r_loss_seg_dice = losses.dice_loss(r_outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

            ### consisteny loss within each student
            minibatch, _, w, h, d = l_model_out.size()
            consistency_weight = get_current_consistency_weight(iter_num//150)
            le_model_out_d = le_model_out.detach()
            re_model_out_d = re_model_out.detach()
            l_consistency_dist = consistency_criterion(l_model_out, le_model_out_d) / minibatch #(batch, 2, 112,112,80)
            r_consistency_dist = consistency_criterion(r_model_out, re_model_out_d) / minibatch #(batch, 2, 112,112,80)

            l_consistency_loss = consistency_weight * l_consistency_dist
            r_consistency_loss = consistency_weight * r_consistency_dist

            l_consistency_loss = torch.mean(l_consistency_loss)
            r_consistency_loss = torch.mean(r_consistency_loss)


            ##### stability loss between two models
            if iter_num > 1500:
                l_model_out_d = l_model_out.detach()
                r_model_out_d = r_model_out.detach()
                l_index,l_logit = l_model_out_d.cpu().numpy(),l_model_out_d.cpu().numpy()
                r_index,r_logit = r_model_out_d.cpu().numpy(),r_model_out_d.cpu().numpy()
                le_index = le_model_out_d.cpu().numpy()
                re_index = re_model_out_d.cpu().numpy()
                l_index[l_index>0.5] = 1
                l_index[l_index<=0.5] = 0
                le_index[le_index>0.5] = 1
                le_index[le_index<=0.5] = 0
                r_index[r_index > 0.5] = 1
                r_index[r_index <= 0.5] = 0
                re_index[re_index > 0.5] = 1
                re_index[re_index <= 0.5] = 0

                ###### construct the consist map
                consist_map_l = - torch.empty_like(l_model_out_d).cuda()
                consist_map_r = - torch.empty_like(l_model_out_d).cuda()

                #### find the indices of stable pixels
                # the 1st condition
                temp1 = l_index - le_index
                l_stable_1 = np.where(temp1==0)
                l_stable_1_list = util.findindex(l_stable_1)
                # the 2nd condition
                l_stable_2 = np.where(l_logit[:,0:1,...] > 0.7)
                l_stable_2_list = util.findindex(l_stable_2)
                l_stable_21 = np.where(l_logit[:, 0:1, ...] < 0.3)
                l_stable_21_list = util.findindex(l_stable_21)
                l_stable_2_list.extend(l_stable_21_list)
                # the indices of stable pixels in student 1
                l_pixel_indexs = np.intersect1d(l_stable_1_list, l_stable_2_list)

                # the 1st condition
                temp2 = r_index - re_index
                r_stable_1 = np.where(temp2 == 0)
                r_stable_1_list = util.findindex(r_stable_1)
                # the 2nd condition
                r_stable_2 = np.where(r_logit[:, 0:1, ...] > 0.7)
                r_stable_2_list = util.findindex(r_stable_2)
                r_stable_21 = np.where(r_logit[:, 0:1, ...] < 0.3)
                r_stable_21_list = util.findindex(r_stable_21)
                r_stable_2_list.extend(r_stable_21_list)
                # the indices of stable pixels in student 2
                r_pixel_indexs = np.intersect1d(r_stable_1_list, r_stable_2_list)

                intersect  = np.intersect1d(l_pixel_indexs,r_pixel_indexs)

                ###### only stable for one student
                l_first = np.setdiff1d(l_pixel_indexs, intersect, assume_unique=True)
                r_first = np.setdiff1d(r_pixel_indexs, intersect, assume_unique=True)
                consist_map_l[l_first] = l_model_out_d[l_first]
                consist_map_l[r_first] = r_model_out_d[r_first]
                consist_map_r[l_first] = l_model_out_d[l_first]
                consist_map_r[r_first] = r_model_out_d[r_first]

                ##### stable or unstable for both student, choose the other student


                consist_map = consist_map_l.detach().cpu().numpy()
                lr_random = np.where(consist_map==-1)
                consist_map_l[lr_random] = r_model_out_d[lr_random]
                consist_map_r[lr_random] = l_model_out_d[lr_random]


                stabilization_weight = get_current_consistency_weight(iter_num // 150)
                l_stabilization_loss = stabilization_weight * stabilization_criterion(l_model_out,
                                                                                      consist_map_l) / minibatch
                r_stabilization_loss = stabilization_weight * stabilization_criterion(r_model_out,
                                                                                     consist_map_r) / minibatch
                l_stabilization_loss = torch.mean(l_stabilization_loss)
                r_stabilization_loss = torch.mean(r_stabilization_loss )
            else:
                l_stabilization_loss = r_stabilization_loss = torch.tensor(0)

            # if iter_num > 2000:
            #     l_outputs_soft_d = l_outputs_soft.detach()
            #     r_outputs_soft_d = r_outputs_soft.detach()
            #     l_index = torch.argmax(l_outputs_soft_d, dim=1, keepdim=True)
            #     le_index = torch.argmax(F.softmax(le_model_out_d, dim=1), dim=1, keepdim=True)
            #     r_index = torch.argmax(r_outputs_soft_d, dim=1, keepdim=True)
            #     re_index = torch.argmax(F.softmax(re_model_out_d, dim=1), dim=1, keepdim=True)
            #
            #     l_cls_i = l_index.cpu().numpy()
            #     r_cls_i = r_index.cpu().numpy()
            #     le_cls_i = le_index.cpu().numpy()
            #     re_cls_i = re_index.cpu().numpy()
            #
            #     l_logit = l_outputs_soft_d.cpu().numpy()
            #     r_logit = r_outputs_soft_d.cpu().numpy()
            #     ###### construct the consist map
            #     l_model_out_d = l_model_out.detach()
            #     r_model_out_d = r_model_out.detach()
            #     consist_map_l = - torch.empty_like(l_model_out_d).cuda()
            #     consist_map_r = - torch.empty_like(l_model_out_d).cuda()
            #
            #     #### find the indices of stable pixels
            #     # the 1st condition
            #     temp1 = l_cls_i - le_cls_i
            #     l_stable_1 = np.where(temp1 == 0)
            #     l_stable_1_list = util.findindex(l_stable_1)
            #     # the 2nd condition
            #     l_stable_2 = np.where(l_logit[:, 1:2, ...] > 0.7)
            #     l_stable_2_list = util.findindex(l_stable_2)
            #     l_stable_21 = np.where(l_logit[:, 1:2, ...] < 0.3)
            #     l_stable_21_list = util.findindex(l_stable_21)
            #     l_stable_2_list.extend(l_stable_21_list)
            #     # the indices of stable pixels in student 1
            #     l_pixel_indexs = np.intersect1d(l_stable_1_list, l_stable_2_list)
            #
            #     # the 1st condition
            #     temp2 = r_cls_i - re_cls_i
            #     r_stable_1 = np.where(temp2 == 0)
            #     r_stable_1_list = util.findindex(r_stable_1)
            #     # the 2nd condition
            #     r_stable_2 = np.where(r_logit[:, 1:2, ...] > 0.7)
            #     r_stable_2_list = util.findindex(r_stable_2)
            #     r_stable_21 = np.where(r_logit[:, 1:2, ...] < 0.3)
            #     r_stable_21_list = util.findindex(r_stable_21)
            #     r_stable_2_list.extend(r_stable_21_list)
            #     # the indices of stable pixels in student 2
            #     r_pixel_indexs = np.intersect1d(r_stable_1_list, r_stable_2_list)
            #
            #     intersect = np.intersect1d(l_pixel_indexs, r_pixel_indexs)
            #
            #     ###### only stable for one student
            #     l_first = np.setdiff1d(l_pixel_indexs, intersect, assume_unique=True)
            #     r_first = np.setdiff1d(r_pixel_indexs, intersect, assume_unique=True)
            #     consist_map_l[l_first] = l_model_out_d[l_first]
            #     consist_map_l[r_first] = r_model_out_d[r_first]
            #     consist_map_r[l_first] = l_model_out_d[l_first]
            #     consist_map_r[r_first] = r_model_out_d[r_first]
            #
            #     ##### stable or unstable for both student, choose the other student
            #     consist_map = consist_map_l.detach().cpu().numpy()
            #     lr_random = np.where(consist_map == -1)
            #     consist_map_l[lr_random] = r_model_out_d[lr_random]
            #     consist_map_r[lr_random] = l_model_out_d[lr_random]
            #
            #     stabilization_weight = get_current_consistency_weight(iter_num // 150)
            #     l_stabilization_loss = stabilization_weight * stabilization_criterion(l_model_out,
            #                                                                           consist_map_l) / minibatch
            #     r_stabilization_loss = stabilization_weight * stabilization_criterion(r_model_out,
            #                                                                           consist_map_r) / minibatch
            #     l_stabilization_loss = torch.mean(l_stabilization_loss)
            #     r_stabilization_loss = torch.mean(r_stabilization_loss)
            # else:
            #     l_stabilization_loss = r_stabilization_loss = torch.tensor(0)

            # tar_l_class_logit = l_model_out.clone().detach()
            # in_r_cons_logit = r_model_out.detach()
            #
            # tar_r_class_logit = r_model_out.clone().detach()
            # in_l_cons_logit = l_model_out.detach()
            #
            # weight_pixels =  np.ones((minibatch,_,w, h, d))
            # for idx in range(minibatch):
            #     for k in range(d):
            #         for n in range(h):
            #             for m in range(w):
            #                 print("ddddd")
            #                 l_stable = False
            #                 if l_cls_i[idx,0,m,n,k] != le_cls_i[idx,0,m,n,k]:
            #                     tar_l_class_logit[idx,:,m,n,k] = in_r_cons_logit[idx,:,m,n,k]
            #                 else:
            #                     l_stable = True
            #
            #                 r_stable = False
            #                 if r_cls_i[idx,0,m,n,k] != re_cls_i[idx,0,m,n,k]:
            #                     tar_r_class_logit[idx,:,m,n,k] = in_l_cons_logit[idx,:,m,n,k]
            #                 else:
            #                     r_stable = True
            #
            #                 if (l_stable and r_stable )or not (l_stable or r_stable) :
            #                     if not (l_stable or r_stable):
            #                         weight_pixels[idx,:,m,n,k] = 0.5
            #                     # compare by consistency
            #                     l_sample_cons = consistency_criterion(l_model_out[idx:idx+1,:,m,n,k],
            #                                                           le_model_out[idx:idx+1,:,m,n,k])
            #                     r_sample_cons = consistency_criterion(r_model_out[idx:idx+1,:,m,n,k],
            #                                                           re_model_out[idx:idx+1,:,m,n,k])
            #
            #                     if l_sample_cons.data.cpu().numpy()[0].sum()< r_sample_cons.data.cpu().numpy()[0].sum():
            #                         # loss: l -> r
            #                         tar_r_class_logit[idx,:,m,n,k] = in_l_cons_logit[idx,:,m,n,k]
            #                     elif l_sample_cons.data.cpu().numpy()[0].sum() > r_sample_cons.data.cpu().numpy()[0].sum():
            #                         # loss: r -> l
            #                         tar_l_class_logit[idx,:,m,n,k] = in_r_cons_logit[idx,:,m,n,k]
            #
            # stabilization_weight = get_current_consistency_weight(iter_num // 150)
            # l_stabilization_loss = stabilization_weight * stabilization_criterion(l_model_out,
            #                                                                       tar_r_class_logit) / minibatch
            # r_stabilization_loss = stabilization_weight * stabilization_criterion(r_model_out,
            #                                                                       tar_l_class_logit) / minibatch
            # print(l_loss.size())
            # print(l_class_loss.size())
            # print(l_loss_seg_dice.size())
            # print(l_consistency_loss.size())
            # print(l_stabilization_loss.size())
            # l_loss = 0.5*(l_class_loss+l_loss_seg_dice) + l_consistency_loss + l_stabilization_loss
            # r_loss = 0.5*(r_class_loss+r_loss_seg_dice) + r_consistency_loss + r_stabilization_loss

            l_loss = 0.5 * (l_class_loss + l_loss_seg_dice) + l_consistency_loss + l_stabilization_loss
            r_loss = 0.5 * (r_class_loss + r_loss_seg_dice) + r_consistency_loss + r_stabilization_loss

            # update model
            optimizer_1.zero_grad()
            l_loss.backward()
            optimizer_1.step()

            optimizer_2.zero_grad()
            r_loss.backward()
            optimizer_2.step()


            iter_num = iter_num + 1
            # writer.add_scalar('lr', lr_, iter_num)
            # writer.add_scalar('loss/loss', loss, iter_num)
            # writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            # writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            # writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            # writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            # writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('iteration %d : l_loss : %f l_cons_loss: %f, l_stable_loss: %f, r_loss : %f r_cons_loss: %f, r_stable_loss: %f' %
                         (iter_num, l_loss.item(), l_consistency_loss.item(),l_stabilization_loss.item(),
                          r_loss.item(), r_consistency_loss.item(), r_stabilization_loss.item()))
            # if iter_num % 50 == 0:
                # image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                # grid_image = make_grid(image, 5, normalize=True)
                # writer.add_image('train/Image', grid_image, iter_num)
                #
                # # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                # image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                # image = utils.decode_seg_map_sequence(image)
                # grid_image = make_grid(image, 5, normalize=False)
                # writer.add_image('train/Predicted_label', grid_image, iter_num)
                #
                # image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                # grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                # writer.add_image('train/Groundtruth_label', grid_image, iter_num)
                #
                # image = uncertainty[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                # grid_image = make_grid(image, 5, normalize=True)
                # writer.add_image('train/uncertainty', grid_image, iter_num)
                #
                # mask2 = (uncertainty > threshold).float()
                # image = mask2[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                # grid_image = make_grid(image, 5, normalize=True)
                # writer.add_image('train/mask', grid_image, iter_num)
                # #####
                # image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                # grid_image = make_grid(image, 5, normalize=True)
                # writer.add_image('unlabel/Image', grid_image, iter_num)
                #
                # # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                # image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                # image = utils.decode_seg_map_sequence(image)
                # grid_image = make_grid(image, 5, normalize=False)
                # writer.add_image('unlabel/Predicted_label', grid_image, iter_num)
                #
                # image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                # grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                # writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer_1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_2.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                state = {'iter': iter_num,  # 保存的当前轮数
                         'l_state_dict': model_1.module.state_dict(),  # 训练好的参数
                         'r_state_dict': model_2.module.state_dict(),
                         'optimizer_l': optimizer_1.state_dict(),
                         'optimizer_r': optimizer_2.state_dict(),# 优化器参数,为了后续的resume
                         }
                torch.save(state, save_mode_path)
                logging.info("save Ourmodel to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    state = {'iter': iter_num,  # 保存的当前轮数
             'l_state_dict': model_1.module.state_dict(),  # 训练好的参数
             'r_state_dict': model_2.module.state_dict(),
             'optimizer_l': optimizer_1.state_dict(),
             'optimizer_r': optimizer_2.state_dict(),  # 优化器参数,为了后续的resume
             }
    torch.save(state, save_mode_path)
    logging.info("save Ourmodel to {}".format(save_mode_path))
    writer.close()
