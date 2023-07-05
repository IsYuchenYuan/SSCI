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
# from networks.UNet_3d import unet_3D
# from networks.HUNet import dense_rnn_net
# from networks.HDenseUNet import dense_rnn_net
# from networks.DenseUNet3d import dense_rnn_net
from networks.NNUNet import initialize_network
from utils.losses import dice_loss,InstanceWeightedBCELoss,FocalLoss,compute_sdf1_1, boundary_loss
from utils.util import aggreate,aggreate_pred
from utils import ramps
from dataloaders.brainvessel import BrainVessel, RandomCrop, CenterCrop, RandomRotFlip, ToTensor
from PIL import Image
import cleanlab

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='NNUNet_CL_smooth', help='model_name')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
parser.add_argument('--multi_gpu', type=int, default=1,  help='whether use multipule gpu')
parser.add_argument('--patchsize', type=list, default=[128,128,128],  help='whether use unlabel data')
# CL
parser.add_argument('--CL_type', type=str,
                    default='both', help='CL implement type')
parser.add_argument('--weak_weight', type=float,
                    default=5.0, help='weak_weight')
parser.add_argument('--refine_type', type=str,
                    default="smooth", help='refine types')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../Ourmodel/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
# criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor((0.3,0.7), device='cuda'), reduction='mean')
criterion = torch.nn.CrossEntropyLoss()
focal_loss = FocalLoss()
criterion.cuda()
focal_loss.cuda()
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

########################distribution##############
# if args.multi_gpu:
#     dist.init_process_group(backend='nccl')
#     local_rank = torch.distributed.get_rank()
#     torch.cuda.set_device(local_rank)
#     device = torch.device("cuda", local_rank)
###############loss##############
# wbce = InstanceWeightedBCELoss()
# wbce.cuda()

# patch_size = (224, 224, 12)
# (112, 112, 80)
# patch_size = (128,128,96)
patch_size = (128,128,128)
num_classes = 2
device_ids=[0,1,2,3]

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def soft_cross_entropy(predicted, target):
    #print(predicted.type(), target.type())
    return -(target * torch.log(predicted)).sum(dim=1).mean()


def validation(net, testloader):
    net.eval()
    val_dice_loss = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for i, sampled_batch in enumerate(testloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            image = volume_batch.cuda()
            label = label_batch.cuda()

            Y = net(image)
            Y_softmax = F.softmax(Y, dim=1)

            val_dice_loss += dice_loss(Y_softmax[:, 1, :, :, :], label==1).item()

    val_dice_loss /= (i + 1)
    accuracy /= (i + 1)
    return val_dice_loss

if __name__ == "__main__":
    # make logger file
    bestloss = np.inf
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        # net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        # net = unet_3D(n_classes=num_classes)
        # net = dense_rnn_net()
        net = initialize_network(threeD=True)
        model = net.cuda()
        # model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    db_train = BrainVessel(base_dir=train_data_path,
                           data='label',
                           split='train',
                           fold=5,
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    db_test = BrainVessel(base_dir=train_data_path,
                          data='label',
                          split='test',
                          fold=5,
                          transform=transforms.Compose([
                              RandomCrop(patch_size),
                              ToTensor(),
                          ]))


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                 worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=batch_size, num_workers=4, pin_memory=True,
                            worker_init_fn=worker_init_fn)
    x_criterion = soft_cross_entropy

    model.train()
    ema_model.train()
    # optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch = volume_batch.permute(0,1,4,2,3)
            label_batch = label_batch.permute(0,3,1,2)
            outputs3d = model(volume_batch)
            loss_seg_3d = criterion(outputs3d, label_batch)
            outputs_soft_3d = F.softmax(outputs3d, dim=1)
            loss_seg_dice_3d = dice_loss(outputs_soft_3d[:, 1, :, :, :], label_batch == 1)
            loss_focal = focal_loss(outputs3d, label_batch.long())

            # boundary loss
            with torch.no_grad():
                gt_sdf_npy = compute_sdf1_1(label_batch.cpu().numpy(), outputs_soft_3d.shape)
                gt_sdf = torch.from_numpy(gt_sdf_npy).float().cuda(outputs_soft_3d.device.index)
            loss_bd = boundary_loss(outputs_soft_3d, gt_sdf)

            supervised_loss = 0.5 * (loss_seg_3d + loss_seg_dice_3d) + loss_focal + 0.5 * loss_bd

            # noise detection
            with torch.no_grad():
                ema_output_no_noise = ema_model(volume_batch)
                ema_output_soft_no_noise = torch.softmax(ema_output_no_noise, dim=1)
            # 1: tensor to npy
            masks_np = label_batch.cpu().detach().numpy()
            ema_output_soft_np = ema_output_soft_no_noise.cpu().detach().numpy()

            # 2: identify the noise map
            ema_output_soft_np_accumulated_0 = np.swapaxes(ema_output_soft_np, 1, 2)
            ema_output_soft_np_accumulated_1 = np.swapaxes(ema_output_soft_np_accumulated_0, 2, 3)
            ema_output_soft_np_accumulated_2 = np.swapaxes(ema_output_soft_np_accumulated_1, 3, 4)
            ema_output_soft_np_accumulated_3 = ema_output_soft_np_accumulated_2.reshape(-1, num_classes)
            ema_output_soft_np_accumulated = np.ascontiguousarray(ema_output_soft_np_accumulated_3)
            masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)
            assert masks_np_accumulated.shape[0] == ema_output_soft_np_accumulated.shape[0]

            CL_type = args.CL_type


            # For cleanlab v1.0
            # if CL_type in ['both']:
            # noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, ema_output_soft_np_accumulated, prune_method='both', n_jobs=1)
            # elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
            # noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, ema_output_soft_np_accumulated, prune_method=CL_type, n_jobs=1)
            # For cleanlab v2.0
            if CL_type in ['both']:
                noise = cleanlab.filter.find_label_issues(masks_np_accumulated, ema_output_soft_np_accumulated,
                                                          filter_by='both', n_jobs=1)
            elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                noise = cleanlab.filter.find_label_issues(masks_np_accumulated, ema_output_soft_np_accumulated,
                                                          filter_by=CL_type, n_jobs=1)
            confident_maps_np = noise.reshape(-1, patch_size[0], patch_size[1], patch_size[2]).astype(np.uint8)

            # Correct the LQ label for our focused binary task
            correct_type = args.refine_type
            if correct_type == 'smooth':
                smooth_arg = 0.8
                corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
                print('Smoothly correct the noisy label')
                corrected_masks_np = corrected_masks_np[:,np.newaxis,...].astype(np.float32)
                corrected_masks_np = np.concatenate((1 - corrected_masks_np, corrected_masks_np), axis=1)
            else:
                corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np)
                print('Hard correct the noisy label')

            noisy_label_batch = torch.from_numpy(corrected_masks_np).cuda(outputs_soft_3d.device.index)

            # noisy supervised loss
            if correct_type == 'smooth':
                loss_ce_weak = x_criterion(outputs_soft_3d, noisy_label_batch)
                loss_dice_weak = dice_loss(outputs_soft_3d[:, 1, :, :, :], noisy_label_batch[:, 1, :, :, :])
            else:
                loss_ce_weak = criterion(outputs3d, noisy_label_batch.long())
                loss_dice_weak = dice_loss(outputs_soft_3d[:, 1, :, :, :], noisy_label_batch.long()==1)

            weak_supervised_loss = 0.5 * (loss_dice_weak + loss_ce_weak)


            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = supervised_loss + consistency_weight * (args.weak_weight * weak_supervised_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg_3d', loss_seg_3d, iter_num)
            writer.add_scalar('loss/loss_seg_3d', loss_seg_dice_3d, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f ce : %f dice : %f '
                         'focal : %f   boundary : %f  week_sup : %f'
                         % (iter_num, loss.item(), loss_seg_3d.item(), loss_seg_dice_3d.item(),
                            loss_focal.item(), loss_bd.item(), weak_supervised_loss.item()))

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num >2000 and iter_num % 50 == 0:
                testing_dice_loss = validation(net=model, testloader=testloader)
                logging.info('iter %d : testing_dice_loss : %f ' %
                             (iter_num, testing_dice_loss))
                if bestloss > testing_dice_loss:
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save(model.module.state_dict(), save_mode_path)
                    logging.info("save Ourmodel to {}".format(save_mode_path))
                    bestloss = testing_dice_loss

            if iter_num > max_iterations:
                break
            time1 = time.time()

        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations + 1) + '.pth')
    if args.multi_gpu:
        torch.save(model.module.state_dict(), save_mode_path)
    else:
        torch.save(model.state_dict(), save_mode_path)
    logging.info("save Ourmodel to {}".format(save_mode_path))
    writer.close()
