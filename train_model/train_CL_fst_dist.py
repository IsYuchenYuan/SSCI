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
from PIL import Image
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from networks.NNUNet import initialize_network
from networks.vnet import VNet
from networks.UNet_3d import unet_3D
# from networks.HUNet import dense_rnn_net
from dataloaders import utils
import cleanlab
from utils import ramps, losses
from dataloaders.brainvessel import BrainVessel, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='NNUNet_CL_fst_largerHW', help='model_name')
parser.add_argument('--max_iterations', type=int, default=8000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
parser.add_argument('--patchsize', type=list, default=[256, 256, 32],  help='whether use unlabel data')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.5, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
# CL
parser.add_argument('--CL_type', type=str,
                    default='both', help='CL implement type')
parser.add_argument('--weak_weight', type=float,
                    default=5.0, help='weak_weight')
parser.add_argument('--refine_type', type=str,
                    default="smooth", help='refine types')
parser.add_argument("--local_rank", type=int, help="")
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../Ourmodel/" + args.exp + "/"
batch_size = 1
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

########################distribution##############

dist.init_process_group(backend='nccl')
# local_rank = torch.distributed.get_rank()
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


num_classes = 2
# patch_size = (128, 128, 128)
patch_size = (256, 256, 32)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


@torch.no_grad()
def record_model(model, tmp_model):
    # store params in model to tmp_model
    mp = list(model.parameters())
    mcp = list(tmp_model.parameters())
    for i in range(0, len(mp)):
        if not mcp[i].data.shape:  # scalar tensor
            mcp[i].data = mp[i].data.clone()
        else:
            mcp[i].data[:] = mp[i].data[:].clone()


@torch.no_grad()
def restore_model(model, tmp_model):
    # release tmp_model params to model
    mp = list(tmp_model.parameters())
    mcp = list(model.parameters())
    for i in range(0, len(mp)):
        if not mcp[i].data.shape:  # scalar tensor
            mcp[i].data = mp[i].data.clone()
        else:
            mcp[i].data[:] = mp[i].data[:].clone()


def soft_cross_entropy(predicted, target):
    # print(predicted.type(), target.type())
    return -(target * torch.log(predicted)).sum(dim=1).mean()


def validation(net, testloader):
    net.eval()
    val_dice_loss = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for i, sampled_batch in enumerate(testloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch = volume_batch.permute(0, 1, 4, 2, 3)
            label_batch = label_batch.permute(0, 3, 1, 2)
            image = volume_batch.to(device)
            label = label_batch.to(device)

            Y = net(image)
            Y_softmax = F.softmax(Y, dim=1)

            val_dice_loss += losses.dice_loss(Y_softmax[:, 1, :, :, :], label == 1).item()

    val_dice_loss /= (i + 1)
    accuracy /= (i + 1)
    return val_dice_loss


if __name__ == "__main__":
    ## make logger file
    bestloss = np.inf

    if dist.get_rank() == 0:
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
        net = net.to(device)
        model = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model()
    ema_model = create_model(ema=True)
    tmp_model = create_model(ema=True)

    db_train_l = BrainVessel(base_dir=train_data_path,
                           data='label',
                           fold='5',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))

    db_train_ul = BrainVessel(base_dir=train_data_path,
                           data='un_only',
                           fold='5',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))

    db_test = BrainVessel(base_dir=train_data_path,
                          data='unlabel',
                          split='test',
                          fold=5,
                          transform=transforms.Compose([
                              RandomCrop(patch_size),
                              ToTensor(),
                          ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    l_sampler = DistributedSampler(db_train_l)
    ul_sampler = DistributedSampler(db_train_ul)

    l_trainloader = DataLoader(db_train_l, batch_size=batch_size, num_workers=4, pin_memory=True,
                               sampler=l_sampler, worker_init_fn=worker_init_fn)

    ul_trainloader = DataLoader(db_train_ul, batch_size=batch_size, num_workers=4, pin_memory=True,
                               sampler=ul_sampler, worker_init_fn=worker_init_fn)

    testloader = DataLoader(db_test, batch_size=batch_size, num_workers=4, pin_memory=True,
                            worker_init_fn=worker_init_fn)
    model.train()
    ema_model.train()
    x_criterion = soft_cross_entropy
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    if args.consistency_type == 'mse':
        if num_classes == 1:
            consistency_criterion = losses.sigmoid_mse_loss
        else:
            consistency_criterion = losses.softmax_mse_loss
            # feature_consistency_criterion = losses.mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    if dist.get_rank() == 0:
        writer = SummaryWriter(snapshot_path + '/log')
        logging.info("{} itertations per epoch".format(len(l_trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(l_trainloader) + 1
    lr_ = base_lr


    def loss_back():
        outputs = model(volume_batch)
        outputs_soft = F.softmax(outputs, dim=1)

        with torch.no_grad():
            ema_output = ema_model(ema_inputs)
            # ema_output_soft = torch.softmax(ema_output, dim=1)

        # supervised loss

        loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
        loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
        supervised_loss = 0.5 * (loss_seg + loss_seg_dice)

        # consistency loss

        consistency_weight = get_current_consistency_weight(iter_num // (max_iterations // args.consistency_rampup))
        consistency_loss = torch.mean(consistency_criterion(outputs, ema_output))  # (batch, 2, 112,112,80)

        # CL loss
        noisy_label_batch = label_batch
        with torch.no_grad():
            ema_output_no_noise = ema_model(volume_batch)
            ema_output_soft_no_noise = torch.softmax(ema_output_no_noise, dim=1)
        # 1: tensor to npy
        masks_np = noisy_label_batch.cpu().detach().numpy()
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
        try:
            if CL_type in ['both']:
                noise = cleanlab.filter.find_label_issues(masks_np_accumulated, ema_output_soft_np_accumulated,
                                                          filter_by='both', n_jobs=1)
            elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                noise = cleanlab.filter.find_label_issues(masks_np_accumulated, ema_output_soft_np_accumulated,
                                                          filter_by=CL_type, n_jobs=1)
            confident_maps_np = noise.reshape(-1, patch_size[2], patch_size[0], patch_size[1]).astype(np.uint8)
        except:
            confident_maps_np = np.zeros_like(masks_np)

        # Correct the LQ label for our focused binary task
        correct_type = args.refine_type
        if correct_type == 'smooth':
            smooth_arg = 0.8
            corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
            # print('Smoothly correct the noisy label')
            corrected_masks_np = corrected_masks_np[:, np.newaxis, ...].astype(np.float32)
            corrected_masks_np = np.concatenate((1 - corrected_masks_np, corrected_masks_np), axis=1)
        else:
            corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np)
            # print('Hard correct the noisy label')

        noisy_label_batch = torch.from_numpy(corrected_masks_np).cuda(outputs_soft.device.index)

        # noisy supervised loss
        if correct_type == 'smooth':
            loss_ce_weak = x_criterion(outputs_soft, noisy_label_batch)
            loss_dice_weak = losses.dice_loss(outputs_soft[:, 1, :, :, :], noisy_label_batch[:, 1, :, :, :])
        else:
            loss_ce_weak = F.cross_entropy(outputs, noisy_label_batch.long())
            loss_dice_weak = losses.dice_loss(outputs_soft[:, 1, :, :, :], noisy_label_batch.long() == 1)

        weak_supervised_loss = 0.5 * (loss_dice_weak + loss_ce_weak)

        # total loss

        loss = supervised_loss + consistency_weight * (
                consistency_loss + args.weak_weight * weak_supervised_loss)

        return loss, loss_seg, loss_seg_dice, consistency_loss, weak_supervised_loss, consistency_weight


    for epoch_num in tqdm(range(max_epoch), ncols=70):
        l_sampler.set_epoch(epoch_num)
        ul_sampler.set_epoch(epoch_num)

        time1 = time.time()
        for i_batch, (sampled_batch_l,sampled_batch_ul) in enumerate(zip(l_trainloader,ul_trainloader)):
            time2 = time.time()
            volume_batch_l, label_batch_l = sampled_batch_l['image'], sampled_batch_l['label']
            volume_batch_ul, label_batch_ul = sampled_batch_ul['image'], sampled_batch_ul['label']

            volume_batch = torch.cat((volume_batch_l, volume_batch_ul), dim=0)
            label_batch = torch.cat((label_batch_l, label_batch_ul), dim=0)
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            volume_batch = volume_batch.permute(0, 1, 4, 2, 3)
            label_batch = label_batch.permute(0, 3, 1, 2)
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise

            # future teacher
            record_model(model, tmp_model)
            for _ in range(3):
                update_ema_variables(model, ema_model, args.ema_decay, iter_num)
                optimizer.zero_grad()
                loss, _, _, _, _, _ = loss_back()
                loss.backward()
                optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            restore_model(model, tmp_model)

            optimizer.zero_grad()
            loss, loss_seg, loss_seg_dice, consistency_loss, weak_supervised_loss, consistency_weight = loss_back()
            loss.backward()
            optimizer.step()
            # update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            if dist.get_rank() == 0:
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
                writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
                writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
                writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('train/weak_supervised_loss', weak_supervised_loss, iter_num)

                logging.info('iteration %d : loss: %f ce: %f dice: %f con: %f weak: %f' %
                             (iter_num, loss.item(), loss_seg.item(), loss_seg_dice.item(),
                              consistency_loss.item(), weak_supervised_loss.item()))

            ## change lr
            if iter_num % 3000 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 3000)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if dist.get_rank() == 0:
                if iter_num >= 1000 and iter_num % 100 == 0:
                    testing_dice_loss = validation(net=model, testloader=testloader)
                    logging.info('iter %d : testing_dice_loss : %f ' %
                                 (iter_num, testing_dice_loss))
                    if bestloss > testing_dice_loss:
                        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                        torch.save(model.module.state_dict(), save_mode_path)
                        logging.info("save Ourmodel to {}".format(save_mode_path))
                        bestloss = testing_dice_loss

                if iter_num % 500 == 0:
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save(model.module.state_dict(), save_mode_path)
                    logging.info("save Ourmodel to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break

    if dist.get_rank() == 0:
        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
        torch.save(model.module.state_dict(), save_mode_path)
        logging.info("save Ourmodel to {}".format(save_mode_path))
        writer.close()
