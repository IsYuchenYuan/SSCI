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
from networks.NNUNet import initialize_network
from utils import ramps, losses
from dataloaders.brainvessel import BrainVessel, RandomCrop, CenterCrop, RandomRotFlip, ToTensor,TwoStreamBatchSampler,Normalization
from utils.util import aggreate,aggreate_pred
from utils.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from utils.lib_tree_filter.modules.tree_filter import TreeFilter2D
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='NNUNet_semi1', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0,1,2,3', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../Ourmodel/" + args.exp + "/"
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
########################distribution##############

dist.init_process_group(backend='nccl')
# local_rank = torch.distributed.get_rank()
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (128,128,128)
mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def converToSlice(input):
    if len(input.shape)==5:
        B, C, H, W, D = input.size()
        input2d = input[:, :, :, :, 0:1]
        for i in range(1, D):
            input2dtmp = input[:, :, :, :, i:i + 1]
            input2d = torch.cat((input2d, input2dtmp), dim=0)
        input2d = input2d[:, 0, ...]  # squeeze the last channel C (BxD,H,W,1)
        input2d = input2d.permute(0, 3, 1, 2)  # BxD,1,H,W
    else:
        B, H, W, D = input.size()
        input2d = input[:, :, :, 0]
        for i in range(1, D):
            input2dtmp = input[:, :, :, i]
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

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):

        net = initialize_network(threeD=True)
        model = net.cuda()
        model = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    db_train_l = BrainVessel(base_dir=train_data_path,
                           data='label',
                           fold=5,
                           transform=transforms.Compose([
                               Normalization(),
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))

    db_train_ul = BrainVessel(base_dir=train_data_path,
                           data='un_only',
                           fold=5,
                           transform=transforms.Compose([
                               Normalization(),
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))

    db_test = BrainVessel(base_dir=train_data_path,
                          data='unlabel',
                          split='test',
                          fold=5,
                          transform=transforms.Compose([
                              Normalization(),
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
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    if args.consistency_type == 'mse':
        if num_classes==1:
            consistency_criterion = losses.sigmoid_mse_loss
        else:
            consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    if dist.get_rank() == 0:
        writer = SummaryWriter(snapshot_path + '/log')
        logging.info("{} itertations per epoch".format(len(l_trainloader)))


    iter_num = 0
    max_epoch = max_iterations//len(l_trainloader)+1
    lr_ = base_lr
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):

        for i_batch, (sampled_batch_l,sampled_batch_ul) in enumerate(zip(l_trainloader,ul_trainloader)):

            volume_batch_l, label_batch_l = sampled_batch_l['image'], sampled_batch_l['label']
            volume_batch_ul, label_batch_ul = sampled_batch_ul['image'], sampled_batch_ul['label']

            volume_batch = torch.cat((volume_batch_l, volume_batch_ul), dim=0)
            label_batch = torch.cat((label_batch_l, label_batch_ul), dim=0)
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            volume_batch = volume_batch.permute(0, 1, 4, 2, 3)
            label_batch = label_batch.permute(0, 3, 1, 2)

            outputs, _ = model(volume_batch)
            ema_output,features = ema_model(volume_batch)

            # consistency loss
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_loss = consistency_criterion(outputs, ema_output)  # (batch, 2, 112,112,80)
            consistency_loss = consistency_loss.mean()

            # supervised loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft = F.softmax(outputs[:labeled_bs], dim=1)
            prob_l = outputs_soft[:, 1, ...]
            loss_seg_dice = losses.dice_loss(prob_l, label_batch[:labeled_bs] == 1)

            outputs_aggreate = aggreate_pred(prob_l)
            label_aggreate = aggreate(label_batch[:labeled_bs])
            loss_seg_dice_3d_aggre = losses.dice_loss(outputs_aggreate, label_aggreate == 1)

            loss_sup = 0.5 * (loss_seg + loss_seg_dice) + 0.25 * loss_seg_dice_3d_aggre

            # weak supervised loss
            volume_batch_u_2d = converToSlice(volume_batch_ul)
            low_feats = volume_batch_u_2d
            features = features[labeled_bs:,].permute(0,1,3,4,2) # b,16,h,w,d
            features_2d = converToSlice(features)
            high_feats = features_2d

            prob = F.softmax(ema_output[labeled_bs:,], dim=1)

            tree = mst_layers(low_feats)
            AS = tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

            # high-level MST
            if high_feats is not None:
                tree = mst_layers(high_feats)
                AS = tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree,
                                        low_tree=False)  # [b, n, h, w]

            refined_label_soft = AS
            refined_label_soft = converToVolumn(refined_label_soft)
            refined_label_soft = refined_label_soft.permute(0,1,4,2,3)

            refined_label = torch.argmax(AS, dim=1)
            refined_label = converToVolumn(refined_label)
            refined_label = refined_label.permute(0, 1, 4, 2, 3)

            # compute weak loss

            loss_seg = F.cross_entropy(outputs[labeled_bs:,], refined_label)
            prob_u = prob[:, 1, ...]
            loss_seg_dice = losses.dice_loss(prob_u, refined_label == 1)

            outputs_aggreate = aggreate_pred(prob_u)
            label_aggreate = aggreate(refined_label)
            loss_seg_dice_3d_aggre = losses.dice_loss(outputs_aggreate, label_aggreate == 1)

            loss_weak = 0.5 * (loss_seg + loss_seg_dice) + 0.25 * loss_seg_dice_3d_aggre

            loss = loss_sup + consistency_weight * (consistency_loss + loss_weak)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                         (iter_num, loss.item(), consistency_dist.item(), consistency_weight))

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.module.state_dict(), save_mode_path)
                logging.info("save Ourmodel to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.module.state_dict(), save_mode_path)
    logging.info("save Ourmodel to {}".format(save_mode_path))
    writer.close()
