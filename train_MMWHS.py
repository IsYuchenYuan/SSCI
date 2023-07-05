import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import time
import random
import numpy as np
from einops import rearrange
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from networks.unet_proto_hybrid import UNetProto
from utils import ramps,losses
from dataloaders.cardiac import Cardiac, RandomCrop, CenterCrop, RandomRotFlip, ToTensor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.util import visulize, converToSlice
from utils.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from utils.lib_tree_filter.modules.tree_filter import TreeFilter2D
import torch.backends.cudnn as cudnn
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='', help='data path')
parser.add_argument('--exp', type=str, default='', help='model name')
parser.add_argument('--percentage', type=float, default='', help='labeled percentage [0.1,0.2,0.4]')

parser.add_argument('--max_iterations', type=int, default=10000, help='maximum epoch number to train the whole framework')
parser.add_argument('--num_classes', type=int, default=8, help='the output classes of model')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
parser.add_argument('--patchsize', type=list, default=[256, 256],  help='size of input patch')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float, default='0.5', help='loss weight of unlabeled data (you can change to suit the dataset)')
parser.add_argument('--consistency_rampup', type=float, default='', help='consistency_rampup')

parser.add_argument("--sub_proto_size", type=int, default= 2, help="whether to use subcluster")
parser.add_argument("--pretrainIter", type=int, default=3000, help="maximum iteration to train both classifiers by using labeled data only")
parser.add_argument("--linearIter", type=int, default=1000, help="maximum iteration to train the LC")
parser.add_argument("--dice_w", type=float, default=0.5, help="the weight of dice loss (you can change to suit the dataset)")
parser.add_argument("--ce_w", type=float, default=0.5, help="the weight of ce loss (you can change to suit the dataset)")
parser.add_argument("--cls_w", type=float, default=0.25, help="the weight of linear-based classifier loss (you can change to suit the dataset)")
parser.add_argument("--proto_w", type=float, default=0.5, help="the weight of proto-based classifier loss (you can change to suit the dataset)")
parser.add_argument("--vol_w", type=float, default=1, help="the weight of 3d volumn loss (you can change to suit the dataset)")
parser.add_argument('--proto_rampup', type=float, default=40.0, help='proto_rampup')
parser.add_argument("--losstype", type=str, default="ce_dice", help="the type of ce and dice loss")

parser.add_argument("--local_rank", type=int, help="")
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../Ourmodel/" + args.exp + "/"
batch_size = 1
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
num_classes = args.num_classes
patch_size = args.patchsize

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

########################distribution##############
dist.init_process_group(backend='nccl')
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

###############loss##########
criterion = torch.nn.CrossEntropyLoss()
proto_loss = losses.PixelPrototypeCELoss()
HD_loss = losses.HausdorffDTLoss()
criterion.to(device)
proto_loss.to(device)
HD_loss.to(device)
dice_loss = losses.DiceLoss(num_classes)
mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
tree_filter_layers = TreeFilter2D(groups=1, sigma=0.05)


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def maybe_update_lr(epoch,optimizer,max_num_epochs,initial_lr):
    """
    if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

    (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
    herefore we need to do +1 here)

    :param epoch:
    :return:
    """
    if epoch is None:
        ep = epoch + 1
    else:
        ep = epoch
    optimizer.param_groups[0]['lr'] = poly_lr(ep, max_num_epochs, initial_lr, 0.9)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def validation(net, testloader):
    net.eval()
    val_dice_loss = 0.0
    with torch.no_grad():
        for i, sampled_batch in enumerate(testloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch = volume_batch.to(device)
            label_batch = label_batch.to(device)

            input2d = converToSlice(volume_batch)

            outputs = net(x_2d=input2d,
                          x_3d=volume_batch,
                          label=None,
                          use_prototype=False)
            cls_seg_3d = outputs["cls_seg_3d"]  # B,2,h,w,d
            loss_cls_seg_3d = criterion(cls_seg_3d, label_batch)
            outputs_soft_3d = F.softmax(cls_seg_3d, dim=1)
            loss_seg_dice_3d = dice_loss(outputs_soft_3d, label_batch)
            loss_cls_3d = 0.5 * (loss_cls_seg_3d + loss_seg_dice_3d)

            loss_cls = loss_cls_3d
            val_dice_loss = loss_cls.item()

    val_dice_loss /= (i + 1)
    return val_dice_loss


if __name__ == "__main__":
    ## make logger file
    bestloss = np.inf
    bestIter = 0
    if dist.get_rank() == 0:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

        logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))


    checkpoint = ""

    def create_model(pretrain=False,ema=False):
        net = UNetProto(
            inchannel=1,
            nclasses=num_classes,
            proj_dim=64,
            projection="v1",
            l2_norm=True,
            proto_mom=0.999,
            sub_proto_size=args.sub_proto_size,
            proto=None,
        )
        net = net.to(device)
        if pretrain:
            model_dict = torch.load(checkpoint)
            net.load_state_dict(model_dict)
        model = DDP(net, device_ids=[local_rank], output_device=local_rank, broadcast_buffers= False, find_unused_parameters=True)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    s_model = create_model(pretrain=False,ema=False)
    t_model = create_model(pretrain=False,ema=True)


    db_train_l = Cardiac(base_dir=train_data_path,
                       split='train_l',
                       percentage=0.2,
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    db_train_ul = Cardiac(base_dir=train_data_path,
                         split='train_ul',
                         percentage=0.2,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(patch_size),
                             ToTensor(),
                         ]))
    db_test = Cardiac(base_dir=train_data_path,
                      split='test',
                      percentage=0.2,
                      transform=transforms.Compose([
                          CenterCrop(patch_size),
                          ToTensor()
                      ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    l_sampler = DistributedSampler(db_train_l)
    ul_sampler = DistributedSampler(db_train_ul)

    l_trainloader = DataLoader(db_train_l, batch_size=batch_size, num_workers=0, pin_memory=True,
                               sampler=l_sampler, worker_init_fn=worker_init_fn)

    ul_trainloader = DataLoader(db_train_ul, batch_size=batch_size, num_workers=0, pin_memory=True,
                               sampler=ul_sampler, worker_init_fn=worker_init_fn)

    testloader = DataLoader(db_test, batch_size=batch_size, num_workers=0, pin_memory=True,
                            worker_init_fn=worker_init_fn)
    s_model.train()
    t_model.train()
    optimizer = optim.SGD(s_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)



    if args.consistency_type == 'mse':
        if num_classes == 1:
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
    max_epoch = max_iterations // len(l_trainloader) + 1
    lr_ = base_lr
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        logging.info("\n")
        l_sampler.set_epoch(epoch_num)
        ul_sampler.set_epoch(epoch_num)
        # maybe_update_lr(epoch_num,optimizer,max_epoch,base_lr)
        # logging.info("learning rate: %.6f" % optimizer.param_groups[0]['lr'])

        for i_batch, (sampled_batch_l,sampled_batch_ul) in enumerate(zip(l_trainloader,ul_trainloader)):
            volume_batch_l, label_batch_l = sampled_batch_l['image'], sampled_batch_l['label']
            volume_batch_ul, label_batch_ul = sampled_batch_ul['image'], sampled_batch_ul['label']
            volume_batch_l, label_batch_l, volume_batch_ul, label_batch_ul = \
            volume_batch_l.to(device), label_batch_l.to(device), volume_batch_ul.to(device),label_batch_ul.to(device)
            # volume_batch_l_2d = converToSlice(volume_batch_l)
            label_batch_l_2d = converToSlice(label_batch_l)
            volume_batch_ul_2d = converToSlice(volume_batch_ul)
            label_batch_ul_2d = converToSlice(label_batch_ul)
            volume_batch = torch.cat((volume_batch_l, volume_batch_ul), dim=0)
            volume_batch_2d = converToSlice(volume_batch)

            # use the trained protos to refine pseudo labels
            with torch.no_grad():
                u_output = t_model(x_2d=volume_batch_ul_2d, x_3d=volume_batch_ul, label=None, use_prototype=False)

                u_cls_seg = u_output["proto_seg"]
                low_feats = volume_batch_ul_2d
                high_feats = u_output["feature"]

                prob = F.softmax(u_cls_seg, dim=1)
                tree = mst_layers(low_feats)
                ASl = tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

                # high-level MST
                if high_feats is not None:
                    tree = mst_layers(high_feats)
                    AS = tree_filter_layers(feature_in=ASl, embed_in=high_feats, tree=tree,
                                            low_tree=False)  # [b, n, h, w]

                refined_label = torch.argmax(AS, dim=1)

                label_batch_2d = torch.cat((label_batch_l_2d, refined_label), dim=0)
                cls_pseudo_labels_3d = rearrange(
                    refined_label, "(b d) h w -> b h w d", b=batch_size)

            outputs = s_model(x_2d=volume_batch_2d, x_3d=volume_batch, label=label_batch_2d, use_prototype=True)


            cls_seg = outputs["cls_seg"]  # b,2,h,w
            loss_cls_ce = criterion(cls_seg[:batch_size*patch_size[2],:], label_batch_l_2d)
            outputs_soft = F.softmax(cls_seg[:batch_size*patch_size[2],:], dim=1)
            loss_seg_dice = dice_loss(outputs_soft, label_batch_l_2d)
            loss_cls_2d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice

            proto_seg = outputs["proto_seg"]
            outputs_soft = F.softmax(proto_seg[:batch_size*patch_size[2],:], dim=1)
            loss_seg_dice = dice_loss(outputs_soft, label_batch_l_2d)
            loss_proto_ce = criterion(proto_seg[:batch_size*patch_size[2],:], label_batch_l_2d)
            loss_proto = args.ce_w * loss_proto_ce + args.dice_w * loss_seg_dice

            cls_seg_3d = outputs["cls_seg_3d"]  # B,2,h,w,d
            loss_cls_seg_3d = criterion(cls_seg_3d[:batch_size,:], label_batch_l)
            outputs_soft_3d = F.softmax(cls_seg_3d[:batch_size,:], dim=1)
            loss_seg_dice_3d = dice_loss(outputs_soft_3d, label_batch_l)
            loss_cls_3d = args.ce_w  * loss_cls_seg_3d + args.dice_w * loss_seg_dice_3d


            loss_l = args.cls_w * loss_cls_2d + args.proto_w  * loss_proto + args.vol_w * loss_cls_3d

            # unlabel
            loss_cls_ce = criterion(cls_seg[batch_size * patch_size[2]:,:], refined_label)
            outputs_soft = F.softmax(cls_seg[batch_size * patch_size[2]:,:], dim=1)
            loss_seg_dice = dice_loss(outputs_soft, refined_label)
            loss_cls_2d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice

            loss_proto_ce = criterion(proto_seg[batch_size * patch_size[2]:,:], refined_label)
            outputs_soft = F.softmax(proto_seg[batch_size * patch_size[2]:,:], dim=1)
            loss_seg_dice = dice_loss(outputs_soft, refined_label)
            loss_proto = args.ce_w * loss_proto_ce + args.dice_w * loss_seg_dice

            loss_cls_seg_3d = criterion(cls_seg_3d[batch_size:,:], cls_pseudo_labels_3d)
            outputs_soft_3d = F.softmax(cls_seg_3d[batch_size:,:], dim=1)
            loss_seg_dice_3d = dice_loss(outputs_soft_3d, cls_pseudo_labels_3d)
            loss_cls_3d = args.ce_w  * loss_cls_seg_3d + args.dice_w * loss_seg_dice_3d

            loss_u = args.cls_w * loss_cls_2d + args.proto_w  * loss_proto + args.vol_w * loss_cls_3d

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = loss_l + consistency_weight * loss_u

            logging.info(
                'iteration %d : loss : %f loss_l : %f loss_u : %f '
                % (iter_num, loss.item(), loss_l.item(), loss_u.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update t_model
            update_ema_variables(s_model, t_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if dist.get_rank() == 0:
                if iter_num >= 1000 and iter_num % 50 == 0:
                    testing_dice_loss = validation(net=s_model, testloader=testloader)
                    logging.info('iter %d : testing_dice_loss : %f ' %
                                 (iter_num, testing_dice_loss))
                    if bestloss > testing_dice_loss:
                        bestIter = iter_num
                        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                        state = {
                            's_model': s_model.module.state_dict(),  # 训练好的参数
                            't_model': t_model.module.state_dict(),  # 训练好的参数
                        }
                        torch.save(state, save_mode_path)
                        logging.info("save Ourmodel to {}".format(save_mode_path))
                        bestloss = testing_dice_loss

                if iter_num % 500 == 0:
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    # torch.save(s_model.module.state_dict(), save_mode_path)
                    state = {
                        's_model': s_model.module.state_dict(),  # 训练好的参数
                        't_model': t_model.module.state_dict(),  # 训练好的参数
                    }
                    torch.save(state, save_mode_path)
                    logging.info("save Ourmodel to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break

    if dist.get_rank() == 0:
        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
        state = {
            's_model': s_model.module.state_dict(),  # 训练好的参数
            't_model': t_model.module.state_dict(),  # 训练好的参数
        }
        torch.save(state, save_mode_path)
        logging.info("save Ourmodel to {}".format(save_mode_path))
        logging.info("Best iternum {}".format(bestIter))
        writer.close()
