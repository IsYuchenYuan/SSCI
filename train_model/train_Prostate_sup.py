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
# from networks.unet_proto import UNetProto
from networks.unet_proto_hybrid import UNetProto
from utils import losses
from dataloaders.cardiac import Prostate3d, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, SmoothLable
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.util import converToSlice


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='Prostate_sup_3d', help='model_name')
parser.add_argument('--max_iterations', type=int, default = 6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
parser.add_argument('--multi_gpu', type=int, default=1,  help='whether use multipule gpu')
parser.add_argument('--patchsize', type=list, default=[112,112,80],  help='whether use unlabel data')
parser.add_argument("--local_rank", type=int, help="")
parser.add_argument("--sub_proto_size", type=int, default= 2, help="whether use subcluster")

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../Ourmodel/" + args.exp + "/"

batch_size = 1
max_iterations = args.max_iterations
base_lr = args.base_lr
patch_size = args.patchsize
num_classes = 4

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

########################distribution##############

dist.init_process_group(backend='nccl')
# local_rank = torch.distributed.get_rank()
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

############ loss ########
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)
dice_loss = losses.DiceLoss(num_classes)
proto_loss = losses.PixelPrototypeCELoss()


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
    # make logger file
    bestloss = np.inf
    bestIter = 0
    if dist.get_rank() == 0:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

    db_train = Prostate3d(data_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                   ]))
    db_test = Prostate3d(data_dir=train_data_path,
                      split='test',
                      transform=transforms.Compose([
                          RandomRotFlip(),
                          CenterCrop(patch_size),
                          ToTensor(),
                      ]))

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
    net = DDP(net, device_ids=[local_rank],output_device=local_rank,
                                             find_unused_parameters=True)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    train_sampler = DistributedSampler(db_train)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                 worker_init_fn=worker_init_fn,sampler=train_sampler)
    testloader = DataLoader(db_test, batch_size=batch_size, num_workers=4, pin_memory=False,
                            worker_init_fn=worker_init_fn)
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    if dist.get_rank() == 0:
        writer = SummaryWriter(snapshot_path + '/log')
        logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        train_sampler.set_epoch(epoch_num)
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            input2d = converToSlice(volume_batch)
            label2d = converToSlice(label_batch)

            if iter_num < 1000:

                cls_seg, cls_seg_3d = net.module.warm_up(x_2d=input2d,x_3d=volume_batch)

                loss_cls_ce = criterion(cls_seg, label2d)
                outputs_soft = F.softmax(cls_seg, dim=1)
                loss_seg_dice = dice_loss(outputs_soft, label2d)
                loss_cls_2d = 0.5 * (loss_cls_ce + loss_seg_dice)

                loss_cls_ce_3d = criterion(cls_seg_3d, label_batch)
                outputs_soft_3d = F.softmax(cls_seg_3d, dim=1)
                loss_seg_dice_3d = dice_loss(outputs_soft_3d, label_batch)
                loss_cls_3d = 0.5 * (loss_cls_ce_3d + loss_seg_dice_3d)

                loss_proto = torch.tensor([0]).to(device)

            elif iter_num >= 1000:
                outputs = net(x_2d=input2d, x_3d = volume_batch, label=label2d, use_prototype=True)

                cls_seg = outputs["cls_seg"]  # b,c,h,w
                loss_cls_ce = criterion(cls_seg, label2d)
                outputs_soft = F.softmax(cls_seg, dim=1)
                loss_seg_dice = dice_loss(outputs_soft, label2d)
                loss_cls_2d = 0.5 * (loss_cls_ce + loss_seg_dice)

                cls_seg_3d = outputs["cls_seg_3d"]  # B,c,h,w,d
                loss_cls_ce_3d = criterion(cls_seg_3d, label_batch)
                outputs_soft_3d = F.softmax(cls_seg_3d, dim=1)
                loss_seg_dice_3d = dice_loss(outputs_soft_3d, label_batch)
                loss_cls_3d = 0.5 * (loss_cls_ce_3d + loss_seg_dice_3d)

                proto_seg = outputs["proto_seg"]
                # guessed = proto_seg ** (1 / 0.5)
                # proto_seg = guessed / guessed.sum(dim=1, keepdim=True)
                if args.sub_proto_size != 1:
                    contrast_logits = outputs["contrast_logits"]
                    contrast_target = outputs["contrast_target"]
                    loss_proto = proto_loss(proto_seg, contrast_logits, contrast_target, label2d)
                else:
                    outputs_soft = F.softmax(proto_seg, dim=1)
                    loss_seg_dice = dice_loss(outputs_soft, label2d)
                    loss_proto_ce = criterion(proto_seg, label2d)
                    loss_proto = (loss_proto_ce + loss_seg_dice) * 0.5


            loss = (loss_cls_2d + loss_proto) * 0.5 + loss_cls_3d
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().clone()
            dist.all_reduce(
                loss.div_(dist.get_world_size()))

            iter_num = iter_num + 1
            if dist.get_rank() == 0:
                logging.info('iteration %d : avg loss : %f  loss_cls_2d : %f loss_cls_3d : %f loss_proto : %f '
                             % (iter_num, loss.item(), loss_cls_2d.item(), loss_cls_3d.item(), loss_proto.item()))

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if dist.get_rank() == 0:
                if iter_num > 2000 and iter_num % 50 == 0:
                    testing_dice_loss = validation(net=net, testloader=testloader)
                    logging.info('iter %d : testing_dice_loss : %f ' %
                                 (iter_num, testing_dice_loss))
                    if bestloss > testing_dice_loss:
                        bestIter = iter_num
                        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                        torch.save(net.module.state_dict(), save_mode_path)
                        logging.info("save Ourmodel to {}".format(save_mode_path))
                        bestloss = testing_dice_loss

                if iter_num % 500 == 0:
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save(net.module.state_dict(), save_mode_path)
                    logging.info("save Ourmodel to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()

        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations + 1) + '.pth')
    if dist.get_rank() == 0:
        state = {
                 'state_dict': net.module.state_dict(),  # 训练好的参数
                 'prototype': net.module.prototypes,  # 优化器参数,为了后续的resume
                 }
        torch.save(state, save_mode_path)
        logging.info("save Ourmodel to {}".format(save_mode_path))
        logging.info("best iter {}".format(bestIter))
        writer.close()
