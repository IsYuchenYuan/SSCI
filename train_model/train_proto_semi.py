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
from networks.unet_proto import UNetProto
import cleanlab
from utils import ramps,losses
from dataloaders.brainvessel import BrainVessel, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler,Normalization
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='Proto_3slice_semi_w1', help='model_name')
parser.add_argument('--max_iterations', type=int, default=8000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
parser.add_argument('--patchsize', type=list, default=[256, 256, 4],  help='whether use unlabel data')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.5, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
# refine
parser.add_argument("--sub_proto_size", type=int, default= 1, help="whether use subcluster")
parser.add_argument('--low_entropy_threshold', type=float,
                    default=0.3, help='percentage to choose low entropy')
parser.add_argument('--threshold', type=float,
                    default=0.7, help='threshold to filter out low reliable pres')
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
patch_size = (256, 256, 4)

###############loss##########
criterion = torch.nn.CrossEntropyLoss()
proto_loss = losses.PixelPrototypeCELoss()
criterion.to(device)
proto_loss.to(device)

mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02)

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


def converToThreeSlice(input):
    if len(input.shape) == 5:
        B, C, H, W, D = input.size()
        input2d = input[:, :, :, :, 0:2]
        single = input[:, :, :, :, 0:1]
        input2d = torch.cat((input2d, single), dim=4)
        for i in range(D - 2):
            input2dtmp = input[:, :, :, :, i:i + 3]
            input2d = torch.cat((input2d, input2dtmp), dim=0)
            if i == D - 3:
                f1 = input[:, :, :, :, D - 2: D]
                f2 = input[:, :, :, :, D - 1: D]
                ff = torch.cat((f1, f2), dim=4)
                input2d = torch.cat((input2d, ff), dim=0)
        input2d = input2d[:, 0, ...]  # squeeze the last channel C (BxD,H,W,3)
        input2d = input2d.permute(0, 3, 1, 2)  # BxD,3,H,W
    else:
        B, H, W, D = input.size()
        input2d = input[:, :, :, 0]
        for i in range(1, D):
            input2dtmp = input[:, :, :, i]
            input2d = torch.cat((input2d, input2dtmp), dim=0)
    return input2d


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def soft_cross_entropy(predicted, target):
    # print(predicted.type(), target.type())
    return -(target * torch.log(predicted)).sum(dim=1).mean()

def weighted_celoss(output,target,mask_fill):
    # input:[1,3,256,256]
    # target:[1,256,256] in {0,1,2}
    #mask_fill:[1,256,256]
    #这一步不用改，因为本来就是用的PyTorch的内置方法
    ls=torch.nn.LogSoftmax(dim=1)
    log_softmax=ls(output)
    bsize,h,w=target.shape[0],target.shape[1],target.shape[2]
    loss=0
    #由于batchsize一般都不会很大，因此该for循环花费时间很少
    for b in range(bsize):
        #下面是本次更改的部分
        #获取每个像素点的真实类别标签
        ind = target[b, :, :].type(torch.int64).unsqueeze(0)
        #print('ind:',ind.shape)#torch.Size([1, 256, 256])
        #获取预测得到的每个像素点的类别取值分布（3代表类别）
        pred_3channels=log_softmax[b,:,:,:]
        #print('pred_3channels:',pred_3channels.shape)#torch.Size([3, 256, 256])
        #使用gather，在第0个维度（类别所在维度）上用ind进行索引得到每个像素点的value
        pred=-pred_3channels.gather(0,ind)
        #print('pred:',pred.shape)#torch.Size([1, 256, 256])
        #添加了这句代码，通过两者的点乘实现了对每个像素点的加权
        pred=pred*mask_fill
        #求这些像素点value的平均值，并累加到总的loss中
        current_loss=torch.mean(pred)
        loss+=current_loss
    return loss/bsize

def validation(net, testloader):
    net.eval()
    val_dice_loss = 0.0

    with torch.no_grad():
        for i, sampled_batch in enumerate(testloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image = volume_batch.to(device)
            label = label_batch.to(device)
            input2d = converToSlice(image)
            label2d = converToSlice(label)

            outputs = net(x=input2d,
                          label=label2d,
                          use_prototype=False)
            cls_seg = outputs["cls_seg"]  # b,2,h,w
            loss_cls_seg = criterion(cls_seg, label2d)
            outputs_soft = F.softmax(cls_seg, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:, 1, :, :], label2d == 1)
            loss_cls = 0.5 * (loss_cls_seg + loss_seg_dice)
            val_dice_loss = loss_cls.item()

    val_dice_loss /= (i + 1)
    return val_dice_loss

from datetime import datetime
currentDateAndTime = datetime.now()

currentTime = currentDateAndTime.strftime("%H%M%S")

def visulize(refine, coarse):
    path = "../testimages/refinelabel/"
    if not os.path.exists(path):
        os.makedirs(path)
    # img=img.cpu().numpy()
    # img = img[0][0]
    # imgs_normalized = ((img - np.min(img)) / (
    #         np.max(img) - np.min(img)))
    # img = imgs_normalized*255.
    # img = Image.fromarray(img.astype(np.uint8))
    # img.save(path +"img_" + currentTime + ".png")
    #
    # gt = gt.cpu().numpy()
    # gt = gt[0]
    # gt = gt * 255.
    # gt = Image.fromarray(gt.astype(np.uint8))
    # gt.save("../testimages/proto/gt.png")
    #
    label = refine.detach().cpu().numpy()
    label = label[0]
    label = label * 255.
    label = Image.fromarray(label.astype(np.uint8))
    label.save(path +"refine_" + currentTime + ".png")

    label = coarse.detach().cpu().numpy()
    label = label[0]
    label = label * 255.
    label = Image.fromarray(label.astype(np.uint8))
    label.save(path + "coarse_" + currentTime + ".png")

    # proto = proto.detach().cpu().numpy()
    # proto = proto[0]
    # proto = proto * 255.
    # proto = Image.fromarray(proto.astype(np.uint8))
    # proto.save(path +"proto_" + currentTime + ".png")
    #
    # refine = refine.cpu().numpy()
    # refine = refine[0]
    # refine = refine * 255.
    # refine = Image.fromarray(refine.astype(np.uint8))
    # refine.save(path +"refine_" + currentTime + ".png")
    #
    # low = low.cpu().numpy()
    # low = low[0]
    # low = low * 255.
    # low = Image.fromarray(low.astype(np.uint8))
    # low.save(path +"hrl_" + currentTime + ".png")
    #
    # high = high.cpu().numpy()
    # high = high[0]
    # high = high * 255.
    # high = Image.fromarray(high.astype(np.uint8))
    # high.save(path +"lrl_" + currentTime + ".png")

def savefile(p):
    proto = p.detach().cpu().numpy()
    proto_0 = proto[0][0] #2,h,w
    np.savetxt('001.txt', proto_0)
    proto_1 = proto[0][1]  # 2,h,w
    np.savetxt('002.txt', proto_1)


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

    checkpoint = "../Ourmodel/Proto_3slice_semi/iter_7000.pth"
    def create_model(pretrain=False,ema=False):
        net = UNetProto(
            inchannel=3,
            nclasses=2,
            proj_dim=256,
            projection="v1",
            l2_norm=True,
            proto_mom=0.999,
            sub_proto_size=args.sub_proto_size,
        )
        net = net.to(device)
        if pretrain:
            net.load_state_dict(torch.load(checkpoint))
        model = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    s_model = create_model(pretrain=False,ema=False)
    t_model = create_model(pretrain=False,ema=True)


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
    s_model.train()
    t_model.train()
    x_criterion = soft_cross_entropy
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
        l_sampler.set_epoch(epoch_num)
        ul_sampler.set_epoch(epoch_num)

        alpha_t = args.low_entropy_threshold * (
                1 - epoch_num / max_epoch
        )
        time1 = time.time()
        for i_batch, (sampled_batch_l,sampled_batch_ul) in enumerate(zip(l_trainloader,ul_trainloader)):
            time2 = time.time()
            volume_batch_l, label_batch_l = sampled_batch_l['image'], sampled_batch_l['label']
            volume_batch_ul, label_batch_ul = sampled_batch_ul['image'], sampled_batch_ul['label']
            volume_batch_l, label_batch_l, volume_batch_ul = \
                volume_batch_l.to(device), label_batch_l.to(device), volume_batch_ul.to(device)
            # turn 3D label into 2D
            label_batch_l_2d = converToSlice(label_batch_l)
            # for student model, with all data
            volume_batch_l_2d = converToSlice(volume_batch_l)
            volume_batch_ul_2d = converToSlice(volume_batch_ul)
            noise = torch.clamp(torch.randn_like(volume_batch_ul) * 0.1, -0.2, 0.2)
            noise_batch_ul = volume_batch_ul + noise
            noise_batch_ul_2d = converToSlice(noise_batch_ul)
            volume_batch_2d = torch.cat((volume_batch_l_2d, noise_batch_ul_2d),dim=0)

            # volume_batch = torch.cat((volume_batch_l, noise_batch_ul), dim=0)
            # volume_batch = volume_batch.to(device)
            # volume_batch_2d = converToThreeSlice(volume_batch)


            # warm up phase
            if iter_num < 200:
                if iter_num ==0:
                    logging.info("*********************************************************")
                    logging.info("This is warming up stage..........")
                    logging.info("*********************************************************")
                cls_seg = s_model.module.warm_up(x=volume_batch_l_2d)
                loss = criterion(cls_seg, label_batch_l_2d)
                logging.info(
                    'iteration %d : loss : %f '
                    % (iter_num, loss.item()))

            # initialize and update protos using labeled data
            elif iter_num >= 200 and iter_num < 1000:
                if iter_num == 200:
                    # if dist.get_rank()==0:
                    logging.info("*********************************************************")
                    logging.info("initialize and update protos using labeled data..........")
                    logging.info("*********************************************************")
                outputs = s_model(x=volume_batch_l_2d, label=label_batch_l_2d, use_prototype=True)
                cls_seg = outputs["cls_seg"]  # b,2,h,w
                proto_seg = outputs["proto_seg"]

                loss_cls_seg = criterion(cls_seg, label_batch_l_2d)
                outputs_soft = F.softmax(cls_seg, dim=1)
                loss_seg_dice = losses.dice_loss(outputs_soft[:, 1, :, :], label_batch_l_2d == 1)
                loss_cls = 0.5 * (loss_cls_seg + loss_seg_dice)
                if args.sub_proto_size != 1:
                    contrast_logits = outputs["contrast_logits"]
                    contrast_target = outputs["contrast_target"]
                    loss_proto = proto_loss(proto_seg,contrast_logits,contrast_target, label_batch_l_2d)
                else:
                    outputs_soft = F.softmax(proto_seg, dim=1)
                    loss_seg_dice = losses.dice_loss(outputs_soft[:, 1, :, :], label_batch_l_2d == 1)
                    loss_proto_ce = criterion(proto_seg, label_batch_l_2d)
                    loss_proto = (loss_proto_ce + loss_seg_dice) * 0.5

                loss = loss_cls + 0.5 * loss_proto


                logging.info(
                    'iteration %d : loss_l : %f  loss_cls : %f   loss_proto : %f '
                    % (iter_num, loss.item(), loss_cls.item(), loss_proto.item()))

            # update protos using both labeled and ul data
            else:
                if iter_num == 1001:
                    logging.info("*********************************************************")
                    logging.info("update protos using both labeled data and ul data..........")
                    logging.info("*********************************************************")
                # use the trained protos to refine pseudo labels
                with torch.no_grad():
                    u_output = t_model(x=volume_batch_ul_2d, label = None, use_prototype = False)
                    u_cls_seg = u_output["cls_seg"]
                    high_feats = u_output["feature"]
                    low_feats = volume_batch_ul_2d
                    prob = F.softmax(u_cls_seg, dim=1)

                    tree = mst_layers(low_feats)
                    AS = tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

                    # high-level MST
                    if high_feats is not None:
                        tree = mst_layers(high_feats)
                        AS =  tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree,
                                                     low_tree=False)  # [b, n, h, w]

                    refined_label = torch.argmax(AS,dim=1)
                    if iter_num == 7000:
                        if dist.get_rank()==0:
                            try:
                                coarse_label = torch.argmax(prob,dim=1)
                                visulize(refined_label,coarse_label)
                            except:
                                print("error in visualizing")

                    label_batch_2d = torch.cat((label_batch_l_2d,refined_label),dim = 0)

                outputs = s_model(x=volume_batch_2d, label=label_batch_2d, use_prototype=True)
                cls_seg = outputs["cls_seg"]  # b,2,h,w
                proto_seg = outputs["proto_seg"]

                loss_cls_seg = criterion(cls_seg[:batch_size*patch_size[2], :], label_batch_l_2d)
                outputs_soft = F.softmax(cls_seg[:batch_size*patch_size[2], :], dim=1)
                loss_seg_dice = losses.dice_loss(outputs_soft[:, 1, :, :], label_batch_l_2d == 1)
                loss_cls = 0.5 * (loss_cls_seg + loss_seg_dice)

                loss_proto_cls = criterion(proto_seg[:batch_size*patch_size[2], :], label_batch_l_2d)
                outputs_soft_proto = F.softmax(proto_seg[:batch_size*patch_size[2], :], dim=1)
                loss_seg_dice_proto = losses.dice_loss(outputs_soft_proto[:, 1, :, :], label_batch_l_2d == 1)
                loss_proto = (loss_seg_dice_proto + loss_proto_cls) * 0.5

                loss_l = loss_cls + 0.5 * loss_proto

                loss_cls_seg = criterion(cls_seg[batch_size * patch_size[2]:, :],refined_label)
                outputs_soft = F.softmax(cls_seg[batch_size * patch_size[2]:, :], dim=1)
                loss_seg_dice = losses.dice_loss(outputs_soft[:, 1, :, :], refined_label == 1)
                loss_cls = 0.5 * (loss_cls_seg + loss_seg_dice)

                loss_proto_cls = weighted_celoss(proto_seg[batch_size * patch_size[2]:, :], refined_label)
                outputs_soft_proto = F.softmax(proto_seg[batch_size * patch_size[2]:, :], dim=1)
                loss_seg_dice_proto = losses.dice_loss(outputs_soft_proto[:, 1, :, :], refined_label == 1)
                loss_proto = (loss_seg_dice_proto + loss_proto_cls) * 0.5

                loss_ul = loss_cls_seg + 0.5 * loss_proto_cls

                consistency_weight = get_current_consistency_weight(
                    iter_num // (max_iterations // args.consistency_rampup))

                loss = loss_l + consistency_weight * loss_ul

                if dist.get_rank() == 0:
                    writer.add_scalar('lr', lr_, iter_num)
                    writer.add_scalar('loss/loss_l', loss_l, iter_num)
                    writer.add_scalar('loss/loss_ul', loss_ul, iter_num)
                    writer.add_scalar('loss/loss', loss, iter_num)
                logging.info(
                    'iteration %d : loss : %f loss_l : %f loss_ul : %f '
                    % (iter_num, loss.item(), loss_l.item(), loss_ul.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update t_model
            update_ema_variables(s_model, t_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1


            ## change lr
            if iter_num % 3000 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 3000)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if dist.get_rank() == 0:
                if iter_num >= 2000 and iter_num % 50 == 0:
                    testing_dice_loss = validation(net=s_model, testloader=testloader)
                    logging.info('iter %d : testing_dice_loss : %f ' %
                                 (iter_num, testing_dice_loss))
                    if bestloss > testing_dice_loss:
                        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                        torch.save(s_model.module.state_dict(), save_mode_path)
                        logging.info("save Ourmodel to {}".format(save_mode_path))
                        bestloss = testing_dice_loss

                if iter_num % 500 == 0:
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save(s_model.module.state_dict(), save_mode_path)
                    logging.info("save Ourmodel to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break

    if dist.get_rank() == 0:
        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
        torch.save(s_model.module.state_dict(), save_mode_path)
        logging.info("save Ourmodel to {}".format(save_mode_path))
        writer.close()
