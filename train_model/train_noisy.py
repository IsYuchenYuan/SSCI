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
from utils.losses import dice_loss,InstanceWeightedBCELoss
from utils.util import aggreate,aggreate_pred
from dataloaders.brainvessel import BrainVessel, RandomCrop, CenterCrop, RandomRotFlip, ToTensor_mixmatch
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='NNUNet_noisy_w', help='model_name')
parser.add_argument('--max_iterations', type=int, default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
parser.add_argument('--multi_gpu', type=int, default=1,  help='whether use multipule gpu')
parser.add_argument('--label', type=str, default='label',  help='whether use unlabel data')
parser.add_argument('--patchsize', type=list, default=[128,128,128],  help='whether use unlabel data')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../Ourmodel/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
# criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor((0.3,0.7), device='cuda'), reduction='mean')
def weighted_cross_entropy(predicted, target, weighted):
    #print(predicted.type(), target.type())
    ce = -(target * torch.log(predicted)).sum(dim=1)
    ce = ce * weighted
    return ce.mean()

def dice_loss_weighted(score, target, weighted):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target * weighted)
    y_sum = torch.sum(target * target * weighted)
    z_sum = torch.sum(score * score * weighted)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

criterion = weighted_cross_entropy
dice_criterion = dice_loss_weighted
# criterion.cuda()
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



if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    db_train = BrainVessel(base_dir=train_data_path,
                       data=args.label,
                       fold=5,
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor_mixmatch(),
                       ]))


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

    net1 = create_model()
    net2 = create_model()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                 worker_init_fn=worker_init_fn)

    net1.train()
    net2.train()
    # optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer1 = optim.Adam(net1.parameters(), lr=base_lr, weight_decay=0.0001)
    optimizer2 = optim.Adam(net2.parameters(), lr=base_lr, weight_decay=0.0001)
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
            label_batch = label_batch.permute(0,1,4,2,3)
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            perturbation_inputs = volume_batch + noise
            outputs1 = net1(volume_batch)
            outputs2 = net2(perturbation_inputs)
            # the same pre: easy sample and hard sample -> 0.1
            # the different pre: label is good -> 0.9
            outputs1 = F.softmax(outputs1, dim=1)
            outputs2 = F.softmax(outputs2, dim=1)
            a =outputs1.detach().cpu().numpy()
            b =outputs2.detach().cpu().numpy()
            a = np.argmax(a, axis=1)
            b = np.argmax(b, axis=1)
            att = a^b
            att[att==1] = 0.7
            att[att == 0] = 0.3
            att = torch.from_numpy(att)
            att = att.cuda()
            loss_seg_1 = criterion(outputs1, label_batch,att)
            loss_seg_2 = criterion(outputs2, label_batch,att)

            loss_seg_dice_1 = dice_loss(outputs1[:, 1, :, :, :], label_batch[:, 1, :, :, :] == 1)
            loss_seg_dice_2 = dice_loss(outputs2[:, 1, :, :, :], label_batch[:, 1, :, :, :] == 1)

            # outputs_soft_3d = outputs_soft_3d[:,1,...]
            # outputs_aggreate = aggreate_pred(outputs_soft_3d)
            # label_batch = label_batch.cpu().numpy()
            # label_aggreate = aggreate(label_batch)
            # label_aggreate = torch.from_numpy(label_aggreate).cuda()
            # loss_seg_dice_3d_aggre = dice_loss(outputs_aggreate, label_aggreate == 1)

            loss1 = (loss_seg_1 + loss_seg_dice_1) * 0.5
            loss2 = (loss_seg_2 + loss_seg_dice_2) * 0.5
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            iter_num = iter_num + 1
            # writer.add_scalar('lr', lr_, iter_num)
            # writer.add_scalar('loss/loss_seg_3d', loss_seg_3d, iter_num)
            # writer.add_scalar('loss/loss_seg_3d', loss_seg_dice_3d, iter_num)
            # writer.add_scalar('loss/loss_seg_3d_aggre', loss_seg_dice_3d_aggre, iter_num)
            # writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss1 : %f loss_seg_3d1 : %f loss_seg_dice_3d1 : %f loss_seg_3d2 : %f loss_seg_dice_3d2 : %f '
                         % (iter_num, loss1.item(), loss_seg_1.item(), loss_seg_dice_1.item(), loss_seg_2.item(), loss_seg_dice_2))

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0 :
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                if args.multi_gpu:
                    torch.save(net1.module.state_dict(), save_mode_path)
                    torch.save(net2.module.state_dict(), save_mode_path)
                else:
                    torch.save(net1.state_dict(), save_mode_path)
                logging.info("save Ourmodel to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()

        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations + 1) + '.pth')
    if args.multi_gpu:
        torch.save(net1.module.state_dict(), save_mode_path)
        torch.save(net2.module.state_dict(), save_mode_path)
    else:
        torch.save(net1.state_dict(), save_mode_path)
        torch.save(net2.state_dict(), save_mode_path)
    logging.info("save Ourmodel to {}".format(save_mode_path))
    writer.close()
