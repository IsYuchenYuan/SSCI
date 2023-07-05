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
from networks.nnunet import initialize_network
from networks.vnet import VNet
# from networks.UNet_3d import unet_3D
from networks.DenseUNet3d import dense_rnn_net
from utils.losses import dice_loss,InstanceWeightedBCELoss
from dataloaders.brainvessel import BrainVessel, RandomCrop, CenterCrop, RandomRotFlip, ToTensor



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DenseUNet3d', help='model_name')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1', help='GPU to use')
parser.add_argument('--multi_gpu', type=int, default=1,  help='whether use multipule gpu')
parser.add_argument('--label', type=str, default='label',  help='whether use unlabel data')
parser.add_argument('--patchsize', type=list, default=[128,128,96],  help='whether use unlabel data')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../Ourmodel/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor((0.3,0.7), device='cuda'), reduction='mean')
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
wbce = InstanceWeightedBCELoss()
wbce.cuda()

# patch_size = (224, 224, 12)
# patch_size = (112, 112, 80)
patch_size = (128,128,96)
num_classes = 2
device_ids=[0,1]
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
                           ToTensor(),
                       ]))
    # db_test = BrainVessel(base_dir=train_data_path,
    #                   split='test',
    #                   fold=1,
    #                   transform=transforms.Compose([
    #                       CenterCrop(patch_size),
    #                       ToTensor()
    #                   ]))

    # net = initialize_network(threeD=True)
    # net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    # net = unet_3D(n_classes=num_classes)
    net= dense_rnn_net()
    net = net.cuda()
    ## use dataparallel
    if args.multi_gpu:
        net = torch.nn.DataParallel(net, device_ids=device_ids)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                 worker_init_fn=worker_init_fn)

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs2d = net(volume_batch)
            if num_classes==1:
                label_batch_x = label_batch.float().unsqueeze(dim=1)
                loss_seg = wbce(outputs2d, label_batch_x)
                loss_seg_dice = dice_loss(outputs2d[:, 0, :, :, :], label_batch == 1)
            else:
                loss_seg_2d = criterion(outputs2d, label_batch)
                outputs_soft_2d = F.softmax(outputs2d, dim=1)
                loss_seg_dice_2d = dice_loss(outputs_soft_2d[:, 1, :, :, :], label_batch == 1)


            loss = (loss_seg_2d + loss_seg_dice_2d ) * 0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg_2d', loss_seg_2d, iter_num)
            writer.add_scalar('loss/loss_seg_dice_2d', loss_seg_dice_2d, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f loss_seg_2d : %f  loss_seg_dice_2d : %f'
                         % (iter_num, loss.item(),loss_seg_2d.item(),loss_seg_dice_2d.item()))
            # if iter_num % 50 == 0:
            #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)
            #
            #     outputs_soft = F.softmax(outputs, 1)
            #     image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Predicted_label', grid_image, iter_num)
            #
            #     image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 100 == 0 :
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                if args.multi_gpu:
                    torch.save(net.module.state_dict(), save_mode_path)
                else:
                    torch.save(net.state_dict(), save_mode_path)
                logging.info("save Ourmodel to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations + 1) + '.pth')
    if args.multi_gpu:
        torch.save(net.module.state_dict(), save_mode_path)
    else:
        torch.save(net.state_dict(), save_mode_path)
    logging.info("save Ourmodel to {}".format(save_mode_path))
    writer.close()
