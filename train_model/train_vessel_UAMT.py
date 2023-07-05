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
from utils import ramps, losses
from dataloaders.brainvessel import BrainVessel, RandomCrop, CenterCrop, RandomRotFlip, ToTensor,TwoStreamBatchSampler
from utils.util import aggreate,aggreate_pred
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
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
device_ids=[0,1,2,3]
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (128,128,128)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

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
        # net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        # net = unet_3D(n_classes=num_classes)
        # net = dense_rnn_net()
        net = initialize_network(threeD=True)
        net.load_state_dict(torch.load('../Ourmodel/NNUNet_semi1/iter_6000.pth'))
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
                       data='unlabel',
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

    model.train()
    ema_model.train()
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    if args.consistency_type == 'mse':
        if num_classes==1:
            consistency_criterion = losses.sigmoid_mse_loss
        else:
            consistency_criterion = losses.softmax_mse_loss
            # feature_consistency_criterion = losses.mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch = volume_batch.permute(0, 1, 4, 2, 3)
            label_batch = label_batch.permute(0, 3, 1, 2)
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise
            outputs = model(volume_batch)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                U_cap = ema_inputs.detach().cpu().numpy()
                U_cap = U_cap[0, 0, 80,:, :] * 255.
                im = Image.fromarray(U_cap.astype(np.uint8))
                im.save('b2' + '.png')
                print('done11!')
                # visualize
                outputs_soft = F.softmax(ema_output, dim=1)
                outputs_soft = outputs_soft.detach().cpu().numpy()
                outputs_soft = np.argmax(outputs_soft,axis=1)
                outputs_soft = outputs_soft[0]*255.
                im = Image.fromarray(outputs_soft[80,:,:].astype(np.uint8))
                im.save('b1' + '.png')
                print("done!")
                break

            T = 8
            volume_batch_r = volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            if num_classes ==1:
                preds = torch.zeros([stride * T, 1, patch_size[0], patch_size[1], patch_size[2]]).cuda()
                preds = preds.permute(0, 1, 4, 2, 3)
                for i in range(T // 2):
                    ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                        preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
                # preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, 1, patch_size[0], patch_size[1], patch_size[2])
                preds = torch.mean(preds, dim=0)  # (batch, 2, 112,112,80)
                uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1,
                                               keepdim=True)  # (batch, 1, 112,112,80)
            else:
                preds = torch.zeros([stride * T, 2, patch_size[2],patch_size[0], patch_size[1]]).cuda()
                for i in range(T//2):
                    ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                        preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, 2, patch_size[2],patch_size[0], patch_size[1])
                preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
                uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)

            ## calculate the loss
            if num_classes ==1:
                pass
                # loss_seg = wbce(outputs[:labeled_bs], label_batch[:labeled_bs].unsqueeze(dim=1).float())
                # loss_seg_dice = losses.dice_loss(outputs[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)
            else:
                loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
                outputs_soft = F.softmax(outputs, dim=1)
                loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

                outputs_soft_3d = outputs_soft[:, 1, ...]
                outputs_aggreate = aggreate_pred(outputs_soft_3d)
                label_batch = label_batch.cpu().numpy()
                label_aggreate = aggreate(label_batch)
                label_aggreate = torch.from_numpy(label_aggreate).cuda()
                loss_seg_dice_3d_aggre = losses.dice_loss(outputs_aggreate, label_aggreate == 1)

            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = consistency_criterion(outputs, ema_output) #(batch, 2, 112,112,80)
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(2)
            mask = (uncertainty<threshold).float()
            consistency_dist = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
            consistency_loss = consistency_weight * consistency_dist
            loss = loss_seg+loss_seg_dice+0.5*loss_seg_dice_3d_aggre + consistency_loss

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
