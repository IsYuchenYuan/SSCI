import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from networks.unet_proto import UNetProto
from utils import ramps,losses
import torch.backends.cudnn as cudnn
from utils.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from utils.lib_tree_filter.modules.tree_filter import TreeFilter2D
from utils.torch_poly_lr_decay import PolyLR
from dataloaders.cardiac import ACDC
from test_slice import test_eval


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC/training/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='', help='model name')
parser.add_argument('--percentage', type=float, default='', help='labeled percentage [0.0125,0.025,0.1]')

parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train the whole framework')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--min_lr', type=float, default=1e-6, help='minmum lr the scheduler to reach')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--patchsize', type=list, default=[256, 256],  help='size of input patch')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float, default='2', help='loss weight of unlabeled data (you can change to suit the dataset)')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')

parser.add_argument("--sub_proto_size", type=int, default= 2, help="whether to use subcluster")
parser.add_argument("--pretrainIter", type=int, default=10000, help="maximum iteration to train both classifiers by using labeled data only")
parser.add_argument("--linearIter", type=int, default=1000, help="maximum iteration to train the LC")
parser.add_argument("--initialP", type=bool, default=False, help="whether to initialize prototype")
parser.add_argument("--dice_w", type=float, default=0.5, help="the weight of dice loss (you can change to suit the dataset)")
parser.add_argument("--ce_w", type=float, default=0.5, help="the weight of ce loss (you can change to suit the dataset)")
parser.add_argument("--proto_w", type=float, default=1, help="the weight of proto loss (you can change to suit the dataset)")
parser.add_argument('--proto_rampup', type=float, default=40.0, help='proto_rampup')
parser.add_argument("--multiDSC", type=bool, default=True, help="whether to use multiDSC (set False if labeled data is very few)")
parser.add_argument("--losstype", type=str, default="ce_dice", help="the type of ce and dice loss")

parser.add_argument("--temp", type=float, default=10, help="softmax temperature")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
train_data_path = args.root_path
snapshot_path = "../Ourmodel/final/" + args.exp + "/"
batch_size = args.batch_size
labeled_bs = args.labeled_bs
max_iterations = args.max_iterations
base_lr = args.base_lr
num_classes = 4

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

###############loss##########
weights = torch.from_numpy(np.asarray([0.05,0.35,0.35,0.25])).float().cuda()

if args.losstype=="wce_dice":
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    dice_loss = losses.DiceLoss(num_classes,weights=None)
elif args.losstype=="wce_wdice":
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    dice_loss = losses.DiceLoss(num_classes,weights=weights)
elif args.losstype=="ce_dice":
    criterion = torch.nn.CrossEntropyLoss(weight=None)
    dice_loss = losses.DiceLoss(num_classes,weights=None)

proto_loss = losses.PixelPrototypeCELoss()
mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
tree_filter_layers = TreeFilter2D(groups=1, sigma=0.05,gamma=1)


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_current_proto_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.proto_w * ramps.sigmoid_rampup(epoch, args.proto_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def rand_bbox_1(size, lam=None):
    # past implementation
    W = size[2]
    H = size[3]
    B = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mix(unlabeled_image=None, unlabeled_mask=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]].cuda()
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_1(unlabeled_image.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_image, unlabeled_mask

    return mix_unlabeled_image, mix_unlabeled_target


with open('../data/ACDC/' + 'train.txt', 'r') as f:
    lists = f.readlines()
image_list = lists[80:]
image_list = [item.replace('\n', '') for item in image_list]


if __name__ == "__main__":
    ## make logger file
    bestloss = np.inf
    bestdice = -np.inf
    bestIter = 0
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pretrain = False
    if args.initialP:
        pretrain = True

    def create_model(pretrain=False,ema=False):
        net = UNetProto(
            inchannel=1,
            nclasses=num_classes,
            proj_dim=64,
            projection=None,
            l2_norm=True,
            proto_mom=0.999,
            sub_proto_size=args.sub_proto_size,
            proto=None,
            temp=args.temp
        )
        net = net.cuda()
        if pretrain:
            checkpoint = ''
            net.load_state_dict(torch.load(checkpoint))
        if ema:
            for param in net.parameters():
                param.detach_()
        return net


    s_model = create_model(pretrain=pretrain,ema=False)
    t_model = create_model(pretrain=pretrain,ema=True)


    db_train_l = ACDC(data_dir=train_data_path,
                        split='train_l',
                        percentage=args.percentage
                        )

    db_train_ul = ACDC(data_dir=train_data_path,
                    split='train_ul',
                    percentage=args.percentage
                    )

    db_test = ACDC(data_dir=train_data_path,
                         split='test'
                        )


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    l_trainloader = DataLoader(db_train_l, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    ul_trainloader = DataLoader(db_train_ul, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=batch_size, num_workers=1, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    max_epoch = max_iterations // len(l_trainloader) + 1

    s_model.train()
    t_model.train()

    optimizer = optim.SGD(s_model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    lr_scheduler = PolyLR(optimizer, max_epoch, min_lr=args.min_lr)


    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(l_trainloader)))

    if args.initialP:
        iter_num = args.pretrainIter
    else:
        iter_num = 0
    count = 0
    for epoch_num in tqdm(range(max_epoch), ncols=70):

        for i_batch, (sampled_batch_l,sampled_batch_ul) in enumerate(zip(l_trainloader,ul_trainloader)):
            for param_group in optimizer.param_groups:
                lr_ = param_group['lr']
            for param_group in optimizer.param_groups:
                logging.info("current lr: {}".format(param_group['lr']))
                break
            volume_batch_l, label_batch_l = sampled_batch_l['image'].cuda(), sampled_batch_l['label'].cuda()
            volume_batch_ul, label_batch_ul = sampled_batch_ul['image'].cuda(), sampled_batch_ul['label'].cuda()

            if iter_num < args.linearIter:
                cls_seg = s_model.warm_up(x_2d=volume_batch_l)
                loss_cls_ce = criterion(cls_seg, label_batch_l)
                outputs_soft = F.softmax(cls_seg, dim=1)
                loss_seg_dice = dice_loss(outputs_soft, label_batch_l)
                loss_cls_2d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice
                loss = loss_cls_2d
                logging.info('training linear cls only !!!!! iteration %d : avg loss : %f  '
                             % (iter_num, loss.item()))
            elif iter_num < args.pretrainIter:
                outputs = s_model(x_2d=volume_batch_l, label=label_batch_l, use_prototype=True)

                cls_seg = outputs["cls_seg"]  # b,c,h,w
                loss_cls_ce = criterion(cls_seg, label_batch_l)
                if args.multiDSC:
                    outputs_soft = F.softmax(cls_seg, dim=1)
                    loss_seg_dice = dice_loss(outputs_soft, label_batch_l)
                else:
                    loss_seg_dice = losses.dice_loss_foreground(cls_seg,label_batch_l)
                loss_cls_2d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice

                proto_seg = outputs["proto_seg"]

                loss_proto_ce = criterion(proto_seg, label_batch_l)
                if args.multiDSC:
                    outputs_soft = F.softmax(proto_seg, dim=1)
                    loss_proto_dice = dice_loss(outputs_soft, label_batch_l)
                else:
                    loss_proto_dice = losses.dice_loss_foreground(proto_seg, label_batch_l)
                loss_proto_2d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice

                loss = loss_cls_2d + args.proto_w * loss_proto_2d
                logging.info('train both cls!!!!!!!!! iteration %d : avg loss : %f  loss_cls_2d : %f  '
                             'loss_proto : %f '
                             % (iter_num, loss.item(), loss_cls_2d.item(),
                                loss_proto_2d.item()))
            elif iter_num >= args.pretrainIter:
                with torch.no_grad():
                    u_output = t_model(x_2d=volume_batch_ul, label=None, use_prototype=False)
                    u_cls_seg_p = u_output["proto_seg"]
                    high_feats = u_output["feature"]
                    low_feats = volume_batch_ul
                    prob = F.softmax(u_cls_seg_p, dim=1)

                    treel = mst_layers(low_feats)
                    ASl = tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=treel)  # [b, n, h, w]
                    if high_feats is not None:
                        treeh = mst_layers(high_feats)
                        ASh = tree_filter_layers(feature_in=prob, embed_in=high_feats, tree=treeh,
                                                low_tree=False)  # [b, n, h, w]
                        ASlh = tree_filter_layers(feature_in=ASl, embed_in=high_feats, tree=treeh,
                                                  low_tree=False)  # [b, n, h, w]

                    # refineh = torch.argmax(ASlh, dim=1)
                    refineh = torch.argmax(ASh, dim=1)
                    refined_label = refineh


                # no cutmix
                volume_batch = torch.cat((volume_batch_l, volume_batch_ul), dim=0)
                label_batch = torch.cat((label_batch_l, refined_label), dim=0)


                outputs = s_model(x_2d=volume_batch, label=label_batch, use_prototype= True)
                cls_seg = outputs["cls_seg"]  # b,2,h,w
                proto_seg = outputs["proto_seg"]


                # supervised loss
                loss_cls_ce = criterion(cls_seg[:labeled_bs], label_batch[:labeled_bs])
                if args.multiDSC:
                    outputs_soft = F.softmax(cls_seg[:labeled_bs], dim=1)
                    loss_seg_dice = dice_loss(outputs_soft, label_batch[:labeled_bs])
                else:
                    loss_seg_dice = losses.dice_loss_foreground(cls_seg[:labeled_bs], label_batch[:labeled_bs])
                loss_cls_2d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice

                loss_proto_ce = criterion(proto_seg[:labeled_bs], label_batch[:labeled_bs])
                if args.multiDSC:
                    outputs_soft = F.softmax(proto_seg[:labeled_bs], dim=1)
                    loss_proto_dice = dice_loss(outputs_soft, label_batch[:labeled_bs])
                else:
                    loss_proto_dice = losses.dice_loss_foreground(proto_seg[:labeled_bs], label_batch[:labeled_bs])
                loss_proto_2d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice

                loss_l = loss_cls_2d + args.proto_w * loss_proto_2d
                #
                ## unsupervised loss

                loss_cls_ce = criterion(cls_seg[labeled_bs:], label_batch[labeled_bs:])
                if args.multiDSC:
                    outputs_soft = F.softmax(cls_seg[labeled_bs:], dim=1)
                    loss_seg_dice = dice_loss(outputs_soft, label_batch[labeled_bs:])
                else:
                    loss_seg_dice = losses.dice_loss_foreground(cls_seg[labeled_bs:], label_batch[labeled_bs:])
                loss_cls_2d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice

                loss_proto_ce = criterion(proto_seg[labeled_bs:], label_batch[labeled_bs:])
                if args.multiDSC:
                    outputs_soft = F.softmax(proto_seg[labeled_bs:], dim=1)
                    loss_proto_dice = dice_loss(outputs_soft, label_batch[labeled_bs:])
                else:
                    loss_proto_dice = losses.dice_loss_foreground(proto_seg[labeled_bs:], label_batch[labeled_bs:])
                loss_proto_2d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice

                loss_u = loss_cls_2d + args.proto_w * loss_proto_2d

                consistency_weight = get_current_consistency_weight((iter_num - args.pretrainIter) // 100)
                loss = loss_l + consistency_weight * loss_u

                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss_sup', loss_l, iter_num)
                writer.add_scalar('loss/loss_ul', loss_u, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                logging.info(
                    'iteration %d : loss : %f loss_sup : %f loss_ul : %f consistency_weight : %f'
                    % (iter_num, loss.item(), loss_l.item(), loss_u.item(), consistency_weight))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update t_model
            update_ema_variables(s_model, t_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1

            if iter_num >= 2000 and iter_num % 200 == 0:
                s_model.eval()
                testing_dice,_ = test_eval(net=s_model, image_list=image_list,log=False)
                logging.info('iter %d : testing_dice : %f ' %
                             (iter_num, testing_dice))
                if bestdice < testing_dice:
                    bestIter = iter_num
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) +
                                                  '_dice_' + str(testing_dice) + '.pth')
                    torch.save(s_model.state_dict(), save_mode_path)

                    logging.info("currently best!!!! save Ourmodel to {}".format(save_mode_path))
                    bestdice = testing_dice
                s_model.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        lr_scheduler.step()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(s_model.state_dict(), save_mode_path)
    logging.info("Save Ourmodel to {}".format(save_mode_path))
    logging.info("best iter is: {}".format(bestIter))
    logging.info("highest dice is: {}".format(bestdice))
    writer.close()
