# Train_Stage2_K.py Fine tune model with MOM on KITTI only
# Copyright (C) 2021  Juan Luis Gonzalez Bello (juanluisgb@kaist.ac.kr)
# This software is not for commercial use
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import argparse
import datetime
import time
import numpy as np

import Datasets
import models

dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)

parser = argparse.ArgumentParser(description='FAL_net in pytorch',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data', metavar='DIR', default='C:\\Users\\Kaist\\Desktop', help='path to dataset')
parser.add_argument('-n0', '--dataName0', metavar='Data Set Name 0', default='Kitti', choices=dataset_names)
parser.add_argument('-train_split', '--train_split', default='eigen_train_split')
parser.add_argument('-vdn', '--vdataName', metavar='Val data set Name', default='Kitti2015', choices=dataset_names)
parser.add_argument('-relbase_test', '--rel_baset', default=1, help='Relative baseline of testing dataset')
parser.add_argument('-maxd', '--max_disp', default=300)
parser.add_argument('-mind', '--min_disp', default=2)
# -----------------------------------------------------------------------------
parser.add_argument('-gpu_no', '--gpu_no', default='0', help='Select your GPU ID, if you have multiple GPU.')
parser.add_argument('-mm', '--m_model', metavar='Mono Model', default='FAL_netB', choices=model_names)
parser.add_argument('-no_levels', '--no_levels', default=49, help='Number of quantization levels in MED')
parser.add_argument('-perc', '--a_p', default=0.01, help='Perceptual loss weight')
parser.add_argument('-smooth', '--a_sm', default=0.4 * 2 / 512, help='Smoothness loss weight')
parser.add_argument('-mirror_loss', '--a_mr', default=1, help='Mirror loss weight')
# ------------------------------------------------------------------------------
parser.add_argument('-w', '--workers', metavar='Workers', default=4)
parser.add_argument('-b', '--batch_size', metavar='Batch Size', default=4)
parser.add_argument('-ch', '--crop_height', metavar='Batch crop H Size', default=192)
parser.add_argument('-cw', '--crop_width', metavar='Batch crop W Size', default=640)
parser.add_argument('-tbs', '--tbatch_size', metavar='Val Batch Size', default=1)
parser.add_argument('-op', '--optimizer', metavar='Optimizer', default='adam')
parser.add_argument('--lr', metavar='learning Rate', default=0.00005)
parser.add_argument('--beta', metavar='BETA', type=float, help='Beta parameter for adam', default=0.999)
parser.add_argument('--momentum', default=0.5, type=float, metavar='Momentum', help='Momentum for Optimizer')
parser.add_argument('--milestones', default=[5, 10], metavar='N', nargs='*',
                    help='epochs at which learning rate is divided by 2')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0.0, type=float, metavar='B', help='bias decay')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('--sparse', default=True, action='store_true',
                    help='Depth GT is sparse, automatically seleted when choosing a KITTIdataset')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--fix_model', dest='fix_model',
                    default='Kitti_stage1\\10-15-21_11\\FAL_netB,e50es,b8,lr0.0001/checkpoint.pth.tar')
parser.add_argument('--pretrained', dest='pretrained',
                    default='Kitti_stage1\\10-15-21_11\\FAL_netB,e50es,b8,lr0.0001/checkpoint.pth.tar',
                    help='directory of run')


def display_config(save_path):
    settings = ''
    settings = settings + '############################################################\n'
    settings = settings + '# FAL_net - Pytorch implementation                         #\n'
    settings = settings + '# by Juan Luis Gonzalez   juanluisgb@kaist.ac.kr           #\n'
    settings = settings + '############################################################\n'
    settings = settings + '-------YOUR TRAINING SETTINGS---------\n'
    for arg in vars(args):
        settings = settings + "%15s: %s\n" % (str(arg), str(getattr(args, arg)))
    print(settings)
    # Save config in txt file
    with open(os.path.join(save_path, 'settings.txt'), 'w+') as f:
        f.write(settings)


def main():
    print('-------Training on gpu ' + args.gpu_no + '-------')
    best_rmse = -1

    save_path = '{},e{}es{},b{},lr{}'.format(
        args.m_model,
        args.epochs,
        str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr,
    )
    timestamp = datetime.datetime.now().strftime("%m-%d-%H_%M")
    save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.dataName0 + "_stage2", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    display_config(save_path)
    print('=> will save everything to {}'.format(save_path))

    # Set output writters for showing up progress on tensorboardX
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    # Set up data augmentations
    co_transform = data_transforms.Compose([
        data_transforms.RandomResizeCrop((args.crop_height, args.crop_width), down=0.75, up=1.5),
        data_transforms.RandomHorizontalFlip(),
        data_transforms.RandomGamma(min=0.8, max=1.2),
        data_transforms.RandomBrightness(min=0.5, max=2.0),
        data_transforms.RandomCBrightness(min=0.8, max=1.2),
    ])

    input_transform = transforms.Compose([
        data_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),  # (input - mean) / std
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    target_transform = transforms.Compose([
        data_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0], std=[1]),
    ])

    # Torch Data Set List
    input_path = os.path.join(args.data, args.dataName0)
    [train_dataset0, _] = Datasets.__dict__[args.dataName0](split=1, # all for training
                                                            root=input_path,
                                                            transform=input_transform,
                                                            target_transform=target_transform,
                                                            co_transform=co_transform,
                                                            max_pix=args.max_disp,
                                                            train_split=args.train_split,
                                                            fix=True)
    input_path = os.path.join(args.data, args.vdataName)
    [_, test_dataset] = Datasets.__dict__[args.vdataName](split=0,  # all to be tested
                                                          root=input_path,
                                                          disp=True,
                                                          of=False,
                                                          shuffle_test=False,
                                                          transform=input_transform,
                                                          target_transform=target_transform,
                                                          co_transform=co_transform)

    # Torch Data Loaders
    train_loader0 = torch.utils.data.DataLoader(train_dataset0, batch_size=args.batch_size,
                                                num_workers=args.workers,
                                                pin_memory=False, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.tbatch_size, num_workers=args.workers,
                                             pin_memory=False, shuffle=False)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.m_model = network_data['m_model']
        print("=> using pre-trained model '{}'".format(args.m_model))
    else:
        network_data = None
        print("=> creating m model '{}'".format(args.m_model))

    m_model = models.__dict__[args.m_model](network_data, no_levels=args.no_levels).cuda()
    m_model = torch.nn.DataParallel(m_model, device_ids=[0]).cuda()
    print("=> Number of parameters m-model '{}'".format(utils.get_n_params(m_model)))

    # create fix model
    network_data = torch.load(args.fix_model)
    fix_model_name = network_data['m_model']
    print("=> using pre-trained model '{}'".format(fix_model_name))
    fix_model = models.__dict__[fix_model_name](network_data, no_levels=args.no_levels).cuda()
    fix_model = torch.nn.DataParallel(fix_model, device_ids=[0]).cuda()
    print("=> Number of parameters m-model '{}'".format(utils.get_n_params(fix_model)))
    fix_model.eval()

    # Optimizer Settings
    print('Setting {} Optimizer'.format(args.optimizer))
    param_groups = [{'params': m_model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': m_model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.optimizer == 'adam':
        g_optimizer = torch.optim.Adam(params=param_groups, lr=args.lr, betas=(args.momentum, args.beta))
    g_scheduler = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=args.milestones, gamma=0.5)

    for epoch in range(args.start_epoch):
        g_scheduler.step()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss = train(train_loader0, m_model, fix_model, g_optimizer, epoch)
        train_writer.add_scalar('train_loss', train_loss, epoch)

        # evaluate on validation set, RMSE is from stereoscopic view synthesis task
        rmse = validate(val_loader, m_model, epoch, output_writers)
        test_writer.add_scalar('mean RMSE', rmse, epoch)

        # Apply LR schedule (after optimizer.step() has been called for recent pyTorch versions)
        g_scheduler.step()

        if best_rmse < 0:
            best_rmse = rmse
        is_best = rmse < best_rmse
        best_rmse = min(rmse, best_rmse)
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'm_model': args.m_model,
            'state_dict': m_model.module.state_dict(),
            'best_rmse': best_rmse,
        }, is_best, save_path)


def train(train_loader, m_model, fix_model, g_optimizer, epoch):
    global args
    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    rec_losses = utils.AverageMeter()
    losses = utils.AverageMeter()

    # switch to train mode
    m_model.train()

    end = time.time()
    for i, input_data0 in enumerate(train_loader):
        # Read training data
        left_view = input_data0[0][0].cuda()
        right_view = input_data0[0][1].cuda()
        max_disp = input_data0[1].unsqueeze(1).unsqueeze(1).type(left_view.type())
        min_disp = max_disp * args.min_disp / args.max_disp
        B, C, H, W = left_view.shape

        # measure data loading time
        data_time.update(time.time() - end)

        # Reset gradients
        g_optimizer.zero_grad()

        # Flip Grid (differentiable)
        i_tetha = torch.autograd.Variable(torch.zeros(B, 2, 3)).cuda()
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        i_grid = F.affine_grid(i_tetha, [B, C, H, W], align_corners=True)
        flip_grid = i_grid.clone()
        flip_grid[:, :, :, 0] = -flip_grid[:, :, :, 0]

        # Get mirrored disparity from fixed falnet model
        if args.a_mr > 0:
            with torch.no_grad():
                disp = fix_model(torch.cat(
                    (F.grid_sample(left_view, flip_grid, align_corners=True), right_view), 0),
                    torch.cat((min_disp, min_disp), 0),
                    torch.cat((max_disp, max_disp), 0),
                    ret_disp=True, ret_pan=False, ret_subocc=False)
                mldisp = F.grid_sample(disp[0:B,:,:,:], flip_grid, align_corners=True).detach()
                mrdisp = disp[B::, :, :, :].detach()

        ###### LEFT disp
        pan, disp, mask0, mask1 = m_model(torch.cat((left_view,
                                                     F.grid_sample(right_view, flip_grid, align_corners=True)), 0),
                                          torch.cat((min_disp, min_disp), 0),
                                          torch.cat((max_disp, max_disp), 0),
                                          ret_disp=True, ret_pan=True, ret_subocc=True)
        rpan = pan[0:B, :, :, :]
        lpan = pan[B::, :, :, :]
        ldisp = disp[0:B, :, :, :]
        rdisp = disp[B::, :, :, :]

        lmask = mask0[0:B, :, :, :]
        rmask = mask0[B::, :, :, :]
        rlmask = mask1[0:B, :, :, :]
        lrmask = mask1[B::, :, :, :]

        # Unflip right view stuff
        lpan = F.grid_sample(lpan, flip_grid, align_corners=True)
        rdisp = F.grid_sample(rdisp, flip_grid, align_corners=True)
        rmask = F.grid_sample(rmask, flip_grid, align_corners=True)
        lrmask = F.grid_sample(lrmask, flip_grid, align_corners=True)

        # Compute rec loss
        if args.a_p > 0:
            vgg_right = vgg(right_view)
            vgg_left = vgg(left_view)
        else:
            vgg_right = None
            vgg_left = None
        # Obtain final occlusion masks
        O_L = lmask * lrmask
        O_L[:, :, :, 0:int(0.20 * W)] = 1
        O_R = rmask * rlmask
        O_R[:, :, :, int(0.80 * W)::] = 1
        if args.a_mr == 0: # no mirror loss, then it is just more training
            O_L = 1
            O_R = 1
        # Over 2 as measured twice for left and right
        rec_loss = (rec_loss_fnc(O_R, rpan, right_view, vgg_right, args.a_p) + \
                  rec_loss_fnc(O_L, lpan, left_view, vgg_left, args.a_p)) / 2
        rec_losses.update(rec_loss.detach().cpu(), args.batch_size)

        # Compute smooth loss
        sm_loss = 0
        if args.a_sm > 0:
            # Here we ignore the 20% left dis-occluded region, as there is no suppervision for it due to parralax
            sm_loss = (smoothness(left_view[:, :, :, int(0.20 * W)::], ldisp[:, :, :, int(0.20 * W)::], gamma=2) +
                      smoothness(right_view[:, :, :, 0:int(0.80 * W)], rdisp[:, :, :, 0:int(0.80 * W)], gamma=2)) / 2

        # Compute mirror loss
        mirror_loss = 0
        if args.a_mr > 0:
            # Normalize error ~ between 0-1, by diving over the max disparity value
            nmaxl = 1 / F.max_pool2d(mldisp, kernel_size=(H, W))
            nmaxr = 1 / F.max_pool2d(mrdisp, kernel_size=(H, W))
            mirror_loss = (torch.mean(nmaxl * (1 - O_L)[:, :, :, int(0.20 * W)::] *
                                      torch.abs(ldisp - mldisp)[:, :, :, int(0.20 * W)::]) +
                           torch.mean(nmaxr * (1 - O_R)[:, :, :, 0:int(0.80 * W)] *
                                      torch.abs(rdisp - mrdisp)[:, :, :, 0:int(0.80 * W)])) / 2

        # compute gradient and do optimization step
        loss = rec_loss + args.a_sm * sm_loss + args.a_mr * mirror_loss
        losses.update(loss.detach().cpu(), args.batch_size)
        loss.backward()
        g_optimizer.step()
        g_optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] Time {3}  Data {4}  Loss {5} RecLoss {6}'
                  .format(epoch, i, epoch_size, batch_time, data_time, losses, rec_losses))

        # End training epoch earlier if args.epoch_size != 0
        if i >= epoch_size:
            break

    return losses.avg


def validate(val_loader, m_model, epoch, output_writers):
    global args

    test_time = utils.AverageMeter()
    RMSES = utils.AverageMeter()
    EPEs = utils.AverageMeter()
    kitti_erros = utils.multiAverageMeter(utils.kitti_error_names)

    # switch to evaluate mode
    m_model.eval()

    # Disable gradients to save memory
    with torch.no_grad():
        for i, input_data in enumerate(val_loader):
            input_left = input_data[0][0].cuda()
            input_right = input_data[0][1].cuda()
            target = input_data[1][0].cuda()
            max_disp = torch.Tensor([args.max_disp * args.rel_baset]).unsqueeze(1).unsqueeze(1).type(input_left.type())
            B, _, H, W = input_left.shape

            # Prepare input data
            end = time.time()
            min_disp = max_disp * args.min_disp / args.max_disp
            p_im, disp, maskL, maskRL = m_model(input_left, min_disp, max_disp,
                                                ret_disp=True, ret_pan=True, ret_subocc=True)
            test_time.update(time.time() - end)

            # Measure RMSE
            rmse = utils.get_rmse(p_im, input_right)
            RMSES.update(rmse)

            # record EPE
            flow2_EPE = realEPE(disp, target, sparse=args.sparse)
            EPEs.update(flow2_EPE.detach(), target.size(0))

            # Record kitti metrics
            target_depth, pred_depth = utils.disps_to_depths_kitti2015(target.detach().squeeze(1).cpu().numpy(),
                                                                       disp.detach().squeeze(1).cpu().numpy())
            kitti_erros.update(utils.compute_kitti_errors(target_depth[0], pred_depth[0]), target.size(0))

            denormalize = np.array([0.411, 0.432, 0.45])
            denormalize = denormalize[:, np.newaxis, np.newaxis]

            if i < len(output_writers):  # log first output of first batches
                if epoch == 0:
                    output_writers[i].add_image('Input left', input_left[0].cpu().numpy() + denormalize, 0)

                # Plot disp
                output_writers[i].add_image('Left disparity', utils.disp2rgb(disp[0].cpu().numpy(), None), epoch)

                # Plot left subocclsion mask
                output_writers[i].add_image('Left sub-occ', utils.disp2rgb(maskL[0].cpu().numpy(), None), epoch)

                # Plot right-from-left subocclsion mask
                output_writers[i].add_image('RightL sub-occ', utils.disp2rgb(maskRL[0].cpu().numpy(), None), epoch)

                # Plot synthetic right (or panned) view output
                p_im = p_im[0].detach().cpu().numpy() + denormalize
                p_im[p_im > 1] = 1
                p_im[p_im < 0] = 0
                output_writers[i].add_image('Output Pan', p_im, epoch)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t Time {2}\t RMSE {3}'.format(i, len(val_loader), test_time, RMSES))

    print('* RMSE {0}'.format(RMSES.avg))
    print(' * EPE {:.3f}'.format(EPEs.avg))
    print(kitti_erros)
    return RMSES.avg


if __name__ == '__main__':
    import os

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no

    import torch
    import torch.utils.data
    import torchvision.transforms as transforms
    from tensorboardX import SummaryWriter
    import torch.nn.functional as F

    # Usefull tensorboard call
    # tensorboard --logdir=C:ProjectDir/NeurIPS2020_FAL_net/Kitti --port=6012

    import myUtils as utils
    import data_transforms
    from loss_functions import rec_loss_fnc, realEPE, smoothness, vgg

    args = parser.parse_args()

    main()
