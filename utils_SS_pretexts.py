# -*- coding:utf-8 -*-
'''
Abstract.

Version 1.0  2020-07-06 14:32:04
by QiJi Refence:
TODO:
'''

import os
import time
from collections import deque
# from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
# import torch.nn as nn
import torch.nn.parallel
# import torch.distributed as dist
import torch.optim
import torch.utils.data
try:
    from apex import amp
except ImportError:
    amp = None






import datetime
def TMCNet_train(train_loader, model, criterion, optimizer, epoch, args, summary_writer=None):
    ''' One epoch training use TMCNet. '''

    #def __init__(self):
     #   self.convfusion = nn.Conv2d(8, 4, kernel_size=3)

    model.train()
    loss_hist = deque(maxlen=len(train_loader))
    lr=optimizer.param_groups[0]['lr']
    #tic = time.time()
    for i, data in enumerate(train_loader):
        #print(i)

        optimizer.zero_grad()

        input1, input2,input_sar1,input_sar2=data['image'], data['image1'],data['image_sar'],data['image_sar1']#,data['index']
        rois=data['rois']
        rois = rois.cuda(non_blocking=args.non_blocking)  # [NCHW]
        input1 = input1.cuda(non_blocking=args.non_blocking)  # [NCHW]
        input_sar1=input_sar1.cuda(non_blocking=args.non_blocking)
        input2 = input2.cuda(non_blocking=args.non_blocking)  # [NCHW]
        input_sar2 = input_sar2.cuda(non_blocking=args.non_blocking)

        if args.self_mode==12:
            q1, k1 = model(input1, input2, rois)
            loss = criterion[1](q1, k1)
        elif args.self_mode==13:
            q, k = model(input1, input2, rois)
            loss = criterion[0](q, k)
        else:

            q,k,q1,k1,m,n,m1,n1= model(input1 ,input2,input_sar1, input_sar2,rois)  #slow# ,output,output1


            loss = args.lamuda*criterion[0](q, k,m,n)+(1-args.lamuda)*criterion[1](q1,k1,m1,n1)


            #q2, k2, q3, k3 = model(input_sar1, input_sar2, rois)  # slow# ,output,output1

            #loss2 = args.lamuda * criterion[0](q2, k2) + (1 - args.lamuda) * criterion[1](q3, k3)

            #loss=(loss1+loss2)/2

        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()  #slow
        #loss.backward()
        optimizer.step()
        #print('end%d'%(i))
        #print(datetime.datetime.now())


        # Meters update and visualize
        loss_hist.append(loss.item())
        if summary_writer is not None:
            if i % args.print_freq == 0:
                step = (epoch - 1) * len(train_loader) + i
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', np.mean(loss_hist), step)
    return np.mean(loss_hist)

import torch.nn.functional as F
def Test_train(train_loader, model, criterion, optimizer, epoch, args,summary_writer=None):
    ''' One epoch training use SimCLR. '''
    model.train()
    loss_hist = deque(maxlen=len(train_loader))
    lr=optimizer.param_groups[0]['lr']
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        input1, input2=data['image'], data['image1']
        label1, label2 = data['label'], data['label1']
        if args.gpu is not None:
            input1 = input1.cuda(args.gpu, non_blocking=args.non_blocking)  # [NCHW]
            input2 = input2.cuda(args.gpu, non_blocking=args.non_blocking)  # [NCHW]
            label1 = label1.cuda(args.gpu, non_blocking=args.non_blocking)  # [NCHW]
            label2 = label2.cuda(args.gpu, non_blocking=args.non_blocking)
        else:
            input1 = input1.cuda(non_blocking=args.non_blocking)  # [NCHW]
            input2 = input2.cuda(non_blocking=args.non_blocking)  #
            label1 = label1.cuda(non_blocking=args.non_blocking)  # [NCHW]
            label2 = label2.cuda(non_blocking=args.non_blocking)


        # [NCHW]
        q, k,output1,output2 = model(input1, input2)  # ,output,output1
        '''
        label1 = F.interpolate(
            label1.unsqueeze(1),size=output1.size()[2:4],mode='nearest').squeeze(1)
        label2 = F.interpolate(
            label2.unsqueeze(1), size=output2.size()[2:4], mode='nearest').squeeze(1)
        '''
        loss = criterion[0](q, k)+criterion[1](output1,label1)+criterion[1](output2,label2)
        del q,k,output1,output2
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # Meters update and visualize
        loss_hist.append(loss.item())
        if summary_writer is not None:
            if i % args.print_freq == 0:
                step = (epoch - 1) * len(train_loader) + i
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', np.mean(loss_hist), step)
    #if args.vis and (epoch == args.sepoch or epoch == args.mepoch) and (
            #not args.mp_distributed or args.gpu == 0):
        #log_data_for_ss(input1, input2, epoch, args)
    return np.mean(loss_hist)
def get_erase_mask(bs, opts, erase_shape=[16, 16], erase_count=16):
    #Random block
    H, W = opts.input_size
    masks = torch.ones((bs, opts.n_channels, H, W))
    for n in range(bs):
        for _ in range(erase_count):
            row = np.random.randint(0, H - erase_shape[0] - 1)
            col = np.random.randint(0, W - erase_shape[1] - 1)
            masks[n, :, row: row+erase_shape[0], col: col+erase_shape[1]] = 0
    return masks
def get_central_mask(bs, opts, erase_ratio=1/2):
    #Central region
    H, W = opts.input_size
    masks = torch.ones((bs, opts.n_channels, H, W))
    eH, eW = int(H*erase_ratio), int(W*erase_ratio)
    row_st = (H - eH) // 2
    col_st = (W - eW) // 2
    masks[:, :, row_st: row_st+eH, col_st: col_st+eW] = 0
    return masks






if __name__ == '__main__':
    # main()
    pass

