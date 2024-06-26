#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.backends.cudnn as cudnn
from pathlib import Path
#from apex.parallel import DistributedDataParallel
import torch.distributed as dist
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
#torch.set_num_threads(8)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from config.get_opt import get_opt
from utils import data_for_seg as dataset1#
from utils.tools import net_predict1
from dl_tools.basictools.dltrain import compute_class_iou
from dl_tools.basictools.dldata import get_label_info
from dl_tools.basictools.dlimage import colour_code_label
import utils.contrast_loss as contrast_loss
import utils_SS_pretexts
import builtins
from models.deeplab_utils.sync_batchnorm import convert_model
from models import build_model
from utils import get_self_dataset
import models.v3p as v3p
import models.v3p_encoder_ft as v3p_encoder_ft
import models.v3p_resnet_ft as v3p_resnet_ft
import cv2
import tifffile
from evalute import hist, get_scores
#from torch.utils.tensorboard import SummaryWriter
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
torch.autograd.set_detect_anomaly(True)

Type2Band = {'RGB': 3, 'NIR': 1, 'SAR': 1,'RGBIR':4}
Self_Mode_Name = {1: 'TMCNet',11:'TMCNet_noStyle',12:'TMCNet_noGlobal',13:'TMCNet_noLocal'}

try:
    from apex import amp
except ImportError:
    amp = None
import datetime
import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser(description='PyTorch TMCNet for Segmentation Training')
parser.add_argument('--ex_mode', default=1, type=int, help="0-just_ss,1-ss_ft,2-just_ft")
parser.add_argument('--self_mode', default=1, type=int, help="Self-supervised method")
parser.add_argument('--root', default=r'', help='dataset root')
parser.add_argument('--dtype', default='RGB', help='dataset dtype: RGB')

parser.add_argument('--patch_size', default=48, type=int, help="patch_size_for_TMCNet_self-supervised")
parser.add_argument('--patch_num', default=4, type=int, help="patch_num_for_TMCNet_self-supervised")
parser.add_argument('--self_batch_size', default=256, type=int, help="")
parser.add_argument('--ft_batch_size', default=16, type=int, help="")
parser.add_argument('--self_max_epoch', default=40, type=int, help="")
parser.add_argument('--ft_max_epoch', default=15, type=int, help="")

parser.add_argument('--class_num', default=None, type=int, metavar='N',
                    help='default: None(from class_dict.txt)')
parser.add_argument('--lr_ft', default=0.0001, type=float)
parser.add_argument('--lr_self', default=0.001, type=float)
parser.add_argument('--mdl_path', default='', type=str, help='model_path')
parser.add_argument('--self_data_name', default='train', type=str, help='self_data_txt_name')
parser.add_argument('--ft_train_name', default='trainR1', type=str, help='ft_train_data_txt_name')
parser.add_argument('--ft_val_name', default='val', type=str, help='self_val_data_txt_name')
parser.add_argument('--mdl_name', default=None, type=str, help='ex_name')


parser.add_argument('--amp_opt_level', default="O0", type=str)#"O0",O1
parser.add_argument('--sync', default=True, type=str2bool)

parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10000', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--use_mp', default=True, type=str2bool, help="Use mp-distributed")

parser.add_argument('--log_path', default='./ex', type=str, help='')
parser.add_argument('--load_self_parameters', default=2, type=int, help='(0-backbone,1-encoder,2-all)')
parser.add_argument('--write_label', default=2, type=int, help='(0-no write,1-RGB_result,2-single_channel_result)')
parser.add_argument('--lamuda', default=0.5, type=float, help='LAMUDA')

def main1(args):
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)

    args.distributed = args.world_size > 1 or args.use_mp or args.ngpus_per_node > 1
    if args.ex_mode==0 or args.ex_mode==1:
        if args.use_mp:
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ))#join=True?
        else:
            main_worker(args.gpu, args)
    if args.ex_mode==1 or args.ex_mode==2 or args.ex_mode==3:
        'fine-tuning just use one GPU!'
        if args.class_num==None:
            class_names, label_values = get_label_info(args.class_dict_road)
            args.class_num = len(class_names)
            print('class num:'+str(args.class_num))

        model=build_ft_model(args)
        if args.use_gpu:
            model.cuda(0)#ft_at_one_GPU
        if args.ex_mode!=3:
            train_seg1(model,args)
            val1(model, args)
        else:
            val(model, args)





def main_worker(gpu, args):
    args.gpu = gpu
    args.input_size = (224, 224)
    # tensorboard
    if not args.use_mp or args.gpu == 0:
        summary_writer = None#SummaryWriter()
    else:
        summary_writer = None
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.use_mp and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    #total_bs = args.self_batch_size
    if args.use_mp:
        torch.cuda.set_device(args.gpu)
        args.rank = args.rank * args.ngpus_per_node + gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
        print('total_batch_size:')
        print(args.self_batch_size*dist.get_world_size())
        args.lr_self=dist.get_world_size()*args.lr_self

    torch.backends.cudnn.benchmark = True

    model=build_model(args)
    if args.self_mode==1 or args.self_mode==11 or args.self_mode==12 or args.self_mode==13:
        train_epoch = utils_SS_pretexts.TMCNet_train
        criterion = [contrast_loss.NTXentLoss(args.self_batch_size, args.gpu, 0.5, cos_sim=True).cuda(args.gpu),contrast_loss.NTXentLoss(args.self_batch_size*args.patch_num, args.gpu, 0.5, cos_sim=True).cuda(args.gpu)]


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr_self, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    if args.amp_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    model = place_model(model, args)
    model.cuda()
    # Data loading code
    train_dataset = get_self_dataset(args)

    train_sampler = None
    if args.use_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, args.self_batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if not args.use_mp or args.gpu == 0:
        print('```\n' + '-' * 50 + '\n```')

    epoch_best_loss = 1000
    print('%s\nBegain training' % args.mdl_name)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.self_max_epoch, eta_min=4e-08)

    for epoch in range(args.self_max_epoch):
        print(datetime.datetime.now())
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        LR = optimizer.param_groups[0]['lr']
        #LR = tools.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        epoch_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, args,summary_writer)

        print('\tEpoch: [%d | %d], lr:%e, loss: %.6f\n' % (epoch, args.self_max_epoch, LR, epoch_loss),
              '\t*****************')
        scheduler.step()

        if not args.use_mp or args.gpu == 0:
            if epoch_loss < epoch_best_loss:
                save_ckpt(model, args.ckpt, 'self_bestloss_%s' % (args.mdl_name))

        epoch_best_loss = min(epoch_loss, epoch_best_loss)

    if not args.use_mp or args.gpu == 0:
        save_ckpt(model, args.ckpt, '%s_epoch%d' % ( args.mdl_name, epoch))
        print('Finish! Stop at epoch %d (max epoch=%d).' % (epoch, args.self_max_epoch))
        print(datetime.datetime.now())
def save_ckpt(model, ckpt='cp', filename='checkpoint'):

    # import shutil
    filepath = ckpt+'/'+filename+'.pth'
    if type(model) is torch.nn.DataParallel or type(model) is torch.nn.parallel.DistributedDataParallel:
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()
    #torch.save(model_dict, filepath,_use_new_zipfile_serialization=False)#torch_version>=1.6
    torch.save(model_dict, filepath)


def place_model(model, args):
    if args.use_mp:
        if args.sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#,find_unused_parameters=True
    else:
        if args.sync:
            model = convert_model(model)
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.ngpus_per_node)])
    return model


def build_ft_model(args):
    if args.load_self_parameters==1:#load_encoder
        model=v3p_encoder_ft.build_model(num_classes=args.class_num, in_channels=args.n_channels, pretrained=False,
                             arch=args.arch)  # num_classes1=50,
    elif args.load_self_parameters==0:#load_resnet
        model=v3p_resnet_ft.build_model(num_classes=args.class_num, in_channels=args.n_channels, pretrained=False,
                             arch=args.arch)  # num_classes1=50,
    else:
        model = v3p.build_model(num_classes=args.class_num, in_channels=args.n_channels, pretrained=False,
                                          arch=args.arch)  # num_classes1=50,
    return model


def train_seg1(model, opt):
    if opt.mdl_path == None:
        opt.mdl_path = opt.ckpt + '/self_bestloss_%s.pth' % (args.mdl_name)
    ckpt = opt.mdl_path
    opt.input_size=(256,256)

    if ckpt==None:
        print('NO model checkpoint to load')
    else:
        print(ckpt)
        if not os.path.isfile(ckpt):
            raise ValueError('NO model checkpoint.')

        pred_dict = torch.load(ckpt)
        model_dict = model.state_dict()
        print('load_from_self_model:')
        for k, v in pred_dict.items():
            if k in model_dict:
                print(k)
        pred_dict = {k: v for k, v in pred_dict.items() if k in model_dict}
        model_dict.update(pred_dict)
        model.load_state_dict(model_dict)


    model.cuda(0)
    # 1. Get data
    train_dataset = dataset1.Train_Dataset(opt.dataset_dir, opt.ft_train_name, opt)
    train_dataloader = DataLoader(
        train_dataset, opt.ft_batch_size,
        shuffle=True, num_workers=opt.num_workers, pin_memory=opt.pin_memory, drop_last=True)  # opt.batch_size
    val_dataset = dataset1.MyDataset_1(opt.dataset_dir, opt.ft_val_name, opt)
    val_dataset.crop_mode = 'random'  # quick val
    val_dataloader = DataLoader(val_dataset, opt.ft_batch_size, pin_memory=opt.pin_memory,
                                num_workers=opt.num_workers)

    basic_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda(0)

    # 3.  Customization LR and Optimizer
    LR = opt.lr_ft
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=opt.weight_decay, betas=(0.9, 0.99))  # ,params

    # 4. Meters
    # loss_meter = meter.AverageValueMeter()
    ckpt_save = False  # control wether to save cur epoch ckpt

    # 5. Begain Train
    print('\n--------------Begain Train1!')
    # plotloss = plot.PlotLine(model.model_name)

    model.train()  # 1)transform model mode
    parameters_tot=0
    for nom,param in model.named_parameters():
        parameters_tot+=torch.prod(torch.tensor(param.data.shape))
    print("Number pf model parameters {}".format(parameters_tot))

    best_OA = 0
    best_OA_epoch = 0
    for epoch in range(opt.ft_max_epoch):
        epoch_loss = 0
        torch.cuda.empty_cache()
        for i, data in enumerate(train_dataloader):
            input,input_sar, target = data['image'],data['image_sar'], data['label']  # #planB

            if opt.use_gpu:
                input,input_sar, target = input.cuda(0,non_blocking=opt.non_blocking),input_sar.cuda(0,non_blocking=opt.non_blocking), target.cuda(0,non_blocking=opt.non_blocking)

            optimizer.zero_grad()
            output = model(input,input_sar)
            torch.cuda.empty_cache()

            total_loss = basic_criterion(output, target)
            total_loss.backward()
            epoch_loss = epoch_loss + total_loss.item()

            optimizer.step()

        log_str = ("epoch:%d, lr:%f" % (epoch + opt.start_epoch + 1, LR))
        oa = quick_val2(model, val_dataloader, opt)

        log_str += ("train_loss:%.5f, val_OA:%.5f" % (epoch_loss / len(train_dataloader), oa))
        print(log_str)
        if epoch == (opt.ft_max_epoch - 1):
            torch.save(model.state_dict(),
                       opt.ckpt + '/ft_' + opt.mdl_name + '_' + opt.ft_train_name + '_last.pth')

        if oa > best_OA:
            best_OA_epoch = epoch
            best_OA = oa
            torch.save(model.state_dict(),
                       opt.ckpt + '/ft_' + opt.mdl_name + '_' + opt.ft_train_name + '_bestOA.pth')
        # 6)Update learning rate per epoc
        if opt.lr_decay is not None:  # and epoch>10
            LR = LR * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
    print('\n-----bestOA_epoch:%d' % (best_OA_epoch))


def compute_global_accuracy(pred, label):
    '''
    Compute the average segmentation accuracy across all classes,
    Input [HW] or [HWC] label
    '''
    count_mat = pred == label
    return np.sum(count_mat) / np.prod(count_mat.shape)


def quick_val2(model, dataloader, opt):

    model.eval()
    with torch.no_grad():
        # iou = meter.AverageValueMeter()
        OA = 0
        for ii, data in enumerate(dataloader):

            input,input_sar, label = data['image'],data['image_sar'], data['label']

            label = np.squeeze(label.numpy())  # [NHW] -> [HW](n=1)
            if opt.use_gpu:
                input = input.cuda(0,non_blocking=True)
                input_sar = input_sar.cuda(0, non_blocking=True)

            output = model(input,input_sar)  # [NCHW] -> [CHW]
            output = torch.argmax(output, 1)
            predict = output.cpu().detach().numpy()

            OA += compute_global_accuracy(predict, label)
            # class_iou += compute_class_iou(predict, label, opt.class_num)

        # class_iou /= (ii + 1)
        OA /= (ii + 1)

        # score = get_scores(confusion_matrix)
    model.train()
    return OA  # mean_iou


def quick_val1(model, dataloader, opt):
    model.eval()
    with torch.no_grad():
        # iou = meter.AverageValueMeter()
        class_iou = np.zeros(opt.class_num)
        confusion_matrix = np.zeros([opt.class_num, opt.class_num], dtype=np.int64)
        for ii, data in enumerate(dataloader):
            input,input_sar, label = data['image'],data['image_sar'], data['label']

            label = np.squeeze(label.numpy())  # [NHW] -> [HW](n=1)
            if opt.use_gpu:
                input = input.cuda(0,non_blocking=opt.non_blocking)
                input_sar = input_sar.cuda(0, non_blocking=True)

            output = model(input,input_sar)  # [NCHW] -> [CHW]

            predict = np.argmax(output.cpu().detach().numpy(), 1)  # [CHW] -> [HW]
            confusion_matrix += hist(predict, label, opt.class_num)
            class_iou += compute_class_iou(predict, label, opt.class_num)
        class_iou /= ii + 1
        score = get_scores(confusion_matrix)
    model.train()
    return class_iou, np.mean(class_iou), score  # mean_iou




def val1(model, opt):

    print('------------begain val:')
    # model.module.encoder_q.decoder.conv4 = nn.Conv2d(256, 10, kernel_size=1)#=outconv(64,10)
    ckpt = opt.ckpt + '/ft_' + opt.mdl_name + '_' + opt.ft_train_name + '_bestOA.pth'
    print(ckpt)
    if not os.path.isfile(ckpt):
        raise ValueError('NO model checkpoint.')
    model_dict = torch.load(ckpt)
    model.load_state_dict(model_dict)

    # 1. Get data
    val_dataset = dataset1.MyDataset_1(opt.dataset_dir, opt.ft_val_name, opt)
    val_dataloader = DataLoader(val_dataset, 1, num_workers=1)

    # 3. Begain val
    model.eval()
    class_iou = np.zeros(opt.class_num)
    confusion_matrix = np.zeros([opt.class_num, opt.class_num], dtype=np.int64)
    class_names, label_values = get_label_info(opt.class_dict_road)
    if opt.write_label != 0:
        if not os.path.exists(opt.outputimage):
            os.mkdir(opt.outputimage)
        out_path = opt.outputimage + '/%s_pre%s' % (opt.mdl_name, opt.ft_val_name)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
    for ii, data in enumerate(val_dataloader):
        input,input_sar, label, name = data['image'],data['image_sar'], data['label'], data['name'][0]  # [NLCHW] [NHW]
        # print(image)
        label = np.squeeze(label.numpy())  # [NHW] -> [HW](N=1)
        crop_info = np.array(data['crop_info']) if 'crop_info' in data.keys() else None
        # Do predict:

        predict = net_predict1(model, input,input_sar, opt, crop_info)
        # output = model(image)[0][0]# [NCHW] -> [CHW]
        # predict = np.argmax(output.cpu().detach().numpy(), 0)  # [CHW] -> [HW]
        predict = predict[:label.shape[0], :label.shape[1]]
        class_iou += compute_class_iou(predict, label, opt.class_num)
        confusion_matrix += hist(predict, label, opt.class_num)
        # Visualize result:

        if opt.write_label == 1:
            colour_code_label(predict, label_values, save_path=out_path + '/' + name[0:-3] + '_predict.tif')
            colour_code_label(label, label_values, save_path=out_path + '/' + name[0:-3] + '_label.tif')
        elif opt.write_label == 2:
            tifffile.imwrite(out_path + '/' + name[0:-3] + '.tif', predict)


    class_iou /= ii + 1
    score = get_scores(confusion_matrix)
    print('\n------Class Acc\n')
    print(score['Class Acc'])
    print('\n------recall\n')
    print(score['recall'])
    print('\n------F1_score\n')
    print(score['F1_score'])
    print('\n------Hist\n')
    print(score['Hist'])
    print('\n------kappa')
    print(score['Kappa'])
    print('-----Overall Acc')
    print(score['Overall Acc'])
    print('-----Mean Acc\n')
    print(score['Mean Acc'])
    # overall_acc /= ii + 1
    print('Class   \tIoU')
    for i in range(opt.class_num):
        print('class %s\t%.10f' % (class_names[i], class_iou[i]))
    print('Mean IoU\t%.10f' % (np.mean(class_iou)))


def val(model, opt):
    print('------------begain val:')
    if opt.mdl_path == None:
        opt.mdl_path =opt.ckpt + '/ft_' + opt.mdl_name + '_' + opt.ft_train_name + '_bestOA.pth'
    ckpt = opt.mdl_path
    print(ckpt)
    if not os.path.isfile(ckpt):
        raise ValueError('NO model checkpoint.')
    model_dict = torch.load(ckpt)
    model.load_state_dict(model_dict)

    # 1. Get data
    val_dataset = dataset1.MyDataset_1(opt.dataset_dir, opt.ft_val_name, opt)
    val_dataloader = DataLoader(val_dataset, 1, num_workers=1)

    # 3. Begain val
    model.eval()
    class_iou = np.zeros(opt.class_num)  #
    confusion_matrix = np.zeros([opt.class_num, opt.class_num], dtype=np.int64)
    class_names, label_values = get_label_info(opt.class_dict_road)
    if opt.write_label != 0:
        if not os.path.exists(opt.outputimage):
            os.mkdir(opt.outputimage)
        out_path = opt.outputimage + '/%s_pre%s' % (opt.mdl_name, opt.ft_val_name)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
    for ii, data in enumerate(val_dataloader):
        input,input_sar, label, name = data['image'],data['image_sar'], data['label'], data['name'][0]  # [NLCHW] [NHW]
        # print(image)
        label = np.squeeze(label.numpy())  # [NHW] -> [HW](N=1)
        crop_info = np.array(data['crop_info']) if 'crop_info' in data.keys() else None
        # Do predict:

        predict = net_predict1(model, input,input_sar, opt, crop_info)
        # output = model(image)[0][0]# [NCHW] -> [CHW]
        # predict = np.argmax(output.cpu().detach().numpy(), 0)  # [CHW] -> [HW]
        predict = predict[:label.shape[0], :label.shape[1]]
        class_iou += compute_class_iou(predict, label, opt.class_num)
        confusion_matrix += hist(predict, label, opt.class_num)
        # Visualize result:

        if opt.write_label == 1:
            colour_code_label(predict, label_values, save_path=out_path + '/' + name[0:-3] + '_predict.tif')
            colour_code_label(label, label_values, save_path=out_path + '/' + name[0:-3] + '_label.tif')
        elif opt.write_label == 2:
            tifffile.imwrite(out_path + '/' + name[0:-3] + '.tif', predict)
            # tifffile.imwrite(out_path + '/' + name[0:-3] + '_lab.tif', label)

    class_iou /= ii + 1
    score = get_scores(confusion_matrix)
    print('\n------Class Acc\n')
    print(score['Class Acc'])
    print('\n------recall\n')
    print(score['recall'])
    print('\n------F1_score\n')
    print(score['F1_score'])
    print('\n------Hist\n')
    print(score['Hist'])
    print('\n------kappa')
    print(score['Kappa'])
    print('-----Overall Acc')
    print(score['Overall Acc'])
    print('-----Mean Acc\n')
    print(score['Mean Acc'])
    # overall_acc /= ii + 1
    print('Class   \tIoU')
    for i in range(opt.class_num):
        print('class %s\t%.10f' % (class_names[i], class_iou[i]))
    print('Mean IoU\t%.10f' % (np.mean(class_iou)))


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

import sys
import datetime
if __name__ == '__main__':
    args = parser.parse_args()

    args.root=r''
    args.ex_mode = 100
    args.self_max_epoch = 100
    args.ft_max_epoch = 100
    args.self_data_name = 'train'
    args.ft_train_name = 'trainR1'
    args.gpu = 0
    args.environ = '0'  # select gpu
    args.device_id = [0]
    # args.self_data_name = 'trainR1'
    args.arch = 'resnet50'            
    args.self_batch_size =16 #16
    args.num_workers = 4  # (args.device_id) * 4
    args.ft_val_name = 'val'
    args.self_mode = 1
    args = get_opt(args)
    if args.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.environ
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"
    cudnn.benchmark = args.benchmark
    cudnn.deterministic = args.deterministic
    cudnn.enabled = args.enabled
    args.ngpus_per_node = torch.cuda.device_count()
    if args.ngpus_per_node <= 1:
        args.use_mp = False
    if args.use_mp:
        args.world_size = args.ngpus_per_node * args.nodes
    total_batch_size = args.self_batch_size * args.world_size
    if args.ex_mode==1:
        args.mdl_path=None
    if args.mdl_name==None:
        args.mdl_name=Self_Mode_Name[args.self_mode]+'_'+args.self_data_name+'_bs'+str(args.self_batch_size)
        if total_batch_size != args.self_batch_size:
            args.mdl_name=args.mdl_name+'_'+str(total_batch_size)
            
    time = datetime.datetime.now()
    txt_name = datetime.datetime.strftime(time, '%Y_%m_%d_%H_%M_%S')
    args.mdl_name = args.mdl_name + '_'+txt_name[5:10]
    if args.save_log==True:
        if args.log_path==None:
            sys.stdout = Logger(txt_name+'_'+args.mdl_name+ '.txt')
        else:
            Path(args.log_path).mkdir(parents=True, exist_ok=True)
            sys.stdout = Logger(args.log_path+'/'+txt_name +'_'+args.mdl_name+ '.txt')
    print('test logger')
    main1(args)

