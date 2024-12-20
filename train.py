#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from torch.autograd import Variable
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
from tools.img_read_save import img_save, image_read_cv2
import logging
import os.path as osp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from logger import setup_logger
from model_TII import BiSeNet

from cityscapes import CityScapes
from loss import OhemCELoss, Fusionloss, cc
from optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
import warnings
from dataset import H5Dataset

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction, Transformer
import sys
import torch.nn as nn
import kornia

warnings.filterwarnings('ignore')



def rgb2y(img):
    y = img[:, 0:1, :, :] * 0.299000 + img[:, 1:2, :, :] * 0.587000 + img[:, 2:3, :, :] * 0.114000
    return y


def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)

    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )

    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def train_seg(i=0, logger=None, args=None):
    load_path = './model/newFusion/model_final.pth'
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    os.makedirs(modelpth, mode=0o777, exist_ok=True)

    # dataset
    n_classes = 9
    n_img_per_gpu = args.batch_size
    n_workers = 4
    cropsize = [640, 480]
    ds = CityScapes('./datasets/MSRS/', cropsize=cropsize, Method=Method)
    dl = DataLoader(
        ds,
        batch_size=n_img_per_gpu,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model
    ignore_idx = 255
    net = BiSeNet(n_classes=n_classes)
    if i > 0:
        net.load_state_dict(torch.load(load_path))
    net.cuda()
    net.train()
    print('Load Pre-trained Segmentation Model:{}!'.format(load_path))
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16

    # loss
    criteria_p = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    # optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    it_start = i * 20000
    iter_nums = 20000

    optim = Optimizer(
        model=net,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
        it=it_start,
    )

    # train loop
    msg_iter = 10
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(iter_nums):
        try:
            im, lb, _ = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            # sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb, _ = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, mid = net(im)

        # 损失---两部分损失相加
        lossp = criteria_p(out, lb)
        loss2 = criteria_16(mid, lb)
        loss = lossp + 0.75 * loss2

        loss.backward()
        optim.step()

        loss_avg.append(loss.item())


        # print training log message
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)

            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join(
                [
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]
            ).format(
                it=it_start + it + 1, max_it=max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed


    # 分割训练完成后,存储最后的模型
    save_pth = osp.join(modelpth, 'model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info(
        '分割训练已完成,模型被保存在: {}'.format(
            save_pth)
    )
    logger.info('\n')


# 训练融合模型
def train_fusion(num=0, logger=None):
    # transformer
    shift_num = 1
    rotate_num = 1
    flip_num = 1
    tran = Transformer(shift_num, rotate_num, flip_num)

    # num: control the segmodel
    optim_step = 20
    lr = 1e-4

    # Coefficients of the loss function
    coeff_mse_loss_B = 1.
    coeff_mse_loss_D = 1.
    coeff_mse_loss_F = 1.
    clip_grad_norm_value = 0.01
    optim_gamma = 0.5
    weight_decay = 0
    num_epochs = 10
    modelpth = '/home/cuixinyu/Fusion_new/model'
    Method = 'newFusion'
    modelpth = os.path.join(modelpth, Method)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    if num > 0 and os.path.exists('/home/cuixinyu/Fusion_new/model/newFusion/fusion_nodecomp.pth'):
        DIDF_Encoder.load_state_dict(torch.load('/home/cuixinyu/Fusion_new/model/newFusion/fusion_nodecomp.pth')['DIDF_Encoder'])
        DIDF_Decoder.load_state_dict(torch.load('/home/cuixinyu/Fusion_new/model/newFusion/fusion_nodecomp.pth')['DIDF_Decoder'])
        BaseFuseLayer.load_state_dict(torch.load('/home/cuixinyu/Fusion_new/model/newFusion/fusion_nodecomp.pth')['BaseFuseLayer'])
        DetailFuseLayer.load_state_dict(torch.load('/home/cuixinyu/Fusion_new/model/newFusion/fusion_nodecomp.pth')['DetailFuseLayer'])
        DIDF_Encoder.eval()
        DIDF_Decoder.eval()
        BaseFuseLayer.eval()
        DetailFuseLayer.eval()


    # optimizer, scheduler and loss function
    optimizer1 = torch.optim.Adam(
        DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer2 = torch.optim.Adam(
        DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer3 = torch.optim.Adam(
        BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer4 = torch.optim.Adam(
        DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
    

    # 加载预训练的分割模型，并将其设置为评估模式
    if num > 0:
        n_classes = 9
        segmodel = BiSeNet(n_classes=n_classes)
        save_pth = osp.join(modelpth, 'model_final.pth')
        if logger == None:
            logger = logging.getLogger()
            setup_logger(modelpth)
        segmodel.load_state_dict(torch.load(save_pth))
        segmodel.cuda()
        segmodel.eval()
        for p in segmodel.parameters():
            p.requires_grad = False
        print('Load Segmentation Model {} Sucessfully~'.format(save_pth))

    MSELoss = nn.MSELoss()
    Loss_ssim = kornia.losses.SSIM(11, reduction='mean')

    trainloader = DataLoader(H5Dataset(r"/home/cuixinyu/Fusion_new/data/MSRS_imgsize_128_stride_new200.h5"),
                             batch_size=8,
                             shuffle=True,
                             num_workers=0)

    loader = {'train': trainloader}


    # 定义了一个名为criteria_fusion的类，它是用于计算图像分割任务的损失函数,从第二轮开始计算
    if num > 0:
        score_thres = 0.7
        ignore_idx = 255
        n_min = 8 * 128 * 128 // 8
        criteria_p = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        criteria_16 = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_fusion = Fusionloss()

    epoch = 10
    logger.info('Training Fusion Model start~')
    torch.backends.cudnn.benchmark = True

    prev_time = time.time()

    for epo in range(0, epoch):
        # print('\n| epo #%s begin...' % epo)
        # 定义学习率和衰减系数
        for it, (image_vis, image_ir, label) in enumerate(trainloader):
            # 第一个数据集的红外和可见光用来做融合
            image_vis = torch.FloatTensor(image_vis).cuda()
            data_IR = torch.FloatTensor(image_ir).cuda()
            label = torch.FloatTensor(label).cuda()

            # 灰度图
            data_VIS = rgb2y(image_vis)


            DIDF_Encoder.train()
            DIDF_Decoder.train()
            BaseFuseLayer.train()
            DetailFuseLayer.train()

            DIDF_Encoder.zero_grad()
            DIDF_Decoder.zero_grad()
            BaseFuseLayer.zero_grad()
            DetailFuseLayer.zero_grad()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()

            # 融合输出模型
            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)

            # 细节和背景特征图进行旋转、平移、翻折变换
            tran_feature_F_D = tran.apply(feature_F_D)
            tran_feature_F_B = tran.apply(feature_F_B)

            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)


            # 第二个数据集的可见光用把融合图拼接为彩色
            image_vis_ycrcb = RGB2YCrCb(image_vis)

            # 将融合后的图像转变为彩色的RGB格式
            fusion_image = torch.cat(
                (
                    data_Fuse,
                    image_vis_ycrcb[:, 1:2, :, :],
                    image_vis_ycrcb[:, 2:, :, :]
                ),
                dim=1,
            )

            fusion_image = YCrCb2RGB(fusion_image)

            # 融合图像
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

            # 融合图进行旋转、平移、翻折变换
            tran_feature_F = tran.apply(data_Fuse)

            feature_F_B_hat, feature_F_D_hat, feature_I_hat = DIDF_Encoder(tran_feature_F)
            data_Fuse_hat, feature_F_hat = DIDF_Decoder(tran_feature_F, feature_F_B_hat, feature_F_D_hat)

            # 消融实验，去掉等变性
            # feature_F_B_hat, feature_F_D_hat, feature_I_hat = DIDF_Encoder(data_Fuse)
            # data_Fuse_hat, feature_F_hat = DIDF_Decoder(data_Fuse, feature_F_B_hat, feature_F_D_hat)

            # 数据集的标签用来算损失函数
            lb = torch.squeeze(label, 1).long()

            # seg loss
            if num > 0:
                out, mid = segmodel(fusion_image)
                lossp = criteria_p(out, lb)
                seg_loss = lossp
            else:
                seg_loss = 0

            # fusion loss
            # # 背景图,细节图和融合图的损失--3个
            mse_loss_b = 5 * Loss_ssim(tran_feature_F_B, feature_F_B_hat) + MSELoss(tran_feature_F_B,
                                                                                    feature_F_B_hat)
            mse_loss_d = 5 * Loss_ssim(tran_feature_F_D, feature_F_D_hat) + MSELoss(tran_feature_F_D,
                                                                                    feature_F_D_hat)
            mse_loss_f = 5 * Loss_ssim(tran_feature_F, data_Fuse_hat) + MSELoss(tran_feature_F, data_Fuse_hat)

            mse_loss = coeff_mse_loss_B * mse_loss_b + coeff_mse_loss_D * mse_loss_d + coeff_mse_loss_F * mse_loss_f

            # # 计算特征分解损失--2个
            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            cc_loss_B_hat = cc(tran_feature_F_B, feature_F_B_hat)
            cc_loss_D_hat = cc(tran_feature_F_D, feature_F_D_hat)
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
            loss_decomp_hat = (cc_loss_D_hat) ** 2 / (1.01 + cc_loss_B_hat)

            # 去掉等变性后的特征损失函数

            # 背景图,细节图和融合图的损失--3个
            # mse_loss_b = 5 * Loss_ssim(feature_F_B, feature_F_B_hat) + MSELoss(feature_F_B,
            #                                                                         feature_F_B_hat)
            # mse_loss_d = 5 * Loss_ssim(feature_F_D, feature_F_D_hat) + MSELoss(feature_F_D,
            #                                                                         feature_F_D_hat)
            # mse_loss_f = 5 * Loss_ssim(data_Fuse, data_Fuse_hat) + MSELoss(data_Fuse, data_Fuse_hat)
            #
            # mse_loss = coeff_mse_loss_B * mse_loss_b + coeff_mse_loss_D * mse_loss_d + coeff_mse_loss_F * mse_loss_f
            #
            # cc_loss_B = cc(feature_V_B, feature_I_B)
            # cc_loss_D = cc(feature_V_D, feature_I_D)
            # cc_loss_B_hat = cc(feature_F_B, feature_F_B_hat)
            # cc_loss_D_hat = cc(feature_F_D, feature_F_D_hat)
            # loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
            # loss_decomp_hat = (cc_loss_D_hat) ** 2 / (1.01 + cc_loss_B_hat)



            # 梯度和强度损失的和--2个
            loss_ig, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

            # loss_fusion = loss_ig + mse_loss + loss_decomp_hat + loss_decomp

            # 消融实验，去掉特征分解损失函数
            loss_fusion = loss_ig + mse_loss


            if num > 0:
                loss_total = loss_fusion + 0.2 * seg_loss
            else:
                loss_total = loss_fusion

            loss_total.backward()

            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

            # Determine approximate time left
            batches_done = epoch * len(loader['train']) + i
            batches_left = num_epochs * len(loader['train']) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

            if num > 0:
                sys.stdout.write(
                    "\r[num %d][Epoch %d/%d] [Batch %d/%d] [loss_total: %f] [loss_fusion: %f] [loss_seg:%f]ETA: %.10s"
                    % (num,
                        epo,
                       num_epochs,
                       it,
                       len(loader['train']),
                       loss_total.item(),
                       loss_fusion,
                       seg_loss,
                       time_left,
                       )
                )
            else:
                sys.stdout.write(
                    "\r[num %d][Epoch %d/%d] [Batch %d/%d] [loss: %f]  [loss_fusion: %f] ETA: %.10s"
                    % (
                        num,
                        epo,
                        num_epochs,
                        it,
                        len(loader['train']),
                        loss_total.item(),
                        loss_fusion,
                        time_left,
                    )
                )

        # 调整学习率
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()

        # 更新所有参数的学习率
        if optimizer1.param_groups[0]['lr'] <= 1e-6:
            optimizer1.param_groups[0]['lr'] = 1e-6
        if optimizer2.param_groups[0]['lr'] <= 1e-6:
            optimizer2.param_groups[0]['lr'] = 1e-6
        if optimizer3.param_groups[0]['lr'] <= 1e-6:
            optimizer3.param_groups[0]['lr'] = 1e-6
        if optimizer4.param_groups[0]['lr'] <= 1e-6:
            optimizer4.param_groups[0]['lr'] = 1e-6

    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    checkpoint_path = f"model/newFusion/fusion_nodecomp.pth"
    torch.save(checkpoint, checkpoint_path)


def run_fusion():
    ckpt_path = r"/home/cuixinyu/Fusion_new/model/newFusion/fusion_nodecomp.pth"
    train_folder = os.path.join('/home/cuixinyu/MMIF-CDDFuse/MSRS_train/train')
    train_out_folder = os.path.join('/home/cuixinyu/Fusion_new/datasets/MSRS/Fusion')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(train_folder, "ir")):
            data_IR = image_read_cv2(os.path.join(train_folder, "ir", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = image_read_cv2(os.path.join(train_folder, "vi", img_name), mode='GRAY')[
                           np.newaxis, np.newaxis, ...] / 255.0
            data_vis = image_read_cv2(os.path.join(train_folder, "vi", img_name))
            data_vis = np.asarray(np.array(data_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0
            data_vis = torch.tensor(data_vis)
            data_vis = data_vis.unsqueeze(0)
            images_vis = torch.FloatTensor(data_vis)
            images_vis = images_vis.cuda()
            images_vis_ycrcb = RGB2YCrCb(images_vis)

            data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)

            data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)

            # 将融合后的图像转变为彩色的RGB格式
            fusion_image = torch.cat(
                (data_Fuse, images_vis_ycrcb[:, 1:2, :, :],
                 images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )

            fusion_image = YCrCb2RGB(fusion_image)
            # 融合图像

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            data_Fuse = fused_image.transpose((0, 2, 3, 1))
            data_Fuse = torch.tensor(data_Fuse, dtype=torch.float32)

            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = fi.astype(np.uint8)

            img_save(fi, img_name.split(sep='.')[0], train_out_folder)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default = 'Fusion')
    parser.add_argument('--batch_size', '-B', type = int, default = 2)
    parser.add_argument('--gpu', '-G', type = int, default = [0, 1, 2, 3])
    parser.add_argument('--num_workers', '-j', type = int, default = 4)
    args = parser.parse_args()
    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    for i in range (0,2):
        train_fusion(i, logger)
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
        run_fusion()
        print("|{0} Fusion Image Sucessfully~!".format(i + 1))
        train_seg(i, logger, args)
        print("|{0} Train Segmentation Model Sucessfully~!".format(i + 1))
    print("training Done!")
