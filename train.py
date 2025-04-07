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
    # Set model save/load path
    load_path = '/home/cuixinyu/model/model_final.pth'
    modelpth = '/home/cuixinyu'
    Method = 'model'
    modelpth = os.path.join(modelpth, Method)
    os.makedirs(modelpth, mode=0o777, exist_ok=True)

    # ----------------------
    # Dataset Preparation
    # ----------------------
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

    # ----------------------
    # Model Initialization
    # ----------------------
    ignore_idx = 255
    net = BiSeNet(n_classes=n_classes)
    if i > 0:
        net.load_state_dict(torch.load(load_path))  # Load pretrained weights if provided
    net.cuda()
    net.train()
    print('Load Pre-trained Segmentation Model:{}!'.format(load_path))

    # ----------------------
    # Loss Function Setup
    # ----------------------
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)  # primary output loss
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)  # auxiliary loss

    # ----------------------
    # Optimizer & Scheduler
    # ----------------------
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

    # ----------------------
    # Training Loop
    # ----------------------
    msg_iter = 10
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0

    for it in range(iter_nums):
        # Load batch data
        try:
            im, lb, _ = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            # If DataLoader is exhausted, reinitialize for next epoch
            epoch += 1
            diter = iter(dl)
            im, lb, _ = next(diter)

        im = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)  # remove channel dimension

        # Forward pass
        optim.zero_grad()
        out, mid = net(im)

        # Compute loss for both outputs
        lossp = criteria_p(out, lb)         # loss for main output
        loss2 = criteria_16(mid, lb)        # loss for auxiliary output
        loss = lossp + 0.75 * loss2         # weighted total loss

        # Backward and optimize
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        # Logging every msg_iter steps
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv = ed - st
            glob_t_intv = ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))

            msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it=it_start + it + 1,
                max_it=max_iter,
                lr=lr,
                loss=loss_avg,
                time=t_intv,
                eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed

    # ----------------------
    # Save Final Model
    # ----------------------
    save_pth = osp.join(modelpth, 'model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info('Segmentation training finished. Model saved at: {}'.format(save_pth))
    logger.info('\n')






def train_fusion(num=0, logger=None):
    # Initialize geometric transformer for consistency learning
    shift_num = 1
    rotate_num = 1
    flip_num = 1
    tran = Transformer(shift_num, rotate_num, flip_num)

    # Set optimization parameters
    optim_step = 20
    lr = 1e-4

    # Loss weights
    coeff_mse_loss_B = 1.
    coeff_mse_loss_D = 1.
    coeff_mse_loss_F = 1.
    clip_grad_norm_value = 0.01
    optim_gamma = 0.5
    weight_decay = 0
    num_epochs = 10

    # Model path and device setup
    modelpth = '/home/cuixinyu/TEDFuse/model'
    Method = 'TEDFuse'
    modelpth = os.path.join(modelpth, Method)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize encoder, decoder, and fusion modules
    DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    # Load pretrained fusion model if available
    if num > 0 and os.path.exists('/home/cuixinyu/Fusion_new/model/newFusion/fusion_nodecomp.pth'):
        ckpt = torch.load('/home/cuixinyu/Fusion_new/model/newFusion/fusion_nodecomp.pth')
        DIDF_Encoder.load_state_dict(ckpt['DIDF_Encoder'])
        DIDF_Decoder.load_state_dict(ckpt['DIDF_Decoder'])
        BaseFuseLayer.load_state_dict(ckpt['BaseFuseLayer'])
        DetailFuseLayer.load_state_dict(ckpt['DetailFuseLayer'])

        DIDF_Encoder.eval()
        DIDF_Decoder.eval()
        BaseFuseLayer.eval()
        DetailFuseLayer.eval()

    # Define separate optimizers for each module
    optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate schedulers
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

    # Load segmentation model if segmentation supervision is enabled
    if num > 0:
        n_classes = 9
        segmodel = BiSeNet(n_classes=n_classes)
        save_pth = osp.join(modelpth, 'model_final.pth')
        if logger is None:
            logger = logging.getLogger()
            setup_logger(modelpth)
        segmodel.load_state_dict(torch.load(save_pth))
        segmodel.cuda()
        segmodel.eval()
        for p in segmodel.parameters():
            p.requires_grad = False
        print('Load Segmentation Model {} Sucessfully~'.format(save_pth))

    # Loss functions
    MSELoss = nn.MSELoss()
    Loss_ssim = kornia.losses.SSIM(11, reduction='mean')

    # Load training dataset
    trainloader = DataLoader(H5Dataset(r"/home/cuixinyu/Fusion_new/data/MSRS_imgsize_128_stride_200.h5"),
                             batch_size=8, shuffle=True, num_workers=0)
    loader = {'train': trainloader}

    # Define segmentation loss if enabled
    if num > 0:
        score_thres = 0.7
        ignore_idx = 255
        n_min = 8 * 128 * 128 // 8
        criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    # Fusion image quality loss
    criteria_fusion = Fusionloss()

    epoch = 10
    logger.info('Training Fusion Model start~')
    torch.backends.cudnn.benchmark = True
    prev_time = time.time()

    for epo in range(epoch):
        for it, (image_vis, image_ir, label) in enumerate(trainloader):
            # Load images and move to GPU
            image_vis = torch.FloatTensor(image_vis).cuda()
            data_IR = torch.FloatTensor(image_ir).cuda()
            label = torch.FloatTensor(label).cuda()
            data_VIS = rgb2y(image_vis)

            # Enable training mode
            DIDF_Encoder.train()
            DIDF_Decoder.train()
            BaseFuseLayer.train()
            DetailFuseLayer.train()

            # Zero all gradients
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()

            # Feature extraction
            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)

            # Fusion of base and detail features
            feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)

            # Apply data-level transformation for equivariant consistency
            tran_feature_F_D = tran.apply(feature_F_D)
            tran_feature_F_B = tran.apply(feature_F_B)

            # Decode fused feature into image
            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)

            # Convert Y channel fusion result into RGB using YCrCb from original
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            fusion_image = torch.cat((data_Fuse, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:, :, :]), dim=1)
            fusion_image = YCrCb2RGB(fusion_image)

            # Clamp values to [0, 1]
            fusion_image = torch.clamp(fusion_image, 0.0, 1.0)

            # Re-encode transformed image to measure equivariance
            tran_feature_F = tran.apply(data_Fuse)
            feature_F_B_hat, feature_F_D_hat, feature_I_hat = DIDF_Encoder(tran_feature_F)
            data_Fuse_hat, feature_F_hat = DIDF_Decoder(tran_feature_F, feature_F_B_hat, feature_F_D_hat)

            lb = torch.squeeze(label, 1).long()

            # Segmentation loss
            if num > 0:
                out, mid = segmodel(fusion_image)
                lossp = criteria_p(out, lb)
                seg_loss = lossp
            else:
                seg_loss = 0

            # Feature-level equivariant loss
            mse_loss_b = 5 * Loss_ssim(tran_feature_F_B, feature_F_B_hat) + MSELoss(tran_feature_F_B, feature_F_B_hat)
            mse_loss_d = 5 * Loss_ssim(tran_feature_F_D, feature_F_D_hat) + MSELoss(tran_feature_F_D, feature_F_D_hat)
            mse_loss_f = 5 * Loss_ssim(tran_feature_F, data_Fuse_hat) + MSELoss(tran_feature_F, data_Fuse_hat)
            mse_loss = coeff_mse_loss_B * mse_loss_b + coeff_mse_loss_D * mse_loss_d + coeff_mse_loss_F * mse_loss_f

            # Decomposition constraint loss
            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            cc_loss_B_hat = cc(tran_feature_F_B, feature_F_B_hat)
            cc_loss_D_hat = cc(tran_feature_F_D, feature_F_D_hat)
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
            loss_decomp_hat = (cc_loss_D_hat) ** 2 / (1.01 + cc_loss_B_hat)

            # Image-level fusion loss
            loss_ig, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

            # Total fusion loss
            loss_fusion = loss_ig + mse_loss + loss_decomp_hat + loss_decomp

            # Final loss with optional segmentation supervision
            if num > 0:
                loss_total = loss_fusion + 0.2 * seg_loss
            else:
                loss_total = loss_fusion

            # Backpropagation
            loss_total.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value)
            nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value)
            nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value)

            # Optimizer update
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

            # Estimate time remaining
            batches_done = epoch * len(loader['train']) + it
            batches_left = num_epochs * len(loader['train']) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

            # Print training log
            if num > 0:
                sys.stdout.write(
                    "\r[num %d][Epoch %d/%d] [Batch %d/%d] [loss_total: %f] [loss_fusion: %f] [loss_seg:%f] ETA: %.10s"
                    % (num, epo, num_epochs, it, len(loader['train']), loss_total.item(), loss_fusion, seg_loss, time_left)
                )
            else:
                sys.stdout.write(
                    "\r[num %d][Epoch %d/%d] [Batch %d/%d] [loss: %f] [loss_fusion: %f] ETA: %.10s"
                    % (num, epo, num_epochs, it, len(loader['train']), loss_total.item(), loss_fusion, time_left)
                )

        # Update learning rate
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()

        # Clamp learning rate to minimum value
        for optimizer in [optimizer1, optimizer2, optimizer3, optimizer4]:
            if optimizer.param_groups[0]['lr'] <= 1e-6:
                optimizer.param_groups[0]['lr'] = 1e-6

    # Save trained model
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    checkpoint_path = f"model/fusion.pth"
    torch.save(checkpoint, checkpoint_path)



def run_fusion():
    # Path to the pre-trained fusion model checkpoint
    ckpt_path = r"/home/cuixinyu/Fusion_new/model/fusion.pth"
    
    # Input dataset path (contains 'ir' and 'vi' folders)
    train_folder = os.path.join('/home/cuixinyuTEDFuse/MSRS')
    
    # Output folder to save fused results
    train_out_folder = os.path.join('/home/cuixinyu/TEDFuse/datasets/MSRS/Fusion')

    # Set device to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load encoder, decoder, and fusion modules, and move them to GPU
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

    # Disable gradient computation (for faster inference)
    with torch.no_grad():
        # Loop over all infrared-visible image pairs
        for img_name in os.listdir(os.path.join(train_folder, "ir")):
            # Load infrared image in grayscale and normalize
            data_IR = image_read_cv2(os.path.join(train_folder, "ir", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            
            
            data_VIS = image_read_cv2(os.path.join(train_folder, "vi", img_name), mode='GRAY')[
                           np.newaxis, np.newaxis, ...] / 255.0

            # Load visible image in RGB for color restoration
            data_vis = image_read_cv2(os.path.join(train_folder, "vi", img_name))
            data_vis = np.asarray(np.array(data_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0
            data_vis = torch.tensor(data_vis).unsqueeze(0).cuda()
            images_vis_ycrcb = RGB2YCrCb(data_vis)  # Convert to YCrCb for luminance replacement

            # Convert IR and VIS grayscale images to tensor and move to GPU
            data_IR, data_VIS = torch.FloatTensor(data_IR).cuda(), torch.FloatTensor(data_VIS).cuda()

            # Pass IR and VIS through encoder to extract features
            feature_V_B, feature_V_D, _ = Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = Encoder(data_IR)

            # Fuse base and detail features
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)

            # Decode fused features to obtain fused grayscale image
            data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)

            # Replace luminance channel in YCrCb image with fused image
            fusion_image = torch.cat(
                (data_Fuse, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            # Convert back to RGB
            fusion_image = YCrCb2RGB(fusion_image)

            # Clamp pixel values to [0,1]
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

            # Convert tensor to numpy array and normalize to [0, 255]
            fused_image = fusion_image.cpu().numpy().transpose((0, 2, 3, 1))
            data_Fuse = torch.tensor(fused_image, dtype=torch.float32)
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))

            # Convert to 8-bit image
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = fi.astype(np.uint8)

            # Save fused result
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
