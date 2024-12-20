from Net.net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
from tools.Evaluator import Evaluator
import torch
import torch.nn as nn
from tools.img_read_save import img_save,image_read_cv2
import warnings
import logging
from PIL import Image
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

ckpt_path=r"/home/cuixinyu/test/model/fusion.pth"
for dataset_name in ["TNO"]:
    print("\n"*2+"="*80)
    model_name="ImageFuse    "
    print("The test result  "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name)
    test_out_folder=os.path.join('result/MSRS',dataset_name)
    # test_out_folder_base = os.path.join('result/test_result_detail_1', dataset_name)
    # test_out_folder_detail = os.path.join('result/test_result_base_1', dataset_name)
    # test_out_folder_hat = os.path.join('result/test_result_fuse_2', dataset_name)
    # test_out_folder_base_hat = os.path.join('result/test_result_detail_2', dataset_name)
    # test_out_folder_detail_hat = os.path.join('result/test_result_base_2', dataset_name)

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
        for img_name in os.listdir(os.path.join(test_folder,"ir")):

            data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)

            # 第一次的融合图
            data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = fi.astype(np.uint8)
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)



            # 背景特征图--第一轮
            # feature_F_B_image,_ = Decoder(data_VIS, feature_F_B, feature_F_B)
            #
            # feature_F_B_image = (feature_F_B_image - torch.min(feature_F_B_image)) / (torch.max(feature_F_B_image) - torch.min(feature_F_B_image))
            # B_image = np.squeeze((feature_F_B_image * 255).cpu().numpy())
            # B_image = B_image.astype(np.uint8)
            # img_save(B_image, img_name.split(sep='.')[0], test_out_folder_base)
            # 细节特征图--第一轮
            # feature_F_D_image, _ = Decoder(data_VIS, feature_F_D, feature_F_D)
            #
            # feature_F_D_image = (feature_F_D_image - torch.min(feature_F_D_image)) / (
            #             torch.max(feature_F_D_image) - torch.min(feature_F_D_image))
            # D_image = np.squeeze((feature_F_D_image * 255).cpu().numpy())
            # D_image = D_image.astype(np.uint8)
            # img_save(D_image, img_name.split(sep='.')[0], test_out_folder_detail)

            # 将融合图分解为背景和细节信息
            feature_F_B_hat, feature_F_D_hat, feature_I_hat = Encoder(data_Fuse)

            # 背景特征图--第二轮
            # feature_F_B_image_hat,_ = Decoder(data_VIS, feature_F_B_hat, feature_F_B_hat)
            # feature_F_B_image_hat = (feature_F_B_image_hat - torch.min(feature_F_B_image_hat)) / (torch.max(feature_F_B_image_hat) - torch.min(feature_F_B_image_hat))
            # B_image_hat = np.squeeze((feature_F_B_image_hat * 255).cpu().numpy())
            # B_image_hat = B_image_hat.astype(np.uint8)
            # img_save(B_image_hat, img_name.split(sep='.')[0], test_out_folder_base_hat)

            # 细节特征图--第二轮
            # feature_F_D_image_hat, _ = Decoder(data_VIS, feature_F_D_hat, feature_F_D_hat)
            #
            # feature_F_D_image_hat = (feature_F_D_image_hat - torch.min(feature_F_D_image_hat)) / (
            #             torch.max(feature_F_D_image_hat) - torch.min(feature_F_D_image_hat))
            # D_image_hat = np.squeeze((feature_F_D_image_hat * 255).cpu().numpy())
            # D_image_hat = D_image.astype(np.uint8)
            # img_save(D_image_hat, img_name.split(sep='.')[0], test_out_folder_detail_hat)


            # 第二轮的融合图
            data_Fuse_hat, _ = Decoder(data_VIS, feature_F_B_hat, feature_F_D_hat)
            data_Fuse_hat = (data_Fuse_hat - torch.min(data_Fuse_hat)) / (torch.max(data_Fuse_hat) - torch.min(data_Fuse_hat))
            fi_hat = np.squeeze((data_Fuse_hat * 255).cpu().numpy())
            fi_hat = fi.astype(np.uint8)
            img_save(fi_hat, img_name.split(sep='.')[0], test_out_folder)




    eval_folder=test_out_folder
    ori_img_folder=test_folder

    metric_result = np.zeros((8))
    for img_name in os.listdir(os.path.join(ori_img_folder,"ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                        , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                        , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 3))+'\t'
            +str(np.round(metric_result[2], 3))+'\t'
            +str(np.round(metric_result[3], 3))+'\t'
            +str(np.round(metric_result[4], 3))+'\t'
            +str(np.round(metric_result[5], 3))+'\t'
            +str(np.round(metric_result[6], 3))+'\t'
            +str(np.round(metric_result[7], 3))
            )
    print("="*80)