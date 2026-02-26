import torch
import random
import numpy as np
import os
import time
import cv2 as cv
import yaml
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn, optim
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import Microscopy_dataset, Microscopy_dataset_MFI
from torchvision.utils import save_image
from Unet import UNet
from noise import *
from common import *
from loss import *
from SharpAttention import *
from Quality_Evaluation import *
from skimage.metrics import structural_similarity as ssim


# 参数解析
def parse():
    parser = argparse.ArgumentParser(description="Train UNet")
    parser.add_argument('--config', default='./config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    return args


# 随机种子
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config):
    # 固定随机种子
    set_random_seed(config["rand_seed"])
    # 学习率
    lr_fuse = config["lr_fuse"]
    lr_enhance = config['lr_enhance']
    # 噪声
    reg_noise_std = config["reg_noise_std"]
    # 融合循环次数
    num_epochs_fuse = config['num_epochs_fuse']
    # 增强循环次数
    num_epochs_enhance = config['num_epochs_enhance']
    # 阈值
    thresh = config['thresh']
    alpha = config['alpha']
    use_att = config['use_att']

    enhance_alpha = config['enhance_alpha']
    enhance_beta = config['enhance_beta']

    # 保存周期
    save_interval = config['save_interval']
    # 输出位置
    output_path = config['output_path']
    # 加载数据集
    test_set = Microscopy_dataset_MFI(config["data_path"])
    print(f"Dataset length: {len(test_set)}")
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    # 加载设备
    device = config["device"]

    psnr_list = []
    SSIM_list = []
    metrics_sum = {}
    metrics_cnt = 0

    for (y1, y2, cr, cb, gt_y, img_name) in loader:
        # 发送到设备
        y1 = y1.to(device)
        y2 = y2.to(device)
        cr = cr.to(device)
        cb = cb.to(device)
        gt_y = gt_y.to(device)
        criterion_grad = GradientConsistencyLoss(device=device).to(device)
        # 获取图像长宽
        _, _, H, W = y1.shape
        # 噪声形状
        # print(net_inputm.shape)
        # 图像融合网络
        netx = UNet(num_input_channels=2,
                    num_output_channels=1,
                    num_channels_down=[16, 32, 64, 128],
                    num_channels_up=[16, 32, 64, 128],
                    num_channels_skip=[16, 16, 16],
                    upsample_mode='bilinear',
                    need_sigmoid=True).to(device)

        # 掩膜生成网络
        netm = UNet(num_input_channels=2,
                    num_output_channels=1,
                    num_channels_down=[16, 32, 64, 128, 128],
                    num_channels_up=[16, 32, 64, 128, 128],
                    num_channels_skip=[16, 16, 16, 16],
                    upsample_mode='bilinear',
                    need_sigmoid=True).to(device)

        # 输入噪声
        net_inputx = torch.cat([y1, y2], dim=1)
        net_inputm = get_noise(like_image=net_inputx).to(device)
        # net_inputm = torch.cat([y1,y2],dim=1)
        # 注意力
        att_module_y1 = ConvSharpnessQKVAttention(embed_dim=64, kernel_size=11, sigma=2.0).to(device)
        att_module_y2 = ConvSharpnessQKVAttention(embed_dim=64, kernel_size=11, sigma=2.0).to(device)

        ############################################### optimizer ######################################################
        parameters = []
        parameters.extend([{'params': netm.parameters()},
                           {'params': net_inputm},
                           {'params': netx.parameters()},
                           ])
        if use_att:
            print('使用注意力')
            parameters.append({'params': att_module_y1.parameters()})
            parameters.append({'params': att_module_y2.parameters()})

        optimizer = torch.optim.Adam(parameters, lr=lr_fuse)
        scheduler = MultiStepLR(optimizer, milestones=[200, 400, 800], gamma=0.5)

        net_input_savedm = net_inputm.detach().clone()
        noisem = net_inputm.detach().clone()

        score_map = []
        #生成初始焦点图 (引导项)
        mask1_init = get_score_map(y1, y2, mode='blur2th')
        mask2_init = 1 - mask1_init
        score_map.append(mask1_init)
        score_map.append(mask2_init)

        pbar = tqdm(range(num_epochs_fuse), desc="DIP-MFF processing", ncols=180)
        for step in pbar:
            optimizer.zero_grad()

            if reg_noise_std > 0:
                net_inputm = net_input_savedm + noisem.normal_() * reg_noise_std

            if use_att:
                y1_att = att_module_y1(y1)
                y2_att = att_module_y2(y2)
                net_inputx = torch.cat([y1_att, y2_att], dim=1)
            else:
                net_inputx = torch.cat([y1, y2], dim=1)

            mask1 = netm(net_inputm)  # 掩膜图像1

            mask2 = 1 - mask1  # 掩膜图像2

            fuse = netx(net_inputx)  # 融合图像
            # base = mask1 * y1 + (1 - mask1) * y2
            # residual = netx(torch.cat([y1, y2, base], dim=1))
            # fuse = base + residual

            ############################################### LOSS ######################################################
            # 初始化损失函数
            loss_percep = 0
            loss_recon = 0
            loss_prior = 0
            loss_ssim = 0

            # 先验损失
            loss_prior = F.l1_loss(mask1, score_map[0]) + F.l1_loss(mask2, score_map[1])
            # 熵损失
            # loss_entropy = Entropy_Loss(mask1)
            loss_grad = criterion_grad(fuse, y1, y2)
            # 重建损失
            recon = mask1 * y1 + mask2 * y2
            # recon = score_map[0] * y1 + score_map[1] * y2
            loss_recon = F.l1_loss(fuse, recon)
            # 结构损失
            loss_ssim = ssim_loss(fuse, recon)

            total_loss = alpha * loss_prior + loss_recon + 0.05 * loss_grad
            # total_loss = alpha * loss_prior + loss_recon
            # total_loss = loss_recon
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'prior': f'{loss_prior.item():.4f}',
                'recon': f'{loss_recon.item():.4f}',
            })

            total_loss.backward()
            optimizer.step()
            scheduler.step()
            #固定轮保存一次掩膜
            if (step + 1) % save_interval == 0:
                with torch.no_grad():
                    save_path_mask1 = os.path.join(output_path, f'mask1/mask1_epoch_{step + 1:04d}.png')
                    save_path_mask2 = os.path.join(output_path, f'mask2/mask2_epoch_{step + 1:04d}.png')
                    save_path_fuse = os.path.join(output_path, f'fuse/fuse_epoch_{step + 1:04d}.png')
                    save_image(fuse.clamp(0, 1), save_path_fuse)
                    save_image(mask1.clamp(0, 1), save_path_mask1)
                    save_image(mask2.clamp(0, 1), save_path_mask2)

        with torch.no_grad():
            final_fuse = fuse.clamp(0, 1)  # [B,1,H,W] in [0,1]

            #PSNR (Y channel)
            fuse_y = final_fuse.squeeze().cpu().numpy()  # [H,W]
            gt_y_np = gt_y.squeeze().cpu().numpy()  # [H,W]

            cur_psnr = psnr(fuse_y, gt_y_np)
            cur_ssim = ssim(fuse_y, gt_y_np, data_range=1.0)

            psnr_list.append(cur_psnr)
            SSIM_list.append(cur_ssim)

            print(f"[{img_name[0]}] PSNR = {cur_psnr} dB")
            print(f"[{img_name[0]}] SSIM = {cur_ssim}")
            #无参考质量指标
            quality_metrics = evaluate_fused_gray(fuse_y)
            for k, v in quality_metrics.items():
                print(f"{k:12s}: {v:.4f}")
            for k, v in quality_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            metrics_cnt += 1

            Y_8u = (final_fuse.squeeze().cpu().numpy() * 255.0).round().astype(np.uint8)  # [H,W]
            Cr_8u = cr.squeeze().cpu().numpy().astype(np.uint8)
            Cb_8u = cb.squeeze().cpu().numpy().astype(np.uint8)
            ycrcb8 = np.stack([Y_8u, Cr_8u, Cb_8u], axis=-1)
            fused_bgr = cv.cvtColor(ycrcb8, cv.COLOR_YCrCb2BGR)

            save_path_rgb = os.path.join(
                output_path, "fuse", f"{img_name[0]}_fused.jpg"
            )
            cv.imwrite(save_path_rgb, fused_bgr)
            #print(f"彩色融合结果已保存到: {save_path_rgb}")

    mean_psnr = sum(psnr_list) / len(psnr_list)
    mean_ssim = sum(SSIM_list) / len(SSIM_list)

    print("=" * 60)
    print(f"Average PSNR over {len(psnr_list)} image pairs: {mean_psnr} dB")
    print(f"Average SSIM over {len(SSIM_list)} image pairs: {mean_ssim} dB")
    print("=" * 60)
    print("\n========= Average Quality Metrics =========")
    for k, v in metrics_sum.items():
        print(f"{k:12s}: {v / metrics_cnt:.4f}")


if __name__ == "__main__":
    args = parse()

    with open(args.config, mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    set_random_seed(config['rand_seed'])

    main(config)
