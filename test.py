import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# 路径设置--------------------------------------------------
pred_folder = "results/LOL_v1/best_23.5233"        # 预测结果文件夹路径
gt_folder = "data/LOLv1/Test/target"    # GT 文件夹路径
#-------------------------------------------------------------


def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

gt_files = list_images(gt_folder)
pred_files = list_images(pred_folder)

print(len(gt_files))
print(len(pred_files))


assert len(gt_files) == len(pred_files)



psnr_list, ssim_list, rmse_list = [], [], []

for gt_name, pred_name in zip(gt_files, pred_files):
    gt_path = os.path.join(gt_folder, gt_name)
    pred_path = os.path.join(pred_folder, pred_name)
    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)

    if gt is None or pred is None:
        print(f"跳过文件 {gt_name} 或 {pred_name}，无法读取。")
        continue

    # 保证大小一致
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    # 转换为 float64 并归一化
    gt_norm = gt.astype(np.float64) / 255.0
    pred_norm = pred.astype(np.float64) / 255.0

    # PSNR
    psnr = peak_signal_noise_ratio(gt_norm, pred_norm, data_range=1.0)

    # SSIM（多通道）
    ssim = structural_similarity(gt_norm, pred_norm, channel_axis=-1, data_range=1.0)
    rmse = np.sqrt(mean_squared_error(gt, pred))

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    rmse_list.append(rmse)

    print(f"{gt_name}: PSNR={psnr:.4f}, SSIM={ssim:.4f}, RMSE={rmse:.4f}")

# 平均结果
print("\n==== 平均结果 ====")
print(f"PSNR: {np.mean(psnr_list):.4f}")
print(f"SSIM: {np.mean(ssim_list):.4f}")
print(f"RMSE: {np.mean(rmse_list):.4f}")
