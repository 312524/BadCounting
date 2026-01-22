# -*- coding: GB2312 -*-
import numpy as np
import cv2
import os
import random
from scipy.ndimage import gaussian_filter, median_filter
from skimage import restoration, exposure


POISON_ROOT = "defense/Data_Fine-tuning_Transformation/train/"   
CLEAN_ROOT = "defense/Data_Fine-tuning_Transformation/defensed_train/"  

UNIVERSAL_DEFENSE_CONFIG = {
    "denoise": {"gaussian_sigma": 1.0, "median_size": 3, "nl_means_h": 0.12},
    "augment": {"random_flip": True, "random_crop": True, "crop_ratio": 0.1},
    "normalize": {"rgb_mean": [0.485, 0.456, 0.406], "rgb_std": [0.229, 0.224, 0.225], "t_clahe_clip": 2.0},
    "repair": {"anomaly_thresh": 3.0, "inpaint_radius": 4}
}

SUPPORTED_FORMATS = ('.jpg')


def create_dirs():
    os.makedirs(os.path.join(CLEAN_ROOT, "RGB"), exist_ok=True)
    os.makedirs(os.path.join(CLEAN_ROOT, "T"), exist_ok=True)

def load_rgb_files(root_dir):
    rgb_files = []
    for root, _, files in os.walk(root_dir):
        for fil in files:
            if fil.lower().endswith(SUPPORTED_FORMATS) and "_RGB" in fil:
                rgb_files.append((root, fil))
    return rgb_files

def load_t_files(root_dir):
    t_files = []
    for root, _, files in os.walk(root_dir):
        for fil in files:
            if fil.lower().endswith(SUPPORTED_FORMATS) and "_T" in fil:
                t_files.append((root, fil))
    return t_files

def universal_image_defense(img, is_thermal=False, config=UNIVERSAL_DEFENSE_CONFIG):
    orig_h, orig_w = img.shape[:2]
    img_defense = img.copy().astype(np.float32) / 255.0

    # Step1: 异常区域检测
    if is_thermal:
        mean = np.mean(img_defense)
        std = np.std(img_defense)
        anomaly_mask = (img_defense > mean + config["repair"]["anomaly_thresh"] * std) | (img_defense < mean - config["repair"]["anomaly_thresh"] * std)
        anomaly_mask = anomaly_mask.astype(np.uint8) * 255
    else:
        mean = np.mean(img_defense, axis=(0,1))
        std = np.std(img_defense, axis=(0,1))
        anomaly_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for c in range(3):
            channel_anomaly = (img_defense[...,c] > mean[c] + config["repair"]["anomaly_thresh"] * std[c]) | (img_defense[...,c] < mean[c] - config["repair"]["anomaly_thresh"] * std[c])
            anomaly_mask = np.maximum(anomaly_mask, channel_anomaly.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)

    # Step2: 通用去噪（核心修复：sigma_est取均值）
    img_defense = median_filter(img_defense, size=config["denoise"]["median_size"])
    img_defense = gaussian_filter(img_defense, sigma=config["denoise"]["gaussian_sigma"])
    if is_thermal:
        # T通道：单通道，sigma_est是单个值
        sigma_est = restoration.estimate_sigma(img_defense, channel_axis=None)
        # 处理sigma_est为None的极端情况
        sigma_est = sigma_est if sigma_est is not None else 0.01
        img_defense = restoration.denoise_nl_means(
            img_defense, 
            h=config["denoise"]["nl_means_h"] * sigma_est,
            fast_mode=True, 
            patch_size=7, 
            patch_distance=11
        )
    else:
        # RGB通道：多通道，sigma_est是数组 → 取均值转为单个值
        sigma_est = restoration.estimate_sigma(img_defense, channel_axis=-1)
        # 处理sigma_est为None或数组的情况
        if sigma_est is None:
            sigma_est = 0.01
        elif isinstance(sigma_est, (np.ndarray, list, tuple)):
            sigma_est = np.mean(sigma_est)  # 核心修复：数组取均值
        img_defense = restoration.denoise_nl_means(
            img_defense, 
            h=config["denoise"]["nl_means_h"] * sigma_est,
            fast_mode=True, 
            patch_size=7, 
            patch_distance=11, 
            channel_axis=-1
        )

    # Step3: 随机数据增强
    if config["augment"]["random_flip"] and random.random() > 0.5:
        img_defense = np.fliplr(img_defense)
        anomaly_mask = np.fliplr(anomaly_mask)
    if config["augment"]["random_crop"]:
        crop_h = int(orig_h * (1 - config["augment"]["crop_ratio"]))
        crop_w = int(orig_w * (1 - config["augment"]["crop_ratio"]))
        start_h = random.randint(0, orig_h - crop_h)
        start_w = random.randint(0, orig_w - crop_w)
        img_defense = img_defense[start_h:start_h+crop_h, start_w:start_w+crop_w]
        anomaly_mask = anomaly_mask[start_h:start_h+crop_h, start_w:start_w+crop_w]
        img_defense = cv2.resize(img_defense, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        anomaly_mask = cv2.resize(anomaly_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Step4: 特征归一化
    if is_thermal:
        img_8bit = (img_defense * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=config["normalize"]["t_clahe_clip"], tileGridSize=(8,8))
        img_defense = clahe.apply(img_8bit).astype(np.float32) / 255.0
    else:
        mean = np.array(config["normalize"]["rgb_mean"]).reshape(1,1,3)
        std = np.array(config["normalize"]["rgb_std"]).reshape(1,1,3)
        img_defense = (img_defense - mean) / std
        img_defense = (img_defense - img_defense.min()) / (img_defense.max() - img_defense.min() + 1e-8)

    # Step5: 自适应修复
    if np.sum(anomaly_mask) > 0:
        if is_thermal:
            img_bgr = cv2.cvtColor((img_defense * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            img_repaired = cv2.inpaint(img_bgr, anomaly_mask, config["repair"]["inpaint_radius"], cv2.INPAINT_NS)
            img_defense = cv2.cvtColor(img_repaired, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            img_bgr = (img_defense * 255).astype(np.uint8)
            img_repaired = cv2.inpaint(img_bgr, anomaly_mask, config["repair"]["inpaint_radius"], cv2.INPAINT_NS)
            img_defense = img_repaired.astype(np.float32) / 255.0

    img_defense = np.clip(img_defense * 255, 0, 255).astype(np.uint8)
    return img_defense

def batch_defense_rgb():
    create_dirs()
    rgb_files = load_rgb_files(POISON_ROOT)
    total = len(rgb_files)
    success = 0

    print(f"\n===== 开始RGB防御（共{total}个文件） =====")
    for root, fil in rgb_files:
        img_path = os.path.join(root, fil)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  跳过无效RGB文件：{img_path}")
            continue
        
        img_clean = universal_image_defense(img, is_thermal=False)
        save_path = os.path.join(CLEAN_ROOT, "RGB", fil)
        cv2.imwrite(save_path, img_clean)
        success += 1
        print(f" [{success}/{total}] 防御完成：{fil}")

    print(f"\nRGB防御完成！成功处理 {success}/{total} 个文件")

def batch_defense_t():
    create_dirs()
    t_files = load_t_files(POISON_ROOT)
    total = len(t_files)
    success = 0

    print(f"\n===== 开始T防御（共{total}个文件） =====")
    for root, fil in t_files:
        img_path = os.path.join(root, fil)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"?  跳过无效T文件：{img_path}")
            continue
        
        img_clean = universal_image_defense(img, is_thermal=True)
        save_path = os.path.join(CLEAN_ROOT, "T", fil)
        cv2.imwrite(save_path, img_clean)
        success += 1
        print(f" [{success}/{total}] 防御完成：{fil}")

    print(f"\nT防御完成！成功处理 {success}/{total} 个文件")

if __name__ == "__main__":
    random.seed(12345)
    np.random.seed(12345)

    batch_defense_rgb()
    batch_defense_t()

    print(f"\n 防御完成！")
    print(f"   - 原始数据路径：{POISON_ROOT}")
    print(f"   - 防御后数据路径：{CLEAN_ROOT}")
