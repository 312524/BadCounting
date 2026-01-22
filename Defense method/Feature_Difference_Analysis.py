# -*- coding: GB2312 -*-
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from tqdm import tqdm


# Dataset root path (all _RGB.jpg / _T.jpg / .npy files are in this folder)
DATASET_ROOT = "The_other_backdoor_attack/Blended_Attack/train/"
# Feature extractor configuration
BACKBONE = "resnet50"
PCA_DIM = 64  # Dimension after PCA
# Anomaly detection thresholds
MAHALANOBIS_QUANTILE = 0.9
ISOLATION_FOREST_CONTAMINATION = "auto"

SUPPORTED_FORMATS = ('.jpg', '.npy')
RGB_SUFFIX = "_RGB.jpg"
T_SUFFIX = "_T.jpg"
# Device (auto-detect GPU / CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Manual Mahalanobis distance
def mahalanobis_distance_manual(x, mean_vec, cov_mat):
    try:
        cov_inv = np.linalg.inv(cov_mat + 1e-6 * np.eye(cov_mat.shape[0]))
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_mat)
    x_centered = x - mean_vec
    mahal = np.einsum('ni,ij,nj->n', x_centered, cov_inv, x_centered)
    return np.sqrt(mahal)

# 2. Feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name="resnet50"):
        super(FeatureExtractor, self).__init__()
        # Load pre-trained model
        if backbone_name == "resnet50":
            from torchvision.models import ResNet50_Weights
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.feature_layer = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 2048
        elif backbone_name == "resnet18":
            from torchvision.models import ResNet18_Weights
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.feature_layer = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze parameters (feature extraction only, no training)
        for param in self.feature_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.feature_layer(x)  # (batch, dim, 1, 1)
        return torch.flatten(features, 1)  # (batch, dim)

# 3. Load RGB-T file
def load_rgt_file(file_path, is_thermal=False):
    # Handle .npy files
    if file_path.endswith(".npy"):
        data = np.load(file_path)
        # Single channel to 3 channels
        if len(data.shape) == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        # Normalize to 0-255
        data = (data - data.min()) / (data.max() - data.min() + 1e-8) * 255
        data = data.astype(np.uint8)
    # Handle .jpg files
    else:
        if is_thermal:
            # T channel: single channel + contrast enhancement
            data = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if data is None:
                return None
            data = cv2.equalizeHist(data)
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        else:
            # RGB channel: color read
            data = cv2.imread(file_path)
            if data is None:
                return None
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    
    # Unified preprocessing (for ResNet)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(data).unsqueeze(0)  # Add batch dimension

# 4. Separate RGB / T files and extract features
def extract_rgt_features(dataset_root, extractor):
    # Step 1: walk directory and separate RGB / T files
    rgb_files = []  # (base_name, file_path)
    t_files = []    # (base_name, file_path)
    
    for root, dirs, files in os.walk(dataset_root):
        for file_name in files:
            if file_name.lower().endswith(SUPPORTED_FORMATS):
                file_path = os.path.join(root, file_name)
                # Match _RGB.jpg
                if file_name.endswith(RGB_SUFFIX):
                    base_name = file_name.replace(RGB_SUFFIX, "")
                    rgb_files.append((base_name, file_path))
                # Match _T.jpg
                elif file_name.endswith(T_SUFFIX):
                    base_name = file_name.replace(T_SUFFIX, "")
                    t_files.append((base_name, file_path))
                # Match _RGB.npy / _T.npy
                elif file_name.endswith(".npy"):
                    if "_RGB" in file_name:
                        base_name = file_name.replace("_RGB.npy", "")
                        rgb_files.append((base_name, file_path))
                    elif "_T" in file_name:
                        base_name = file_name.replace("_T.npy", "")
                        t_files.append((base_name, file_path))
    
    # Step 2: extract RGB features
    print(f"\nExtracting RGB features ({len(rgb_files)} files)...")
    rgb_features, rgb_base_names = [], []
    for base_name, file_path in tqdm(rgb_files):
        img_tensor = load_rgt_file(file_path, is_thermal=False)
        if img_tensor is None:
            print(f"Skipping invalid RGB file: {file_path}")
            continue
        with torch.no_grad():
            feature = extractor(img_tensor.to(DEVICE))
        rgb_features.append(feature.cpu().numpy().squeeze())
        rgb_base_names.append(base_name)
    
    # Step 3: extract T features
    print(f"\nExtracting T features ({len(t_files)} files)...")
    t_features, t_base_names = [], []
    for base_name, file_path in tqdm(t_files):
        img_tensor = load_rgt_file(file_path, is_thermal=True)
        if img_tensor is None:
            print(f"Skipping invalid T file: {file_path}")
            continue
        with torch.no_grad():
            feature = extractor(img_tensor.to(DEVICE))
        t_features.append(feature.cpu().numpy().squeeze())
        t_base_names.append(base_name)
    
    # Dimensionality reduction (handle empty)
    rgb_feat_mat = np.array(rgb_features) if len(rgb_features) > 0 else np.array([])
    t_feat_mat = np.array(t_features) if len(t_features) > 0 else np.array([])
    
    # RGB PCA
    if len(rgb_features) > 0 and PCA_DIM < rgb_feat_mat.shape[1]:
        rgb_feat_mat = PCA(n_components=PCA_DIM, random_state=12345).fit_transform(rgb_feat_mat)
    # T PCA
    if len(t_features) > 0 and PCA_DIM < t_feat_mat.shape[1]:
        t_feat_mat = PCA(n_components=PCA_DIM, random_state=12345).fit_transform(t_feat_mat)
    
    return rgb_feat_mat, rgb_base_names, t_feat_mat, t_base_names

# 5. Single-channel anomaly detection (output base name + anomaly flag only)
def detect_channel_anomalies(features_matrix, base_names, channel_name):
    if len(features_matrix) == 0:
        print(f"\nNo valid features for {channel_name} channel, skipping detection")
        return {}
    
    # Mahalanobis distance
    cov = EmpiricalCovariance().fit(features_matrix)
    mahal_dist = mahalanobis_distance_manual(features_matrix, cov.location_, cov.covariance_)
    mahal_thresh = np.quantile(mahal_dist, MAHALANOBIS_QUANTILE)
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=12345,
        n_estimators=100,
        bootstrap=True
    )
    iso_pred = iso_forest.fit_predict(features_matrix)
    
    # Build anomaly dict {base_name: is_anomaly}
    anomaly_dict = {}
    for i, base_name in enumerate(base_names):
        is_mahal_anomaly = mahal_dist[i] > mahal_thresh
        is_iso_anomaly = iso_pred[i] == -1
        anomaly_dict[base_name] = is_mahal_anomaly and is_iso_anomaly
    
    print(f"\n{channel_name} channel detection completed: anomalies = {sum(anomaly_dict.values())}")
    return anomaly_dict

# 6. Fuse two-channel results (keep base names only)
def fuse_rgt_anomalies(rgb_anomaly_dict, t_anomaly_dict):
    all_base_names = set(rgb_anomaly_dict.keys()).union(set(t_anomaly_dict.keys()))
    fused_anomalous = []  # anomaly sample base names only
    fused_normal = []     # normal sample base names only
    
    for base_name in all_base_names:
        rgb_anomaly = rgb_anomaly_dict.get(base_name, False)
        t_anomaly = t_anomaly_dict.get(base_name, False)
        
        if rgb_anomaly or t_anomaly:
            fused_anomalous.append(base_name)  # keep base name string only
        else:
            fused_normal.append(base_name)     # keep base name string only
    
    return fused_anomalous, fused_normal

# 7. Save results (base names only)
def save_results(fused_anomalous, fused_normal):
    # Save anomaly samples (suspected poisoned)
    with open("anomalous_samples.txt", 'w', encoding='gbk') as f:
        f.write("Anomaly sample base names\n")
        for base_name in fused_anomalous:
            f.write(f"{base_name}\n")
    
    # Save normal samples
    with open("normal_samples.txt", 'w', encoding='gbk') as f:
        f.write("Normal sample base names\n")
        for base_name in fused_normal:
            f.write(f"{base_name}\n")
    
    print(f"\nResults saved:")
    print(f"   - Anomaly samples: anomalous_samples.txt")
    print(f"   - Normal samples: normal_samples.txt")

# Main function (core pipeline)
if __name__ == "__main__":
    # 1. Initialize feature extractor
    print("="*50)
    print("Initializing feature extractor...")
    extractor = FeatureExtractor(BACKBONE).to(DEVICE)
    extractor.eval()
    
    # 2. Extract RGB / T features
    print("="*50)
    rgb_feat, rgb_base, t_feat, t_base = extract_rgt_features(DATASET_ROOT, extractor)
    
    # Empty check
    if len(rgb_feat) == 0 and len(t_feat) == 0:
        print("\nNo valid features extracted! Check dataset path / format")
        exit(1)
    
    # 3. Detect single-channel anomalies
    print("="*50)
    rgb_anomaly = detect_channel_anomalies(rgb_feat, rgb_base, "RGB")
    t_anomaly = detect_channel_anomalies(t_feat, t_base, "T")
    
    # 4. Fuse results
    print("="*50)
    fused_anomalous, fused_normal = fuse_rgt_anomalies(rgb_anomaly, t_anomaly)
    
    # 5. Output statistics
    print("\n" + "="*50)
    print("Final detection results:")
    total = len(fused_anomalous) + len(fused_normal)
    print(f"Total RGB-T samples: {total}")
    print(f"Anomaly samples (suspected poisoned): {len(fused_anomalous)}")
    print(f"Normal samples: {len(fused_normal)}")
    print(f"Anomaly ratio: {len(fused_anomalous)/total*100:.2f}%" if total > 0 else "No samples")
    
    # 6. Save results
    if total > 0:
        save_results(fused_anomalous, fused_normal)
    
    # 7. Print first 10 anomaly samples
    if len(fused_anomalous) > 0:
        print(f"\nFirst 10 anomaly sample base names:")
        for i, base_name in enumerate(fused_anomalous[:10]):
            print(f"{i+1}. {base_name}")
    else:
        print("\n No anomaly samples detected (clean data only)")

