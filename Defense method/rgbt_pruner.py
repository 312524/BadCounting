#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from models.fusion import CANNet
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

# ========== 1. Dataset structure ==========
# Use your dataset structure
ROOT_PATH = '/data_sda/hly88/cross-model/rgb_yuan'  # Modify as needed
CLEAN_DATA_DIR = "/data_sda/hly88/cross-model/rgb_yuan/test/"  # Clean data
TRIGGER_DATA_DIR = '/data_sda/hly88/cross-model/RGB_T/RGB_T_gt3_yuantu_6/test'  # Trigger data, path may need adjustment
CKPT_PATH = "/data_sda/hly88/cross-model/EAAI/train_result/1227-161129/best_model_37.pth"
PRUNE_RATIO = 0.02
BATCH_SIZE = 32
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPOCHS = 20
LR = 1e-4

# ========== 2. Data ==========
def random_crop(ht, wd, crop_size_h, crop_size_w):
    i = random.randint(0, ht - crop_size_h)
    j = random.randint(0, wd - crop_size_w)
    return i, j, crop_size_h, crop_size_w

class Crowd(Dataset):
    def __init__(self, root_path, crop_size=256,
                 downsample_ratio=8,
                 method='train'):

        self.root_path = root_path
        self.gt_list = sorted(glob.glob(os.path.join(self.root_path, '*.npy')))  # change to npy for gt_list
        if method not in ['train', 'val', 'test']:
            raise Exception("not implemented")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        self.RGB_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.407, 0.389, 0.396],
                std=[0.241, 0.246, 0.242]),
        ])
        self.T_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.492, 0.168, 0.430],
                std=[0.317, 0.174, 0.191]),
        ])

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, item):
        gt_path = self.gt_list[item]
        rgb_path = gt_path.replace('GT', 'RGB').replace('npy', 'jpg')
        t_path = gt_path.replace('GT', 'T').replace('npy', 'jpg')

        RGB = cv2.imread(rgb_path)[..., ::-1].copy()
        T = cv2.imread(t_path)[..., ::-1].copy()

        if self.method == 'train':
            keypoints = np.load(gt_path)
            return self.train_transform(RGB, T, keypoints)

        elif self.method == 'val' or self.method == 'test':  # TODO
            keypoints = np.load(gt_path)
            gt = keypoints
            k = np.zeros((T.shape[0], T.shape[1]))
            for i in range(0, len(gt)):
                if int(gt[i][1]) < T.shape[0] and int(gt[i][0]) < T.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            
            # Resize images
            RGB = cv2.resize(RGB, (224, 224))
            T = cv2.resize(T, (224, 224))
            k = cv2.resize(k, (28, 28))  # Adjust according to downsample ratio
            
            RGB = self.RGB_transform(RGB)
            T = self.T_transform(T)
            
            input = [RGB, T]
            count = float(np.sum(k))
            return input[0], input[1], count

        else:
            raise Exception("Not implemented")

    def train_transform(self, RGB, T, keypoints):
        ht, wd, _ = RGB.shape
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        RGB = RGB[i:i+h, j:j+w, :]
        T = T[i:i+h, j:j+w, :]
        keypoints = keypoints - [j, i]
        idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                   (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
        keypoints = keypoints[idx_mask]
        
        # Resize to standard size
        RGB = cv2.resize(RGB, (224, 224))
        T = cv2.resize(T, (224, 224))

        RGB = self.RGB_transform(RGB)
        T = self.T_transform(T)
        input = [RGB, T]
        count = float(len(keypoints))
        return input[0], input[1], count

def make_loader(data_dir, method='test', shuffle=False):
    dataset = Crowd(root_path=data_dir, method=method)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=4, pin_memory=True)

# ========== 3. Model ==========
def load_model():
    model = CANNet()
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    model.load_state_dict(ckpt)
    return model.to(DEVICE)

# ========== 4. Importance computation (fixed version) ==========
activation = {}
def get_activation(name):
    def hook(mod, inp, out):
        activation[name] = out.detach().mean(dim=(2, 3)).cpu()
    return hook

@torch.no_grad()
def compute_importance_fixed(model, loader_clean, loader_trig):
    
    # Register hooks
    handles = []
    conv_names = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_names.append(name)
            handles.append(module.register_forward_hook(get_activation(name)))
    
    if not conv_names:
        print("Warning: No Conv2d layers found in model!")
        return {}
    
    model.eval()
    
    def collect_activations(loader, desc):
        #Collect activations
        # Initialize dictionary
        acts = {name: [] for name in conv_names}
        
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
            try:
                # Unpack batch correctly
                if len(batch) == 3:
                    rgb, t, _ = batch
                else:
                    print(f"Warning: batch has {len(batch)} elements, expected 3")
                    continue
                
                rgb, t = rgb.to(DEVICE), t.to(DEVICE)
                
                # Clear activation dictionary
                activation.clear()
                
                # Forward pass
                _ = model([rgb, t])
                
                # Collect activations
                for name in conv_names:
                    if name in activation:
                        acts[name].append(activation[name])
                    else:
                        # If no activation for this layer, add zero tensor
                        if len(acts[name]) > 0:
                            # Create zero tensor with same shape as existing
                            zero_tensor = torch.zeros_like(acts[name][0])
                            acts[name].append(zero_tensor)
                
                # Print progress every few batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx+1} batches")
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Concatenate all batch activations
        result = {}
        for name in conv_names:
            if acts[name]:  # If there are activations
                result[name] = torch.cat(acts[name], dim=0)
            else:
                print(f"Warning: No activations collected for layer {name}")
                # Create empty placeholder
                result[name] = torch.zeros(0)
        
        return result
    
    print("Collecting clean activations...")
    clean_act = collect_activations(loader_clean, 'clean')
    
    print("Collecting trigger activations...")
    trig_act = collect_activations(loader_trig, 'trigger')
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Compute importance (safe version)
    importance = {}
    common_layers = set(clean_act.keys()) & set(trig_act.keys())
    
    print(f"\nFound {len(common_layers)} common layers between clean and trigger activations")
    
    for name in common_layers:
        try:
            clean_data = clean_act[name]
            trig_data = trig_act[name]
            
            # Check if data is valid
            if clean_data.numel() == 0 or trig_data.numel() == 0:
                print(f"Warning: Empty activations for layer {name}")
                importance[name] = torch.zeros(clean_data.shape[1] if clean_data.dim() > 1 else 1)
                continue
            
            # Ensure shapes match
            if clean_data.shape != trig_data.shape:
                print(f"Warning: Shape mismatch for layer {name}: clean={clean_data.shape}, trig={trig_data.shape}")
                # Take smaller shape
                min_samples = min(clean_data.shape[0], trig_data.shape[0])
                clean_data = clean_data[:min_samples]
                trig_data = trig_data[:min_samples]
            
            # Compute importance: difference between trigger and clean
            diff = (trig_data - clean_data).abs()
            
            # Average over batch dimension to get importance score per channel
            if diff.dim() > 1:
                importance_score = diff.mean(dim=0)  # Shape: [channels]
            else:
                importance_score = diff.mean().unsqueeze(0)
            
            importance[name] = importance_score
            
            # Print debug info
            if torch.isnan(importance_score).any() or torch.isinf(importance_score).any():
                print(f"Warning: Invalid values in importance for layer {name}")
            
        except Exception as e:
            print(f"Error computing importance for layer {name}: {e}")
            # Create default importance score
            if name in clean_act and clean_act[name].dim() > 1:
                channels = clean_act[name].shape[1]
                importance[name] = torch.zeros(channels)
            else:
                importance[name] = torch.zeros(1)
    
    if not importance:
        print("Error: No importance values computed!")
        return {}
    
    print(f"\nSuccessfully computed importance for {len(importance)} layers")
    
    # Print importance statistics
    print("\nImportance Statistics (top 10 layers by max importance):")
    importance_stats = []
    for name, imp in importance.items():
        if imp.numel() > 0:
            max_val = imp.max().item()
            mean_val = imp.mean().item()
            importance_stats.append((name, max_val, mean_val, imp.shape[0]))
    
    # Sort by max importance
    importance_stats.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, max_val, mean_val, channels) in enumerate(importance_stats[:10]):
        print(f"  {i+1:2d}. {name:40s} channels={channels:3d} max={max_val:.4f} mean={mean_val:.4f}")
    
    return importance

# ========== 5. Zero-out pruning (fixed version) ==========
def prune_model_zero_fixed(model, importance, ratio):
    #Pruning function
    keep_masks = {}
    total_zero, total = 0, 0
    
    print("\nStarting pruning process...")
    
    for name, module in model.named_modules():
        if name in importance and isinstance(module, nn.Conv2d):
            C = module.weight.shape[0]  # Number of output channels
            
            # Check if importance score dimensions match
            if len(importance[name]) != C:
                print(f"Warning: Importance dimension mismatch for {name}: "
                      f"importance has {len(importance[name])} values, but layer has {C} channels")
                # Adjust importance dimensions
                if len(importance[name]) < C:
                    # Pad with zeros
                    padded = torch.zeros(C)
                    padded[:len(importance[name])] = importance[name]
                    importance[name] = padded
                else:
                    # Truncate
                    importance[name] = importance[name][:C]
            
            total += C
            
            # ----------- Protection rules -----------
            protect = False
            reason = ""
            
            if any(k in name for k in ['reg_layer', 'lateral_conv',
                                        'aligned_r', 'aligned_t']):
                protect, reason = True, 'counting-related'
            elif C <= 64:
                protect, reason = True, 'C<=64'
            elif 'offset' in name or 'p_conv' in name:
                protect, reason = True, 'offset'
            elif 'frontend.0' in name or 'frontendT.0' in name:
                protect, reason = True, 'first'
            
            if protect:
                print(f'[protect] {name:35s}  C={C:3d}  {reason}')
                keep_masks[name] = torch.ones(C, dtype=torch.bool, device=DEVICE)
                continue
            # ---------------------------------
            
            # Calculate number of channels to prune
            k = max(1, int(C * ratio))
            
            # Sort importance scores and prune least important
            _, sort_idx = torch.sort(importance[name])
            zero_idx = sort_idx[:k]
            
            # Create mask
            mask = torch.ones(C, dtype=torch.bool, device=DEVICE)
            mask[zero_idx] = False
            keep_masks[name] = mask
            
            # Apply pruning
            with torch.no_grad():
                # Prune weights
                module.weight.data *= mask.view(-1, 1, 1, 1)
                
                # Prune bias if exists
                if module.bias is not None:
                    module.bias.data *= mask
            
            total_zero += k
            print(f'[zero  ] {name:35s}  {k:3d}/{C:3d}  ratio={k/C:.3f}')
    
    if total > 0:
        print(f'\nTotal zero-out: {total_zero}/{total}  ({total_zero/total:.2%})')
    else:
        print('\nNo layers were pruned!')
    
    return keep_masks

# ========== 6. Finetuning (fixed version) ==========
def finetune_fixed(model, keep_masks=None):
    #Fixed version of finetuning function
    #
    print("\nStarting finetuning...")
    
    # Create training data loader
    loader = make_loader(CLEAN_DATA_DIR, method='train', shuffle=True)
    
    # Freeze/unfreeze strategy
    # By default unfreeze all pruned layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if keep_masks and name in keep_masks:
                # Unfreeze pruned layers
                for param in module.parameters():
                    param.requires_grad = True
            else:
                # Freeze other layers
                for param in module.parameters():
                    param.requires_grad = False
    
    # Learnable scale parameter
    scale = nn.Parameter(torch.tensor(3., dtype=torch.float32, device=DEVICE))
    scale.requires_grad = True
    
    # Collect parameters to optimize
    trainable_params = []
    for param in model.parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    trainable_params.append(scale)
    
    if not trainable_params:
        print("Warning: No trainable parameters found!")
        return model, 1.0
    
    # Optimizer
    opt = torch.optim.Adam(trainable_params, lr=LR)
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    
    model.train()
    
    best_loss = float('inf')
    best_wts = None
    best_scale = 3.0  # Default value
    
    for epoch in range(EPOCHS):
        running_loss = 0.
        batch_count = 0
        
        for batch in tqdm(loader, desc=f'FT Epoch {epoch+1}/{EPOCHS}'):
            try:
                if len(batch) == 3:
                    rgb, t, gt = batch
                else:
                    print(f"Warning: batch has {len(batch)} elements, skipping")
                    continue
                
                rgb, t, gt = rgb.to(DEVICE), t.to(DEVICE), gt.to(DEVICE)
                gt = gt.float().to(DEVICE)
                
                # Forward pass
                pred = model([rgb, t])
                
                # Calculate predicted count
                count_pred = pred.view(pred.size(0), -1).sum(1) * scale
                
                # Calculate loss
                loss = loss_fn(count_pred, gt)
                
                # Backward pass
                opt.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                opt.step()
                
                running_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            scheduler.step()
            
            print(f'Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.4f}  Scale: {scale.item():.2f}  '
                  f'LR: {opt.param_groups[0]["lr"]:.2e}')
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_wts = {k: v.cpu() for k, v in model.state_dict().items()}
                best_scale = scale.item()
        else:
            print(f'Epoch {epoch+1}/{EPOCHS}  No valid batches processed')
    
    # Restore best weights
    if best_wts is not None:
        model.load_state_dict(best_wts)
        print(f'Best finetune loss: {best_loss:.4f}  Best scale: {best_scale:.2f}')
    else:
        print('No valid weights saved during finetuning')
    
    return model, best_scale

# ========== 7. Evaluation function ==========
@torch.no_grad()
def evaluate_model(model, loader, scale=1.0):
    #Evaluate model performance
    #
    model.eval()
    total_mae = 0.0
    total_samples = 0
    
    for batch in tqdm(loader, desc='Evaluating'):
        try:
            if len(batch) == 3:
                rgb, t, gt = batch
            else:
                continue
            
            rgb, t, gt = rgb.to(DEVICE), t.to(DEVICE), gt.to(DEVICE)
            gt = gt.float()
            
            pred = model([rgb, t])
            count_pred = pred.view(pred.size(0), -1).sum(1) * scale
            
            total_mae += torch.abs(count_pred - gt).sum().item()
            total_samples += rgb.size(0)
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            continue
    
    if total_samples > 0:
        return total_mae / total_samples
    else:
        return float('inf')

# ========== 8. Main pipeline ==========
def main():
    print("=" * 80)
    print("RGB-T Model Pruning for Backdoor Defense")
    print("=" * 80)
    
    # Load model
    print("\n1. Loading model...")
    model = load_model()
    
    # Create data loaders
    print("\n2. Creating data loaders...")
    clean_loader = make_loader(CLEAN_DATA_DIR, method='test', shuffle=False)
    trig_loader = make_loader(TRIGGER_DATA_DIR, method='test', shuffle=False)
    
    print(f"   Clean samples: {len(clean_loader.dataset)}")
    print(f"   Trigger samples: {len(trig_loader.dataset)}")
    
    # Evaluate original model
    print("\n3. Evaluating original model...")
    orig_clean_mae = evaluate_model(model, clean_loader, scale=1.0)
    orig_trig_mae = evaluate_model(model, trig_loader, scale=1.0)
    
    print(f"   Original Clean MAE: {orig_clean_mae:.4f}")
    print(f"   Original Trigger MAE: {orig_trig_mae:.4f}")
    print(f"   Attack Success Ratio: {orig_trig_mae/orig_clean_mae:.2f}x")
    
    # Compute importance
    print("\n4. Computing importance scores...")
    importance = compute_importance_fixed(model, clean_loader, trig_loader)
    
    if not importance:
        print("Error: Failed to compute importance scores!")
        return
    
    # Prune
    print(f"\n5. Pruning model with ratio {PRUNE_RATIO:.3f}...")
    keep_masks = prune_model_zero_fixed(model, importance, PRUNE_RATIO)
    
    # Evaluate pruned model (before finetuning)
    print("\n6. Evaluating pruned model (before finetuning)...")
    pruned_clean_mae = evaluate_model(model, clean_loader, scale=1.0)
    pruned_trig_mae = evaluate_model(model, trig_loader, scale=1.0)
    
    print(f"   Pruned Clean MAE: {pruned_clean_mae:.4f} "
          f"({(pruned_clean_mae-orig_clean_mae)/orig_clean_mae*100:+.1f}%)")
    print(f"   Pruned Trigger MAE: {pruned_trig_mae:.4f} "
          f"({(pruned_trig_mae-orig_trig_mae)/orig_trig_mae*100:+.1f}%)")
    
    # Save pruned model (before finetuning)
    zero_path = CKPT_PATH.replace('.pth', f'_zero_{PRUNE_RATIO}.pth')
    torch.save({
        'model': model.state_dict(),
        'keep_masks': {k: v.cpu() for k, v in keep_masks.items()},
        'importance': {k: v.cpu() for k, v in importance.items()},
        'prune_ratio': PRUNE_RATIO,
        'metrics': {
            'orig_clean_mae': orig_clean_mae,
            'orig_trig_mae': orig_trig_mae,
            'pruned_clean_mae': pruned_clean_mae,
            'pruned_trig_mae': pruned_trig_mae
        }
    }, zero_path)
    print(f"\nSaved pruned model -> {zero_path}")
    
    # Finetune
    print("\n7. Finetuning pruned model...")
    model, final_scale = finetune_fixed(model, keep_masks)
    
    # Final evaluation
    print("\n8. Evaluating final model...")
    final_clean_mae = evaluate_model(model, clean_loader, scale=final_scale)
    final_trig_mae = evaluate_model(model, trig_loader, scale=final_scale)
    
    print(f"   Final Clean MAE: {final_clean_mae:.4f} "
          f"({(final_clean_mae-orig_clean_mae)/orig_clean_mae*100:+.1f}%)")
    print(f"   Final Trigger MAE: {final_trig_mae:.4f} "
          f"({(final_trig_mae-orig_trig_mae)/orig_trig_mae*100:+.1f}%)")
    
    # Compute defense effectiveness
    orig_attack_ratio = orig_trig_mae / orig_clean_mae
    final_attack_ratio = final_trig_mae / final_clean_mae
    defense_improvement = orig_attack_ratio / final_attack_ratio
    
    print(f"\n   Original Attack Ratio: {orig_attack_ratio:.2f}x")
    print(f"   Final Attack Ratio: {final_attack_ratio:.2f}x")
    print(f"   Defense Improvement: {defense_improvement:.2f}x "
          f"({(1-final_attack_ratio/orig_attack_ratio)*100:.1f}% reduction)")
    
    # Save final model
    final_path = CKPT_PATH.replace('.pth', f'_zero_{PRUNE_RATIO}_finetuned.pth')
    torch.save({
        'model': model.state_dict(),
        'keep_masks': {k: v.cpu() for k, v in keep_masks.items()},
        'scale': torch.tensor(final_scale),
        'prune_ratio': PRUNE_RATIO,
        'metrics': {
            'orig_clean_mae': orig_clean_mae,
            'orig_trig_mae': orig_trig_mae,
            'final_clean_mae': final_clean_mae,
            'final_trig_mae': final_trig_mae,
            'final_scale': final_scale,
            'defense_improvement': defense_improvement
        }
    }, final_path)
    
    print(f"\n{'='*80}")
    print(f"All done! Final model saved to: {final_path}")
    print(f"{'='*80}")
    
if __name__ == '__main__':
    main()

