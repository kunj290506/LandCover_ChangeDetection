import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import ChangeDetectionDataset
from models.snunet import SNUNet
from models.baselines import FCEF
from utils.metrics import get_metrics

def load_snunet(path, device):
    model = SNUNet(3, 1, use_attention=True)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except:
        print(f"Failed to load SNUNet from {path}")
        return None
    model.to(device).eval()
    return model

def load_fcef(path, device):
    model = FCEF(6, 1)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except:
        print(f"Failed to load FCEF from {path}")
        return None
    model.to(device).eval()
    return model

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Models
    snunet = load_snunet(args.snunet_path, device)
    fcef = load_fcef(args.fcef_path, device)
    
    # Dataset
    # Assuming test_list.txt exists from create_dummy_data
    if not os.path.exists('test_list.txt'):
        print("test_list.txt missing")
        return

    dataset = ChangeDetectionDataset(args.data_root, 'test_list.txt', mode='test', patch_size=256)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Comparing models on {len(dataset)} samples...")
    
    metrics_snu = {'f1': [], 'iou': []}
    metrics_fcef = {'f1': [], 'iou': []}
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            label = batch['label'].to(device)
            
            # SNUNet
            if snunet:
                out_snu = snunet(img1, img2)
                m_snu = get_metrics(out_snu, label)
                metrics_snu['f1'].append(m_snu['f1'])
                metrics_snu['iou'].append(m_snu['iou'])
                pred_snu = (torch.sigmoid(out_snu) > 0.5).cpu().numpy().squeeze()
            else:
                pred_snu = np.zeros((256, 256))
                m_snu = {'f1': 0, 'iou': 0}

            # FCEF
            if fcef:
                out_fcef = fcef(img1, img2)
                m_fcef = get_metrics(out_fcef, label)
                metrics_fcef['f1'].append(m_fcef['f1'])
                metrics_fcef['iou'].append(m_fcef['iou'])
                pred_fcef = (torch.sigmoid(out_fcef) > 0.5).cpu().numpy().squeeze()
            else:
                pred_fcef = np.zeros((256, 256))
                m_fcef = {'f1': 0, 'iou': 0}
                
            # Visualization (Save first 5)
            if i < 5:
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                # Input 1
                img1_vis = img1[0].cpu().permute(1, 2, 0).numpy()
                # Denormalize approx
                img1_vis = (img1_vis * 0.22) + 0.45 
                axes[0].imshow(np.clip(img1_vis, 0, 1))
                axes[0].set_title("Input T1")
                
                # GT
                axes[1].imshow(label[0].cpu().squeeze(), cmap='gray')
                axes[1].set_title("Ground Truth")
                
                # SNUNet
                axes[2].imshow(pred_snu, cmap='gray')
                axes[2].set_title(f"SNUNet (F1={m_snu['f1']:.2f})")
                
                # FCEF
                axes[3].imshow(pred_fcef, cmap='gray')
                axes[3].set_title(f"FCEF (F1={m_fcef['f1']:.2f})")
                
                plt.savefig(os.path.join(args.output_dir, f"comparison_{i}.png"))
                plt.close()

    print("--- Results ---")
    if snunet and metrics_snu['f1']:
        print(f"SNUNet Mean F1: {np.mean(metrics_snu['f1']):.4f}")
        print(f"SNUNet Mean IoU: {np.mean(metrics_snu['iou']):.4f}")
    if fcef and metrics_fcef['f1']:
        print(f"FCEF Mean F1: {np.mean(metrics_fcef['f1']):.4f}")
        print(f"FCEF Mean IoU: {np.mean(metrics_fcef['iou']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snunet_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--fcef_path', type=str, default='checkpoints/fcef_best.pth')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='output/comparison')
    args = parser.parse_args()
    main(args)
