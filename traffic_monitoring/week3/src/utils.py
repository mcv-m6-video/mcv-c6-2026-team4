import cv2
import numpy as np
import os

def evaluate_flow(estimated_flow, gt_path, threshold=3.0):
    # Load GT: -1 flag loads 16-bit correctly
    gt_img = cv2.imread(gt_path, -1)
    
    # Extract validity and components
    # Note: cv2 loads as BGR. KITTI maps are: B=u, G=v, R=valid
    gt_u = (gt_img[:, :, 2].astype(float) - 2**15) / 64.0
    gt_v = (gt_img[:, :, 1].astype(float) - 2**15) / 64.0
    gt_valid = gt_img[:, :, 0].astype(bool)

    # Filter only non-occluded (valid) pixels
    pred_u = estimated_flow[:, :, 0][gt_valid]
    pred_v = estimated_flow[:, :, 1][gt_valid]
    gt_u_valid = gt_u[gt_valid]
    gt_v_valid = gt_v[gt_valid]
    
    # Euclidean distance error per pixel
    error = np.sqrt((gt_u_valid - pred_u)**2 + (gt_v_valid - pred_v)**2)
    
    msen = np.mean(error) # Mean Square Error [cite: 172]
    pepn = (np.sum(error > threshold) / len(error)) * 100 # % Erroneous Pixels [cite: 172]
    
    return msen, pepn