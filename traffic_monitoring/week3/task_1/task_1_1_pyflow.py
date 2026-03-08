from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import cv2
import json

pyflow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(pyflow_path)
from src import utils

pyflow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pyflow'))
sys.path.append(pyflow_path)
import pyflow


KITTI_SEQ_45_IMGS = "../data/data_stereo_flow/training/image_0/000045_"
GT_FLOW_OCC = "../data/data_stereo_flow/training/flow_occ/000045_10.png"
GT_FLOW_NOC = "../data/data_stereo_flow/training/flow_noc/000045_10.png"
OUTPUT_FOLDER= "results/"
FAST = False

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    im1 = np.array(Image.open(f"{KITTI_SEQ_45_IMGS}10.png"))
    im1 = im1[:, :, np.newaxis] if len(im1.shape) == 2 else im1
    im2 = np.array(Image.open(f"{KITTI_SEQ_45_IMGS}11.png"))
    im2 = im2[:, :, np.newaxis] if len(im2.shape) == 2 else im2
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7 if not FAST else 1
    nInnerFPIterations = 1
    nSORIterations = 30 if not FAST else 1
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    # Inference
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    total_time = e - s
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        total_time, im1.shape[0], im1.shape[1], im1.shape[2]))
    estimated_flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    np.save(f'{OUTPUT_FOLDER}outFlow.npy', estimated_flow)
    
    # Evaluate against GT
    msen, pepn = utils.evaluate_flow(estimated_flow, GT_FLOW_NOC)
    print(f"MSEN: {msen}, PEPN: {pepn}%")
    
    results = {
        "MSEN": msen,
        "PEPN": pepn,
        "Time": total_time
    }
    
    with open(f'{OUTPUT_FOLDER}{"fast" if FAST else "default"}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Fill HSV image based on flow magnitude and direction
    h, w, _=im1.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 1] = 255 # Saturation set to maximum
    mag, ang = cv2.cartToPolar(estimated_flow[..., 0], estimated_flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2 # Hue represents direction
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # Value represents magnitude
    
    # Convert and save
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f'{OUTPUT_FOLDER}outFlow_fast.png', rgb)

if __name__ == "__main__":
    main()