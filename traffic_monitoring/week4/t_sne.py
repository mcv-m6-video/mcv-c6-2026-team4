import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as T
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import ft_net

# ---------- CONFIGURATION ----------
AI_CITY_BASE_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/train/S01"
GT_FILE = "../data/AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt"
REID_WEIGHTS = "./src/net_19.pth"
OUTPUT_PLOT = "tsne_reid_s01.png"

CAMERAS = ["c001", "c002", "c003", "c004", "c005"]
NUM_IDS_TO_PLOT = 10     # Number of unique cars to visualize
MAX_CROPS_PER_ID = 30    # Prevent the plot from getting too crowded
# -----------------------------------

def load_ground_truth():
    """Loads GT and filters for the S01 sequence."""
    columns = ['CameraId', 'Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']
    gt = pd.read_csv(GT_FILE, header=None, names=columns)
    
    # AI City S01 cameras are usually 1 to 5
    gt = gt[gt['CameraId'].isin([1, 2, 3, 4, 5])]
    return gt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Re-ID Model
    print("Loading Re-ID Feature Extractor (IBN-a)...")
    reid_model = ft_net(class_num=34071, ibn=True, linear_num=2048, circle=True)
    reid_model.load_state_dict(torch.load(REID_WEIGHTS, map_location=device), strict=False)
    reid_model = reid_model.to(device)
    reid_model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Prepare Data
    gt_df = load_ground_truth()
    
    # Find IDs that appear in the most cameras to show cross-camera matching
    id_camera_counts = gt_df.groupby('Id')['CameraId'].nunique()
    best_ids = id_camera_counts.nlargest(NUM_IDS_TO_PLOT).index.tolist()
    
    filtered_gt = gt_df[gt_df['Id'].isin(best_ids)]

    features_list = []
    labels_id = []
    labels_cam = []

    # 3. Extract Features
    for cam_int in range(1, 6):
        cam_str = f"c00{cam_int}"
        video_path = os.path.join(AI_CITY_BASE_PATH, cam_str, "vdo.avi")
        
        cam_gt = filtered_gt[filtered_gt['CameraId'] == cam_int]
        if cam_gt.empty:
            continue
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}")
            continue

        print(f"Processing {cam_str}...")
        
        # Group by ID to sample evenly
        for car_id in best_ids:
            car_boxes = cam_gt[cam_gt['Id'] == car_id]
            if car_boxes.empty:
                continue
                
            # Sample random frames for this car in this camera
            sampled_boxes = car_boxes.sample(n=min(len(car_boxes), MAX_CROPS_PER_ID // 5), random_state=42)
            
            for _, row in sampled_boxes.iterrows():
                frame_id = int(row['FrameId']) - 1 # 0-indexed for cv2
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                
                if ret:
                    x, y, w, h = int(row['X']), int(row['Y']), int(row['Width']), int(row['Height'])
                    # Ensure coordinates are inside the frame
                    x, y = max(0, x), max(0, y)
                    crop = frame[y:y+h, x:x+w]
                    
                    if crop.size == 0: continue
                        
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    tensor = transform(crop_rgb).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        _, feat = reid_model(tensor)
                        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                        features_list.append(feat.cpu().numpy()[0])
                        labels_id.append(f"Car ID: {car_id}")
                        labels_cam.append(cam_str)
                        
        cap.release()

    if not features_list:
        print("No features extracted. Check paths and GT file.")
        return

    # 4. Compute t-SNE
    print("\nComputing t-SNE (this might take a moment)...")
    features_array = np.array(features_list)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_results = tsne.fit_transform(features_array)

    # 5. Plotting
    print("Generating Plot...")
    plot_data = pd.DataFrame({
        'tsne_1': tsne_results[:, 0],
        'tsne_2': tsne_results[:, 1],
        'Identity': labels_id,
        'Camera': labels_cam
    })

    plt.figure(figsize=(14, 10))
    # Color = Identity, Marker Style = Camera
    sns.scatterplot(
        x='tsne_1', y='tsne_2',
        hue='Identity',
        style='Camera',
        palette=sns.color_palette("tab10", NUM_IDS_TO_PLOT),
        data=plot_data,
        s=100,
        alpha=0.8
    )

    plt.title("t-SNE Visualization of IBN-a Re-ID Features (AI City S01)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot saved successfully as {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()