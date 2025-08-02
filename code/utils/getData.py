import os
import cv2
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from ultralytics import YOLO
import pandas as pd

LABEL_MAP = {
    "bosan": 0,
    "terdistraksi": 1,
    "fokus": 2,
    "mengantuk": 3,
    "menggunakan_ponsel": 4
}

class getData(Dataset):
    def __init__(self, frames_path="../../dataset/frames", labels_path="../../dataset", 
                 input_window=15, output_window=15, step=5, save_pkl=True, pkl_path="student_attention.pkl"):

        self.video_names = []
        self.video_to_sample_indices = {}
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.input_window = input_window
        self.output_window = output_window
        self.step = step
        self.save_pkl = save_pkl
        self.pkl_path = pkl_path

        self.src = []
        self.cls = []

        # Load both segmentation and pose estimation models
        self.segmentation_model = YOLO("../models/yolo11n-seg.pt")  # For person segmentation/masking
        self.pose_model = YOLO("../models/yolo11n-pose.pt")  # For pose estimation
        self._load_data()

        if save_pkl:
            with open(self.pkl_path, 'wb') as f:
                pickle.dump({
                    'src': self.src,
                    'cls': self.cls,
                    'video_names': self.video_names
                }, f)
            print(f"[INFO] Dataset saved to {self.pkl_path}")

    def _get_person_masks(self, frame):
        """Get person segmentation masks from YOLO segmentation"""
        results = self.segmentation_model(frame, save=False, verbose=False)
        masks = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None and result.boxes is not None:
                classes = result.boxes.cls.cpu().numpy()
                
                # Filter for person class (class 0 in COCO)
                person_indices = np.where(classes == 0)[0]
                
                for idx in person_indices:
                    # Get the segmentation mask for this person
                    mask = result.masks.data[idx].cpu().numpy()
                    # Resize mask to match frame dimensions
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    # Convert to binary mask
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    masks.append(mask)
        
        return masks

    def _mask_unwanted_persons(self, crop, crop_bbox, original_frame, person_masks):
        """Mask unwanted persons in the cropped image"""
        if len(person_masks) <= 1:
            return crop  # Only one person or no persons detected
        
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        crop_h, crop_w = crop.shape[:2]
        crop_area = crop_h * crop_w
        
        # Find which masks intersect with the crop area and calculate overlap ratios
        intersecting_info = []
        for i, mask in enumerate(person_masks):
            crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
            mask_area_in_crop = np.sum(crop_mask > 0)
            
            if mask_area_in_crop > 0:  # Has intersection
                # Calculate what percentage of the crop this mask covers
                coverage_ratio = mask_area_in_crop / crop_area
                
                # Calculate what percentage of this person's total body is in the crop
                total_person_area = np.sum(mask > 0)
                person_in_crop_ratio = mask_area_in_crop / max(total_person_area, 1)
                
                intersecting_info.append({
                    'mask_idx': i,
                    'crop_mask': crop_mask,
                    'area_in_crop': mask_area_in_crop,
                    'coverage_ratio': coverage_ratio,
                    'person_in_crop_ratio': person_in_crop_ratio
                })
        
        if len(intersecting_info) <= 1:
            return crop  # Only one person in crop area
        
        # Sort by person_in_crop_ratio descending - the main subject should have higher ratio
        intersecting_info.sort(key=lambda x: x['person_in_crop_ratio'], reverse=True)
        
        # The main subject is likely the one with highest person_in_crop_ratio
        # Mark others as unwanted if they have significantly lower ratios
        main_subject = intersecting_info[0]
        main_ratio = main_subject['person_in_crop_ratio']
        
        # Create a copy of crop to modify
        masked_crop = crop.copy()
        
        # Mask subjects that are likely unwanted (much lower person_in_crop_ratio)
        for info in intersecting_info[1:]:  # Skip the main subject
            # Only mask if this person has significantly less of their body in the crop
            # AND covers a reasonable amount of the crop (to avoid masking tiny overlaps)
            if (info['person_in_crop_ratio'] < main_ratio * 0.7 and 
                info['coverage_ratio'] > 0.05):  # At least 5% of crop area
                
                crop_mask = info['crop_mask']
                
                # Resize mask to match crop dimensions if needed
                if crop_mask.shape != (crop_h, crop_w):
                    crop_mask = cv2.resize(crop_mask, (crop_w, crop_h))
                
                # Apply masking - fill with background color
                mask_indices = np.where(crop_mask > 0)
                if len(mask_indices[0]) > 0:
                    mean_color = np.mean(masked_crop, axis=(0, 1))
                    masked_crop[mask_indices] = mean_color
        
        return masked_crop

    def _load_data(self):
        expected_videos = []

        for split in ["train", "test"]:
            anot_path = os.path.join(self.labels_path, f"{split}/anot_{split}.csv")
            if not os.path.exists(anot_path):
                print(f"[WARNING] Missing annotation file: {anot_path}")
                continue

            df = pd.read_csv(anot_path)
            for _, row in df.iterrows():
                video_path = row['video']
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                expected_videos.append((split, video_name))

        print(f"[DEBUG] Total expected videos: {len(expected_videos)}")

        processed_count = 0

        for split, video_name in expected_videos:
            start_idx = len(self.src)
            frame_dir = os.path.join(self.frames_path, split, video_name)
            label_file = os.path.join(self.labels_path, f"{split}/anot_{split}.csv")

            if not os.path.exists(frame_dir):
                print(f"[WARNING] Missing frame dir: {frame_dir}, skipping.")
                continue

            df = pd.read_csv(label_file)
            row = df[df['video'].str.contains(video_name)]
            if row.empty:
                print(f"[WARNING] No label for video: {video_name}, skipping.")
                continue

            try:
                row = row.iloc[0]
                box_info = json.loads(row["box"])
                if len(box_info) != 2:
                    print(f"[SKIP] Invalid box count in {video_name}")
                    continue

                subject_boxes = []
                subject_labels = []
                for box_data in box_info:
                    sequence = box_data["sequence"][0]
                    x = sequence["x"]
                    y = sequence["y"]
                    w = sequence["width"]
                    h = sequence["height"]
                    label = box_data["labels"][0]
                    label_id = LABEL_MAP.get(label, None)
                    if label_id is None:
                        continue
                    subject_boxes.append((x, y, w, h))
                    subject_labels.append(label_id)

                frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
                frame_files = [os.path.join(frame_dir, f) for f in frame_files]
                if len(frame_files) < self.input_window + self.output_window:
                    print(f"[SKIP] Not enough frames in {video_name}")
                    continue

                first_frame = cv2.imread(frame_files[0])
                H, W = first_frame.shape[:2]
                norm_boxes = []
                for (x, y, w, h) in subject_boxes:
                    x1 = int(x / 100 * W)
                    y1 = int(y / 100 * H)
                    x2 = int((x + w) / 100 * W)
                    y2 = int((y + h) / 100 * H)
                    norm_boxes.append((x1, y1, x2, y2))

                subject_keypoints = [[], []]
                failed_counts = [0, 0]

                for fidx, fpath in enumerate(frame_files):
                    frame = cv2.imread(fpath)
                    if frame is None:
                        continue
                    
                    # Step 1: Get person masks from the full frame
                    person_masks = self._get_person_masks(frame)
                    
                    for idx, (x1, y1, x2, y2) in enumerate(norm_boxes):
                        # Step 2: Crop the bounding box
                        crop = frame[y1:y2, x1:x2]
                        crop = cv2.resize(crop, (256, 256))
                        
                        # Step 3: Mask unwanted persons in the crop
                        crop_bbox = (x1, y1, x2, y2)
                        masked_crop = self._mask_unwanted_persons(crop, crop_bbox, frame, person_masks)
                        
                        # Step 4: Perform pose estimation on the masked crop
                        output = self.pose_model(masked_crop, save=False, verbose=False)

                        if output and output[0].keypoints is not None and len(output[0].keypoints.data) > 0:
                            kpt_data = output[0].keypoints.data[0].cpu().numpy()
                            kpts = [[kp[0] / 256, kp[1] / 256] for kp in kpt_data]

                            # Optional: save image sample if video in 1-6
                            if video_name in ["1", "2", "3", "4", "5", "6"] and fidx % 10 == 0:
                                drawn = masked_crop.copy()
                                for pt in kpt_data:
                                    cv2.circle(drawn, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                                out_dir = f"pose_samples/{video_name}/subject_{idx}"
                                os.makedirs(out_dir, exist_ok=True)
                                out_path = os.path.join(out_dir, os.path.basename(fpath))
                                cv2.imwrite(out_path, drawn)
                        else:
                            kpts = [[0.0, 0.0]] * 17
                            failed_counts[idx] += 1

                        subject_keypoints[idx].append(kpts)

                for idx in range(2):
                    kpts_list = subject_keypoints[idx]
                    label_id = subject_labels[idx]
                    full_video_name = f"{split}_{video_name}"
                    for i in range(0, len(kpts_list) - self.input_window - self.output_window + 1, self.step):
                        input_seq = kpts_list[i:i + self.input_window]
                        self.src.append(input_seq)
                        self.cls.append(label_id)
                        self.video_names.append(full_video_name)

                end_idx = len(self.src)
                self.video_to_sample_indices[full_video_name] = list(range(start_idx, end_idx))

                track_log = {
                    "total_frames": len(frame_files),
                    "subject_0_failed": failed_counts[0],
                    "subject_1_failed": failed_counts[1]
                }
                log_path = os.path.join(frame_dir, "track_log.json")
                with open(log_path, 'w') as f:
                    json.dump(track_log, f, indent=2)

                processed_count += 1
                print(f"[INFO] Processed {split}/{video_name} → Failed: {failed_counts}")

            except Exception as e:
                print(f"[ERROR] Failed to process {video_name}: {e}")
                continue

        print(f"[SUMMARY] Processed {processed_count}/{len(expected_videos)} videos")

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src[idx], dtype=torch.float32),
            torch.tensor(self.cls[idx], dtype=torch.float32)
        )

if __name__ == "__main__":
    dataset = getData()
    print(f"[DONE] Loaded dataset with {len(dataset)} sequences.")