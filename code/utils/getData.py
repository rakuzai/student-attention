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

        self.model = YOLO("../models/yolo11n-pose.pt")
        self._load_data()

        if save_pkl:
            with open(self.pkl_path, 'wb') as f:
                pickle.dump({
                    'src': self.src,
                    'cls': self.cls,
                    'video_names': self.video_names
                }, f)
            print(f"[INFO] Dataset saved to {self.pkl_path}")

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
                    for idx, (x1, y1, x2, y2) in enumerate(norm_boxes):
                        crop = frame[y1:y2, x1:x2]
                        crop = cv2.resize(crop, (256, 256))
                        output = self.model(crop, save=False, verbose=False)

                        if output and output[0].keypoints is not None and len(output[0].keypoints.data) > 0:
                            kpt_data = output[0].keypoints.data[0].cpu().numpy()
                            kpts = [[kp[0] / 256, kp[1] / 256] for kp in kpt_data]

                            # Optional: save image sample if video in 1-6
                            if video_name in ["1", "2", "3", "4", "5", "6"] and fidx % 10 == 0:
                                drawn = crop.copy()
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
