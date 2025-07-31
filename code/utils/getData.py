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
                 input_window=5, output_window=5, step=2, save_pkl=True, pkl_path="student_attention.pkl"):

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
        expected_folders = []

        #Collect expected frame folders from c1.json, c2.json, c3.json
        for cam in ["c1", "c2", "c3"]:
            json_path = os.path.join(self.labels_path, f"{cam}.json")
            if not os.path.exists(json_path):
                print(f"[WARNING] Missing expected video list: {json_path}")
                continue

            with open(json_path, "r") as f:
                video_entries = json.load(f)

            for entry in video_entries:
                video_path = entry["video"]
                video_rel = video_path.split("dataset/")[1].replace("/", os.sep)
                video_name = os.path.splitext(os.path.basename(video_rel))[0]
                expected_folders.append((cam, video_name))

        print(f"[DEBUG] Total expected frame folders: {len(expected_folders)}")

        processed_count = 0

        for cam, video_name in expected_folders:
            start_idx = len(self.src)
            frame_dir = os.path.join(self.frames_path, cam, video_name)
            label_file = os.path.join(self.labels_path, f"labels_{cam}.csv")

            if not os.path.exists(frame_dir):
                print(f"[WARNING] Missing frame dir: {frame_dir}, skipping.")
                continue
            if not os.path.exists(label_file):
                print(f"[WARNING] Missing label CSV for {cam}, skipping.")
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

                for fpath in frame_files:
                    frame = cv2.imread(fpath)
                    if frame is None:
                        continue
                    for idx, (x1, y1, x2, y2) in enumerate(norm_boxes):
                        crop = frame[y1:y2, x1:x2]
                        crop = cv2.resize(crop, (256, 256))
                        output = self.model(crop, save=False, verbose=False)
                        if output and output[0].keypoints is not None and len(output[0].keypoints.data) > 0:
                            kpt_data = output[0].keypoints.data[0]
                            kpts = [[kp[0] / 256, kp[1] / 256] for kp in kpt_data]
                        else:
                            kpts = [[0.0, 0.0]] * 17
                            failed_counts[idx] += 1
                        subject_keypoints[idx].append(kpts)

                for idx in range(2):
                    kpts_list = subject_keypoints[idx]
                    label_id = subject_labels[idx]
                    full_video_name = f"{cam}_{video_name}"
                    for i in range(0, len(kpts_list) - self.input_window - self.output_window + 1, self.step):
                        input_seq = kpts_list[i:i + self.input_window]
                        # Use scalar label instead of a list:
                        self.src.append(input_seq)
                        self.cls.append(label_id)               # <-- single scalar label here
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
                print(f"[INFO] Processed {cam}/{video_name} → Failed: {failed_counts}")

            except Exception as e:
                print(f"[ERROR] Failed to process {video_name}: {e}")
                continue

        print(f"[SUMMARY] Processed {processed_count}/{len(expected_folders)} videos")


    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src[idx], dtype=torch.float32),   # [5, 17, 2]
            torch.tensor(self.cls[idx], dtype=torch.float32)    # [5, 5]
        )

if __name__ == "__main__":
    dataset = getData()
    print(f"[DONE] Loaded dataset with {len(dataset)} sequences.")
