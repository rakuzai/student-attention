import os
import cv2
import json
import pandas as pd

class ExtractFrames:
    def __init__(self, dataset_path="../../dataset", output_path="../../dataset/frames", target_fps=30, jpeg_quality=50):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality

        self.label_map = {
            "bosan": 0,
            "terdistraksi": 1,
            "fokus": 2,
            "mengantuk": 3,
            "menggunakan_ponsel": 4
        }

        self.annotations = {}
        os.makedirs(self.output_path, exist_ok=True)

    def load_labels(self):
        print("[INFO] Loading label CSVs...")
        for cam in ["c1", "c2", "c3"]:
            label_csv = os.path.join(self.dataset_path, f"labels_{cam}.csv")
            if not os.path.exists(label_csv):
                continue

            df = pd.read_csv(label_csv)
            for _, row in df.iterrows():
                video_path = row["video"].split("dataset/")[-1]
                cam_name = video_path.split("/")[0]
                video_name = os.path.basename(video_path)
                key = f"{cam_name}/{video_name}"

                box_json = row["box"].replace('""', '"').replace('"{', '{').replace('}"', '}')
                try:
                    box_data = json.loads(box_json)
                except Exception as e:
                    print(f"[ERROR] Failed to parse box for {key}: {e}")
                    continue

                if key not in self.annotations:
                    self.annotations[key] = []

                for item in box_data:
                    label = item["labels"][0]
                    for seq in item["sequence"]:
                        if not seq.get("enabled", False):
                            continue
                        self.annotations[key].append({
                            "x": seq["x"],
                            "y": seq["y"],
                            "w": seq["width"],
                            "h": seq["height"],
                            "label": label
                        })

    def draw_boxes(self, frame, boxes):
        h, w, _ = frame.shape
        for box in boxes:
            x1 = int(box["x"] / 100 * w)
            y1 = int(box["y"] / 100 * h)
            x2 = int((box["x"] + box["w"]) / 100 * w)
            y2 = int((box["y"] + box["h"]) / 100 * h)
            label = box["label"]

            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def write_yolo_labels(self, label_path, boxes, frame_width, frame_height):
        with open(label_path, "w") as f:
            for box in boxes:
                class_id = self.label_map.get(box["label"], -1)
                if class_id == -1:
                    continue

                x = box["x"] / 100 * frame_width
                y = box["y"] / 100 * frame_height
                w = box["w"] / 100 * frame_width
                h = box["h"] / 100 * frame_height

                x_center = (x + w / 2) / frame_width
                y_center = (y + h / 2) / frame_height
                w_rel = w / frame_width
                h_rel = h / frame_height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_rel:.6f} {h_rel:.6f}\n")

    def extract_frames(self):
        print("[INFO] Starting frame extraction...")
        for cam in ["c1", "c2", "c3"]:
            cam_path = os.path.join(self.dataset_path, cam)
            if not os.path.exists(cam_path):
                continue

            videos = [f for f in os.listdir(cam_path) if f.lower().endswith(".mp4")]
            for video in videos:
                video_path = os.path.join(cam_path, video)
                self._extract_video_frames(video_path, cam, video)

    def _extract_video_frames(self, video_path, cam, video_name):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        step = int(round(fps / self.target_fps))
        frame_idx = 0
        saved_idx = 0

        name_no_ext = os.path.splitext(video_name)[0]
        save_img_dir = os.path.join(self.output_path, cam, name_no_ext)
        os.makedirs(save_img_dir, exist_ok=True)

        key = f"{cam}/{video_name}"
        boxes = self.annotations.get(key, [])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                h, w, _ = frame.shape

                # draw and save annotated image
                if boxes:
                    self.draw_boxes(frame, boxes)

                img_path = os.path.join(save_img_dir, f"frame_{saved_idx:05d}.jpg")
                txt_path = os.path.join(save_img_dir, f"frame_{saved_idx:05d}.txt")

                cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])

                # write label txt in YOLO format
                self.write_yolo_labels(txt_path, boxes, w, h)

                saved_idx += 1

            frame_idx += 1

        cap.release()
        print(f"[{cam}] {video_name}: Saved {saved_idx} frames and labels to {save_img_dir}")

if __name__ == "__main__":
    extractor = ExtractFrames(dataset_path="../../dataset", output_path="../../dataset/frames")
    extractor.load_labels()
    extractor.extract_frames()
