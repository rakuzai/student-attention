import numpy as np
import torch
from torch.utils.data import Dataset

class StudentAttentionDataset(Dataset):
    def __init__(self, data_dict):
        self.src = data_dict["src"]               # [N, input_window, 17, 2]
        self.cls = data_dict["cls"]               # [N, output_window, num_classes]
        self.video_names = data_dict["video_names"]  # [N]
        self.video_to_sample_indices = self._map_video_to_indices()

    def _map_video_to_indices(self):
        video_map = {}
        for idx, name in enumerate(self.video_names):
            video_map.setdefault(name, []).append(idx)
        return video_map

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_tensor = torch.tensor(self.src[idx], dtype=torch.float32)
        cls_tensor = torch.tensor(self.cls[idx], dtype=torch.long)
        cls_tensor = cls_tensor.argmax() if cls_tensor.ndim > 0 else cls_tensor
        return src_tensor, cls_tensor