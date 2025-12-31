import os
import csv
import h5py
import torch
from torch.utils.data import Dataset

import torch
import torch.nn.functional as F

def fg_crop_or_pad_3d_with_origin(image, mask, cube_size):
    C, H, W, D = image.shape
    N = cube_size

    fg = mask > 0
    if fg.any():
        coords = fg.nonzero(as_tuple=False)
        h_min, w_min, d_min = coords.min(0)[0]
        h_max, w_max, d_max = coords.max(0)[0]
        center_h = (h_min + h_max) // 2
        center_w = (w_min + w_max) // 2
        center_d = (d_min + d_max) // 2
    else:
        center_h, center_w, center_d = H // 2, W // 2, D // 2

    h0 = center_h - N // 2
    w0 = center_w - N // 2
    d0 = center_d - N // 2

    h1, w1, d1 = h0 + N, w0 + N, d0 + N

    pad_h0 = max(0, -h0)
    pad_w0 = max(0, -w0)
    pad_d0 = max(0, -d0)

    pad_h1 = max(0, h1 - H)
    pad_w1 = max(0, w1 - W)
    pad_d1 = max(0, d1 - D)

    if any([pad_h0, pad_h1, pad_w0, pad_w1, pad_d0, pad_d1]):
        image = F.pad(image, (pad_d0, pad_d1, pad_w0, pad_w1, pad_h0, pad_h1))
        mask  = F.pad(mask.unsqueeze(0),
                      (pad_d0, pad_d1, pad_w0, pad_w1, pad_h0, pad_h1)).squeeze(0)

        h0 += pad_h0
        w0 += pad_w0
        d0 += pad_d0

    image = image[:, h0:h0+N, w0:w0+N, d0:d0+N]
    mask  = mask[h0:h0+N, w0:w0+N, d0:d0+N]

    return image, mask, (h0, w0, d0)

class BratsH5VolumeDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        cube_size=128,
        return_fg_coords=False,   # 🔹 복구
    ):
        self.root = root
        self.split = split
        self.cube_size = cube_size
        self.return_fg_coords = return_fg_coords

        csv_path = os.path.join(root, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        self.files = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.files.append(row["data_path"])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        h5_path = self.files[idx]

        with h5py.File(h5_path, "r") as f:
            image = torch.from_numpy(f["image"][()]).float()  # (C,H,W,D)
            mask  = torch.from_numpy(f["mask"][()]).long()   # (H,W,D)

            # 🔹 원본 fg_coords 읽기
            raw_fg_coords = None
            if self.return_fg_coords and f.attrs.get("has_fg_coords", False):
                raw_fg_coords = {}
                for cls in [1, 2, 3, 4]:
                    key = f"fg_coords_{cls}"
                    if key in f:
                        raw_fg_coords[cls] = torch.from_numpy(f[key][()])

        # --------------------------------------------------
        # fg-aware crop
        # --------------------------------------------------
        image, mask, crop_origin = fg_crop_or_pad_3d_with_origin(
            image, mask, self.cube_size
        )

        sample = {
            "image": image,  # (C,N,N,N)
            "mask": mask,   # (N,N,N)
        }

        # --------------------------------------------------
        # fg_coords 복구 + 좌표 변환
        # --------------------------------------------------
        if self.return_fg_coords and raw_fg_coords is not None:
            fg_coords = {}
            h0, w0, d0 = crop_origin  # crop 시작점

            for cls, coords in raw_fg_coords.items():
                # coords: (K,3) in original volume
                shifted = coords - torch.tensor([h0, w0, d0])
                valid = (
                    (shifted[:, 0] >= 0) & (shifted[:, 0] < self.cube_size) &
                    (shifted[:, 1] >= 0) & (shifted[:, 1] < self.cube_size) &
                    (shifted[:, 2] >= 0) & (shifted[:, 2] < self.cube_size)
                )
                if valid.any():
                    fg_coords[cls] = shifted[valid]

            sample["fg_coords"] = fg_coords

        return sample

# # dataset 생성
# dataset = BratsH5VolumeDataset(
#     root="data/brats2024_seg_all_v1",   # 👈 네 경로
#     split="train",
#     cube_size=128
# )

# print("Dataset length:", len(dataset))

# # 단일 샘플
# sample = dataset[0]

# image = sample["image"]
# mask  = sample["mask"]

# print("Image shape:", image.shape)  # (C,128,128,128)
# print("Mask shape :", mask.shape)   # (128,128,128)

# print("Image dtype:", image.dtype)
# print("Mask dtype :", mask.dtype)

# print("Unique mask labels:", torch.unique(mask))