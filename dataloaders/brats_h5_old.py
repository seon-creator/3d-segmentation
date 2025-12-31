import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import os


class BratsH5VolumeDataset(Dataset):
    """
    HDF5 기반 BraTS volume dataset
    - image: (C, H, W, D)
    - mask:  (H, W, D)
    """

    def __init__(self, h5_dir, return_fg_coords=False):
        self.h5_dir = h5_dir
        self.files = sorted([
            os.path.join(h5_dir, f)
            for f in os.listdir(h5_dir)
            if f.endswith(".h5")
        ])
        self.return_fg_coords = return_fg_coords

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        h5_path = self.files[idx]

        with h5py.File(h5_path, 'r') as f:
            image = f['image'][()]   # (C, H, W, D)
            mask = f['mask'][()]     # (H, W, D)

            sample = {
                'image': torch.from_numpy(image).float(),
                'mask': torch.from_numpy(mask).long(),
            }

            if self.return_fg_coords and f.attrs.get('has_fg_coords', False):
                fg_coords = {}
                for cls in [1, 2, 3]:
                    key = f'fg_coords_{cls}'
                    if key in f:
                        fg_coords[cls] = torch.from_numpy(f[key][()])
                sample['fg_coords'] = fg_coords

        return sample


from torch.utils.data import DataLoader

dataset = BratsH5VolumeDataset(
    h5_dir="/home/work/3D_/processed_data/BRATS2024",
    return_fg_coords=True
)

loader = DataLoader(
    dataset,
    batch_size=1,      # 3D full volume → 보통 1
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

batch = next(iter(loader))
print(batch['image'].shape)  # (1, 4, H, W, D)
print(batch['mask'].shape)   # (1, H, W, D)

# mask
unique_labels = torch.unique(batch['mask'])
print("Unique labels in this volume:", unique_labels.tolist())