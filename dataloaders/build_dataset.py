import sys

sys.path.append("../")

from typing import Dict
from monai.data import DataLoader
from augmentations.augmentations import build_augmentations


######################################################################
def build_dataset(dataset_type: str, dataset_args: Dict):
    """
    dataset_type 문자열에 따라 올바른 Dataset 클래스를 반환.
    """

    # BraTS 데이터셋
    if dataset_type == "brats_seg":
        from .brats import BratsDataset
        return BratsDataset(
            root_dir=dataset_args["root"],
            is_train=dataset_args["train"],
            transform=build_augmentations(dataset_args["train"]),
            fold_id=dataset_args.get("fold_id", None),
        )

    elif dataset_type == "brats_h5":
        from .brats_h5 import BratsH5VolumeDataset
        return BratsH5VolumeDataset(
            root=dataset_args["root"],
            split="train" if dataset_args.get("train", False) else "val",
            return_fg_coords=dataset_args.get("return_fg_coords", False),
        )
    else:
        raise ValueError(f"❌ Unsupported dataset type: {dataset_type}")


######################################################################
def build_dataloader(
    dataset, dataloader_args: Dict, config: Dict = None, train: bool = True
) -> DataLoader:
    """builds the dataloader for given dataset

    Args:
        dataset (_type_): _description_
        dataloader_args (Dict): _description_
        config (Dict, optional): _description_. Defaults to None.
        train (bool, optional): _description_. Defaults to True.

    Returns:
        DataLoader: _description_
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataloader_args["batch_size"],
        shuffle=dataloader_args["shuffle"],
        num_workers=dataloader_args["num_workers"],
        drop_last=dataloader_args["drop_last"],
        pin_memory=True,
    )
    return dataloader
