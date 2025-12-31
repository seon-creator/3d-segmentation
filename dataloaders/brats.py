import os
import torch
import pandas as pd
from torch.utils.data import Dataset


class BratsDataset(Dataset):
    """
    BraTS segmentation dataset
    - 모달리티 2개, 4개 상관없이 로드 가능
    - 라벨은 4채널 (0,1,2,3)으로 학습
    - 전처리 과정에서 crop, normalize, orient 완료 후 .pt 파일로 저장된 상태
    - CSV 컬럼 구조: [data_path, case_name, grade]
    """

    def __init__(
        self,
        root_dir: str,
        is_train: bool = True,     # 학습/검증 여부 지정 (기본값: train)
        split: str = None,         # "train", "val", "test" 명시적 지정 가능
        transform=None,            # MONAI/TorchIO 등의 데이터 변환 함수 (optional)
        fold_id: int = None        # K-fold 수행 시 활용 (현재는 연동하지 않음 나중에 연동해야함)
    ):
        super().__init__()

        # ------------------------------------------------------------
        #  불러올 CSV 파일 이름 결정 로직 변경
        #   - 기존: validation.csv 사용 → 실제 전처리 코드에서는 val.csv로 저장됨
        #   - split 인자를 통해 train/val/test 명시 가능하도록 개선
        # ------------------------------------------------------------
        if split == "test":
            csv_name = "test.csv"
            
        elif fold_id is not None:
            if is_train:
                csv_name = f"train_fold_{fold_id}.csv"
            else:
                csv_name = f"val_fold_{fold_id}.csv"
        else:
            csv_name = "train.csv" if is_train else "val.csv"

        # CSV 파일 경로 구성
        csv_fp = os.path.join(root_dir, csv_name)
        if not os.path.exists(csv_fp):
            raise FileNotFoundError(f"CSV file not found: {csv_fp}")

        # CSV 읽기
        self.csv = pd.read_csv(csv_fp)
        self.transform = transform

    def __len__(self):
        # 데이터 개수 반환
        return len(self.csv)

    def __getitem__(self, idx):
        # ------------------------------------------------------------
        # CSV에서 한 케이스(row) 정보 읽기
        #   data_path: .pt 파일이 저장된 폴더 경로
        #   case_name: 케이스 이름 (예: Brats18_001)
        # ------------------------------------------------------------
        row = self.csv.iloc[idx]
        data_path = row["data_path"]
        case_name = row["case_name"]

        # ------------------------------------------------------------
        # 전처리된 .pt 텐서 파일 경로 구성
        #   - Flair + T1CE 가 결합된 modalities 텐서 경우: (2, H, W, D)
        #   - Flair + T1CE + T1 + T2 가 결합된 modalities 텐서 경우: (4, H, W, D)
        #   - 3채널(WT, TC, ET) 라벨 텐서: (3, H, W, D)
        #   - 4채널(0,1,2,3) 라벨 텐서: (4, H, W, D)
        # ------------------------------------------------------------
        volume_fp = os.path.join(data_path, f"{case_name}_modalities.pt")
        label_fp = os.path.join(data_path, f"{case_name}_label.pt")

        # ------------------------------------------------------------
        # torch.load로 바로 텐서 불러오기
        #   이미 torch.save로 저장되어 있기 때문에 nibabel이나 numpy 로딩 불필요
        # ------------------------------------------------------------
        volume = torch.load(volume_fp)   # shape: (4, H, W, D)
        label = torch.load(label_fp)     # shape: (3, H, W, D)

        # 혹시 numpy 배열로 저장된 경우를 대비해 변환
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume)
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label)

        # float32 형식으로 통일 (학습 중 손실 계산 안정성 확보)
        volume = volume.float()
        label = label.float()

        # ------------------------------------------------------------
        # transform 적용 (있는 경우)
        #   예: MONAI의 RandCropByPosNegLabeld, RandFlipd 등
        # ------------------------------------------------------------
        sample = {"image": volume, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)

        # 최종 반환 (딕셔너리 형태)
        return sample