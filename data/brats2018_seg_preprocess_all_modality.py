import os
import torch
import nibabel
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
from monai.data import MetaTensor
from monai.transforms import Orientation, EnsureType


"""
BraTS 2018 version (수정)
- 모달리티: flair, t1, t1ce, t2 모두 사용
- HGG/LGG 각각 동일 split 비율 적용
"""

class ConvertToMultiChannelBasedOnBrats2018Classes(object):
    """
    BraTS2018 (0=BG, 1=NCR/NET, 2=ED, 3=ET)
    3채널: [WT, TC, ET]
    """
    def __call__(self, img):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        # WT, TC, ET 매핑
        WT = (img == 1) | (img == 2) | (img == 3)
        TC = (img == 1) | (img == 3)
        ET = (img == 3)
        return torch.stack([WT, TC, ET], dim=0) if isinstance(img, torch.Tensor) else np.stack([WT, TC, ET], axis=0)


class Brats2018Task1Preprocess:
    def __init__(
        self,
        root_dir: str,
        train_folder_name: str = "train",
        save_dir: str = "./brats2018_seg/BraTS2018_training_data",
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42,
    ):
        """
        root_dir: 데이터 루트 (예: /home/work/3D_/BT/BRATS2018)
        train_folder_name: 데이터 하위 폴더명 (예: MICCAI_BraTS_2018_Data_Training)
        save_dir: 전처리 결과 저장 폴더
        split 비율: train / val / test
        """

        self.train_folder_dir = os.path.join(root_dir, train_folder_name)
        assert os.path.exists(self.train_folder_dir), f"{self.train_folder_dir} not found"

        self.save_dir = save_dir
        self.grades = ["HGG", "LGG"]
        # ✅ 모든 모달리티 사용
        self.modalities = ["flair", "t1", "t1ce", "t2"]
        self.train_split, self.val_split, self.test_split = train_split, val_split, test_split
        np.random.seed(random_seed)

        # 전체 케이스 수집
        self.cases_by_grade = self._collect_cases_by_grade()

        # split
        self.split_indices = self._split_dataset()

    # ==========================================================
    # 데이터 수집 / 분할
    # ==========================================================
    def _collect_cases_by_grade(self):
        """HGG, LGG별로 seg 파일이 있는 케이스를 dict 형태로 수집"""
        cases_by_grade = {}
        for g in self.grades:
            grade_dir = os.path.join(self.train_folder_dir, g)
            if not os.path.isdir(grade_dir):
                continue
            grade_cases = []
            for case_name in sorted(os.listdir(grade_dir)):
                case_dir = os.path.join(grade_dir, case_name)
                if not os.path.isdir(case_dir):
                    continue
                seg_fp = self._find_modality_path(case_dir, case_name, "seg")
                if seg_fp is not None:
                    grade_cases.append((g, case_name))
            cases_by_grade[g] = grade_cases
        return cases_by_grade

    def _split_dataset(self):
        """
        각 grade(HGG, LGG)에 동일한 비율로 train/val/test split을 적용.
        """
        splits = {"train": [], "val": [], "test": []}

        for g, cases in self.cases_by_grade.items():
            n_total = len(cases)
            idxs = np.arange(n_total)
            np.random.shuffle(idxs)

            n_train = int(n_total * self.train_split)
            n_val   = int(n_total * self.val_split)
            n_test  = int(n_total * self.test_split)

            remainder = n_total - (n_train + n_val + n_test)
            for _ in range(remainder):
                if n_val <= n_test:
                    n_val += 1
                else:
                    n_test += 1

            if n_val != n_test:
                if n_val < n_test and n_train > 0:
                    n_val += 1
                    n_train -= 1
                elif n_test < n_val and n_train > 0:
                    n_test += 1
                    n_train -= 1

            diff = n_total - (n_train + n_val + n_test)
            if diff != 0:
                n_train += diff

            train_idx = idxs[:n_train]
            val_idx   = idxs[n_train:n_train + n_val]
            test_idx  = idxs[n_train + n_val:n_train + n_val + n_test]

            splits["train"].extend([cases[i] for i in train_idx])
            splits["val"].extend([cases[i] for i in val_idx])
            splits["test"].extend([cases[i] for i in test_idx])

            print(f"[{g}] total={n_total} → train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        print(f"\n✅ Total Split Sizes: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        return splits

    # ==========================================================
    # 유틸 함수
    # ==========================================================
    def _find_modality_path(self, case_dir: str, case_name: str, suffix: str):
        base = os.path.join(case_dir, f"{case_name}_{suffix}")
        for ext in (".nii", ".nii.gz"):
            fp = base + ext
            if os.path.exists(fp):
                return fp
        return None

    def get_modality_fp(self, grade: str, case_name: str, modality: str):
        case_dir = os.path.join(self.train_folder_dir, grade, case_name)
        fp = self._find_modality_path(case_dir, case_name, modality)
        assert fp is not None, f"Missing file: {case_name}_{modality}.nii[.gz]"
        return fp

    def normalize(self, x: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        return normalized_1D_array.reshape(x.shape)

    def orient(self, x: MetaTensor) -> MetaTensor:
        assert isinstance(x, MetaTensor)
        return Orientation(axcodes="RAS")(x)

    def detach_meta(self, x: MetaTensor) -> np.ndarray:
        assert isinstance(x, MetaTensor)
        return EnsureType(data_type="numpy", track_meta=False)(x)

    def crop_brats2021_zero_pixels(self, x: np.ndarray) -> np.ndarray:
        return x[:, 56:184, 56:184, 13:141]

    def load_nifti(self, fp):
        nifti_data = nibabel.load(fp)
        return nifti_data.get_fdata(), nifti_data.affine

    # ==========================================================
    # 전처리
    # ==========================================================
    def preprocess_brats_modality(self, data_fp: str, is_label: bool = False) -> np.ndarray:
        data, affine = self.load_nifti(data_fp)
        if is_label:
            data = data.astype(np.uint8)
            data[data == 4] = 3
            data = ConvertToMultiChannelBasedOnBrats2018Classes()(data)
        else:
            data = self.normalize(x=data)
            data = data[np.newaxis, ...]
        data = MetaTensor(x=data, affine=affine)
        data = self.orient(data)
        data = self.detach_meta(data)
        data = self.crop_brats2021_zero_pixels(data)
        return data

    # ==========================================================
    # 데이터셋 호출 / 저장
    # ==========================================================
    def _process_case(self, grade, case_name):
        """4모달리티 모두 불러오기"""
        modality_tensors = []
        for modality in self.modalities:
            fp = self.get_modality_fp(grade, case_name, modality)
            tensor = self.preprocess_brats_modality(fp, is_label=False).swapaxes(1, 3)
            modality_tensors.append(tensor)

        modalities = np.concatenate(modality_tensors, axis=0, dtype=np.float32)

        seg_fp = self.get_modality_fp(grade, case_name, "seg")
        Label = self.preprocess_brats_modality(seg_fp, is_label=True).swapaxes(1, 3)

        data_save_path = os.path.join(self.save_dir, case_name)
        os.makedirs(data_save_path, exist_ok=True)
        torch.save(modalities, os.path.join(data_save_path, f"{case_name}_modalities.pt"))
        torch.save(Label, os.path.join(data_save_path, f"{case_name}_label.pt"))

    # ==========================================================
    # 전체 실행
    # ==========================================================
    def __call__(self):
        print("Started preprocessing BraTS2018 (HGG/LGG, stratified split, 4 modalities)...")

        all_cases = [c for g_cases in self.cases_by_grade.values() for c in g_cases]
        with Pool(processes=os.cpu_count()) as pool:
            pool.starmap(self._process_case, all_cases)

        print("Finished preprocessing.")
        self._create_split_csv()

    # ==========================================================
    # CSV 생성
    # ==========================================================
    def _create_split_csv(self):
        """train/val/test split CSV 저장"""
        os.makedirs(self.save_dir, exist_ok=True)
        for split_name, cases in self.split_indices.items():
            rows = []
            for grade, case_name in cases:
                base_dir = os.path.join(self.save_dir, case_name)
                rows.append([base_dir, case_name, grade])
            df = pd.DataFrame(rows, columns=["data_path", "case_name", "grade"])
            df.to_csv(os.path.join(self.save_dir, f"{split_name}.csv"), index=False)
            print(f"Saved {split_name}.csv ({len(df)} cases)")


if __name__ == "__main__":
    brats2018_task1_prep = Brats2018Task1Preprocess(
        root_dir="/home/work/3D_/BT/BRATS2018",
        train_folder_name="MICCAI_BraTS_2018_Data_Training",
        save_dir="/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/brats2018_seg_all/BraTS2018_training_data",
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
    )

    brats2018_task1_prep()