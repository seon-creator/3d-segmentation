import os
import csv
from pathlib import Path
from typing import List

# train, val, test 폴더가 있는 상위 폴더의 경로를 입력하면, 
# train.csv, val.csv, test.csv 로 경로 저장한 파일을 생성해주는 코드

def collect_h5_files(folder: str) -> List[str]:
    """폴더 하위의 .h5 파일 절대경로 수집"""
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".h5")
    ])


def write_csv(h5_paths: List[str], csv_path: str):
    """h5 파일 경로 목록을 csv로 저장"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["data_path", "case_name"])

        for p in h5_paths:
            case_name = Path(p).stem   # xxx.h5 → xxx
            writer.writerow([p, case_name])


def generate_brats_h5_csvs(
    dataset_root: str,
    save_dir: str,
    splits=("train", "val", "test"),
):
    """
    dataset_root/
      ├ train/
      ├ val/
      ├ test/
    구조에서 각 split별 csv 생성
    """

    dataset_root = os.path.abspath(dataset_root)
    save_dir = os.path.abspath(save_dir)

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] CSV save dir: {save_dir}")

    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] {split_dir} not found. Skipping.")
            continue

        h5_files = collect_h5_files(split_dir)
        csv_path = os.path.join(save_dir, f"{split}.csv")

        write_csv(h5_files, csv_path)

        print(f"[OK] {split}: {len(h5_files)} cases → {csv_path}")


# ---------------------------------------------------------------------
# CLI 실행 예시
# ---------------------------------------------------------------------
if __name__ == "__main__":
    DATASET_ROOT = "/home/work/3D_/processed_data/BRATS2024_5fold_splits/fold_0"
    SAVE_DIR = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/brats2024_seg_all_v1"

    generate_brats_h5_csvs(
        dataset_root=DATASET_ROOT,
        save_dir=SAVE_DIR,
        splits=("train", "val", "test"),
    )