import os
import pandas as pd

def merge_brats_datasets(
    brats2018_dir,
    brats2021_dir,
    save_dir
):
    """
    두 데이터셋 (2018 + 2021) 을 prefix 없이 합쳐서
    train.csv / val.csv / test.csv 생성
    """

    # 저장 폴더 생성
    os.makedirs(save_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        csv_2018 = os.path.join(brats2018_dir, f"{split}.csv")
        csv_2021 = os.path.join(brats2021_dir, f"{split}.csv")

        df_list = []

        # ========================================
        # 1) BRATS2018 불러오기 (prefix 없이)
        # ========================================
        if os.path.exists(csv_2018):
            df18 = pd.read_csv(csv_2018)

            # case_name 그대로 사용
            # data_path 도 그대로 (이미 절대경로 형태이면 OK)
            df_list.append(df18)

        # ========================================
        # 2) BRATS2021 불러오기 (prefix 없이)
        # ========================================
        if os.path.exists(csv_2021):
            df21 = pd.read_csv(csv_2021)

            df_list.append(df21)

        # ========================================
        # 3) 합치기
        # ========================================
        if df_list:
            merged = pd.concat(df_list, axis=0).reset_index(drop=True)
            out_csv = os.path.join(save_dir, f"{split}.csv")
            merged.to_csv(out_csv, index=False)

            print(f"✔ Saved merged {split}.csv ({len(merged)} samples)")
        else:
            print(f"⚠ Warning: No CSV found for split: {split}")


if __name__ == "__main__":
    merge_brats_datasets(
        brats2018_dir="/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/brats2018_seg_all",
        brats2021_dir="/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/brats2021_seg_all",
        save_dir="/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/brats2018_21_seg_all"
    )