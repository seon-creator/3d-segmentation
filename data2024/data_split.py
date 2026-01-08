import os
import csv
import random
from collections import defaultdict
from math import floor


def stratified_split_by_labelset(
    h5_dir,
    label_info_csv,
    output_csv_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    """
    Stratified split by label-set distribution.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)

    # --------------------------------------------------
    # 1. Load label info (file_name, labels)
    # --------------------------------------------------
    cases = []  # (case_name, label_set_str)

    with open(label_info_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row["file_name"]
            label_set = row["labels"].replace(" ", "")
            cases.append((case, label_set))

    # --------------------------------------------------
    # 2. Group cases by label-set (strata)
    # --------------------------------------------------
    strata = defaultdict(list)
    for case, label_set in cases:
        strata[label_set].append(case)

    train_cases, val_cases, test_cases = [], [], []

    # --------------------------------------------------
    # 3. Stratified split per label-set
    # --------------------------------------------------
    for label_set, case_list in strata.items():
        random.shuffle(case_list)

        n = len(case_list)
        n_train = floor(n * train_ratio)
        n_val = floor(n * val_ratio)
        n_test = n - n_train - n_val  # remainder

        train_cases.extend(case_list[:n_train])
        val_cases.extend(case_list[n_train:n_train + n_val])
        test_cases.extend(case_list[n_train + n_val:])

    # --------------------------------------------------
    # 4. Save CSVs
    # --------------------------------------------------
    os.makedirs(output_csv_dir, exist_ok=True)

    def save_csv(cases, filename):
        path = os.path.join(output_csv_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["data_path", "case_name"])
            for case in sorted(cases):
                h5_path = os.path.join(h5_dir, f"{case}.h5")
                writer.writerow([h5_path, case])

    save_csv(train_cases, "train.csv")
    save_csv(val_cases, "val.csv")
    save_csv(test_cases, "test.csv")

    # --------------------------------------------------
    # 5. Summary
    # --------------------------------------------------
    print("Split summary")
    print(f"Train: {len(train_cases)}")
    print(f"Val  : {len(val_cases)}")
    print(f"Test : {len(test_cases)}")
    print(f"Total: {len(train_cases) + len(val_cases) + len(test_cases)}")

h5_dir = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/BraTS2024_PED"
label_info_csv = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/brats2024_ped.csv"

output_csv_dir = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/BraTS2024_PED_5fold_split"

stratified_split_by_labelset(
    h5_dir=h5_dir,
    label_info_csv=label_info_csv,
    output_csv_dir=output_csv_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)