import csv
import ast
from collections import Counter

def count_label_combinations(csv_path):
    """
    CSV 파일에서 labels 컬럼을 읽어
    동일한 라벨 조합별 개수를 집계하여 출력
    """

    counter = Counter()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 문자열 "[0, 1, 2, 3, 4]" → 실제 리스트로 변환
            labels = ast.literal_eval(row["labels"])

            # 순서 무관하게 동일 조합으로 취급하기 위해 tuple(sorted)
            key = tuple(sorted(labels))

            counter[key] += 1

    # =========================
    # 결과 출력
    # =========================
    for labels_tuple, count in counter.items():
        print(f"{list(labels_tuple)} : {count}개")

csv_path = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/check/brats2024_ped.csv"

count_label_combinations(csv_path)