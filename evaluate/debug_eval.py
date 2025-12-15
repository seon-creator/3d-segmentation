import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from monai.metrics import DiceMetric
import csv  # CSV 저장용

from architectures.build_architecture import build_architecture
from dataloaders.build_dataset import build_dataset, build_dataloader

# 실험 폴더명 지정
EXPERIMENT = "brats_2018_21_all_v1/ex2"
CSV_PATH = "result_v1.csv"

# 프로젝트 루트 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # SegFormer3D/evaluate
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # SegFormer3D

# 자동 경로 생성
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments", EXPERIMENT)
CONFIG_PATH = os.path.join(EXPERIMENT_DIR, "config.yaml")
WEIGHT_PATH = os.path.join(EXPERIMENT_DIR, "weight", "pytorch_model.bin")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Dataset Name Mapping 함수
# -------------------------------
def map_dataset_name(dataset_type: str):
    mapping = {
        "brats2018_seg": "Brats2018",
        "brats_2018_seg": "Brats2018",
        "brats2021_seg": "Brats2021",
        "brats_2021_seg": "Brats2021",
        "brats2018_21_seg": "Brats2018+2021",
    }
    return mapping.get(dataset_type, dataset_type)


# 4-class → [WT, TC, ET] 매핑 함수
def convert_classes_to_wt_tc_et(label_4c: torch.Tensor):
    """
    label_4c: (B, H, W, D) 형태의 클래스 맵 (값: 0,1,2,3)
      - 0: Background
      - 1: NCR/NET
      - 2: ED
      - 3: ET

    반환:
      (B, 3, H, W, D)  → [WT, TC, ET]
    """
    # WT: 1, 2, 3
    wt = (label_4c == 1) | (label_4c == 2) | (label_4c == 3)
    # TC: 1, 3
    tc = (label_4c == 1) | (label_4c == 3)
    # ET: 3
    et = (label_4c == 3)

    wt = wt.unsqueeze(1)  # (B,1,H,W,D)
    tc = tc.unsqueeze(1)
    et = et.unsqueeze(1)

    return torch.cat([wt, tc, et], dim=1).float()  # (B,3,H,W,D)


# LOAD CONFIG
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# -------------------------------------------------------
# BUILD MODEL & LOAD WEIGHT
# -------------------------------------------------------
print("🔹 Loading model architecture...")
model = build_architecture(config)
state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print(f"Model loaded from: {WEIGHT_PATH}")

# -------------------------------------------------------
# BUILD TEST DATASET
# -------------------------------------------------------
print("🔹 Building test dataset...")
test_dataset_args = config["dataset_parameters"]["test_dataset_args"]
test_dataloader_args = config["dataset_parameters"]["test_dataloader_args"]

testset = build_dataset(
    dataset_type=config["dataset_parameters"]["dataset_type"],
    dataset_args=test_dataset_args,
)

testloader = build_dataloader(
    dataset=testset,
    dataloader_args=test_dataloader_args,
    config=config,
    train=False,
)
print(f"Test set size: {len(testset)} samples")


# DEFINE DICE METRIC
dice_metric = DiceMetric(include_background=True, reduction="none")


# INFERENCE & EVALUATION LOOP
print("\nRunning inference on test set...")

debug_done = False

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(testloader, total=len(testloader))):
        
        inputs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)  # 0,1,2,3 라벨 (shape: B,1,H,W,D 또는 B,H,W,D)

        # 1) 모델 출력 → 4-class 예측
        outputs = model(inputs)  # (B, 4, H, W, D) 가정 (각 채널: class logit)

        # multi-class argmax로 클래스 맵 생성
        preds_class = torch.argmax(outputs, dim=1)  # (B, H, W, D)

        # GT 라벨도 (B, H, W, D) 형태의 클래스 맵으로 맞추기
        if labels.ndim == 5 and labels.shape[1] == 1:
            labels_class = labels[:, 0].long()  # (B, H, W, D)
        else:
            labels_class = labels.long()        # 이미 (B,H,W,D) 인 경우

        # 2) 4-class → [WT, TC, ET] 매핑
        preds_wt_tc_et = convert_classes_to_wt_tc_et(preds_class)   # (B,3,H,W,D)
        lbls_wt_tc_et  = convert_classes_to_wt_tc_et(labels_class)  # (B,3,H,W,D)

        # DEBUG PRINT (FIRST BATCH)
        if not debug_done:
            print("\n========== DEBUG: first batch ==========")
            print("inputs shape            :", inputs.shape)
            print("outputs (logit) shape   :", outputs.shape)
            print("preds_class shape       :", preds_class.shape)
            print("labels_class shape      :", labels_class.shape)
            print("preds [WT,TC,ET] shape  :", preds_wt_tc_et.shape)
            print("labels [WT,TC,ET] shape :", lbls_wt_tc_et.shape)

            gt_np = lbls_wt_tc_et[0].cpu().numpy()
            pr_np = preds_wt_tc_et[0].cpu().numpy()

            gt_voxels = gt_np.sum(axis=(1, 2, 3))
            pr_voxels = pr_np.sum(axis=(1, 2, 3))

            print("\n[GT channel-wise voxel count] (WT, TC, ET)")
            for c, v in enumerate(gt_voxels):
                print(f"  GT channel {c} (0=WT,1=TC,2=ET): {int(v)} voxels")

            print("\n[PRED channel-wise voxel count] (WT, TC, ET)")
            for c, v in enumerate(pr_voxels):
                print(f"  PRED channel {c} (0=WT,1=TC,2=ET): {int(v)} voxels")

            eps = 1e-6
            print("\n[Manual Dice per channel (WT, TC, ET)]")
            for c in range(gt_np.shape[0]):
                inter = np.logical_and(gt_np[c] > 0.5, pr_np[c] > 0.5).sum()
                gt_sum = (gt_np[c] > 0.5).sum()
                pr_sum = (pr_np[c] > 0.5).sum()
                dice_c = (2 * inter) / (gt_sum + pr_sum + eps) if (gt_sum + pr_sum) > 0 else np.nan
                print(f"  channel {c}: Dice ≈ {dice_c:.4f}")

            debug_done = True

        # ------------------------------
        # 3) MONAI DiceMetric 누적
        # ------------------------------
        dice_metric(y_pred=preds_wt_tc_et, y=lbls_wt_tc_et)

# aggregate 결과
dice_per_class = dice_metric.aggregate().cpu().numpy()  # shape: (N, 3)  [WT,TC,ET]
dice_metric.reset()

# 🔧 Fix missing ET channel
if dice_per_class.shape[1] == 2:
    print("⚠️ Warning: ET channel was dropped. Padding ET=0 for all samples.")
    zeros = np.zeros((dice_per_class.shape[0], 1))
    dice_per_class = np.concatenate([dice_per_class, zeros], axis=1)

print("\n========== AGGREGATED DICE (WT, TC, ET) ==========")
print(dice_per_class)

WT = float(np.nanmean(dice_per_class[:, 0]))
TC = float(np.nanmean(dice_per_class[:, 1]))
ET = float(np.nanmean(dice_per_class[:, 2]))
mean_dice = (WT + TC + ET) / 3.0

print("\n========== FINAL DICE (mapped to [WT, TC, ET]) ==========")
print(f"Dice (WT): {WT:.4f}")
print(f"Dice (TC): {TC:.4f}")
print(f"Dice (ET): {ET:.4f}")
print(f"Mean Dice: {mean_dice:.4f}")

# =======================================================
# CSV 저장부
# =======================================================

model_name = config.get("model_name", "unknown_model")
loss_cfg = config.get("loss_fn", {})
loss_name = loss_cfg.get("loss_type", "unknown_loss")
optim_cfg = config.get("optimizer", {})
optimizer_name = optim_cfg.get("optimizer_type", "")

dataset_type = config["dataset_parameters"]["dataset_type"]
dataset_name = map_dataset_name(dataset_type)

# ⭐ test root 에서 마지막 폴더명만 추출
test_root_full = config["dataset_parameters"]["test_dataset_args"]["root"]   # "data/brats2018_seg"
test_folder_name = os.path.basename(test_root_full)
test_name = map_dataset_name(test_folder_name)

row = {
    "Dataset": dataset_name,
    "Test": test_name,
    "model": model_name,
    "loss": loss_name,
    "optimizer": optimizer_name,
    "WT": round(WT, 3),
    "TC": round(TC, 3),
    "ET": round(ET, 3),
    "Mean_dice": round(mean_dice, 3),
    "Experiment": EXPERIMENT,
}

csv_path = os.path.join(CURRENT_DIR, CSV_PATH)
file_exists = os.path.exists(csv_path)

with open(csv_path, mode="a", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Dataset", "Test", "model", "loss", "optimizer",
            "WT", "TC", "ET", "Mean_dice", "Experiment"
        ]
    )
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"\n📄 Results saved to: {csv_path}")