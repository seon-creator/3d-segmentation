import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from monai.metrics import DiceMetric
import csv  # CSV 저장용

from architectures.build_architecture import build_architecture
from dataloaders.build_dataset import build_dataset, build_dataloader
from post_processing import suppress_small_et

# 실험 폴더명 지정
EXPERIMENT = "brats_2018/segformer3d_mlp/ex1"
CSV_PATH = "result_new.csv"

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


# =======================================================
# DEFINE DICE METRIC
# =======================================================
# WT / TC / ET → 3 channel
dice_metric = DiceMetric(include_background=True, reduction="none")


print("\nRunning inference on test set (WT / TC / ET model)...")

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(testloader, total=len(testloader))):

        inputs = batch["image"].to(DEVICE)      # (B, C, H, W, D)
        labels = batch["label"].to(DEVICE)      # (B, 3, H, W, D)  ← 이미 WT/TC/ET

        # --------------------------------------------------
        # 1) Model forward
        # --------------------------------------------------
        outputs = model(inputs)                 # (B, 3, H, W, D)

        # --------------------------------------------------
        # 2) Prediction → binary mask
        # --------------------------------------------------
        # sigmoid + threshold (WT/TC/ET은 multi-label)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()           # (B, 3, H, W, D)

        # labels shape 보정
        if labels.ndim == 6 and labels.shape[1] == 1:
            labels = labels[:, 0]

        labels = labels.float()                 # (B, 3, H, W, D)

        # --------------------------------------------------
        # 3) Dice 누적
        # --------------------------------------------------
        dice_metric(y_pred=preds, y=labels)


# =======================================================
# AGGREGATE RESULTS
# =======================================================
dice_per_class = dice_metric.aggregate().cpu().numpy()  # (N, 3)
dice_metric.reset()

WT = float(np.nanmean(dice_per_class[:, 0]))
TC = float(np.nanmean(dice_per_class[:, 1]))
ET = float(np.nanmean(dice_per_class[:, 2]))
mean_dice = (WT + TC + ET) / 3.0

print("\n========== FINAL DICE (WT / TC / ET) ==========")
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