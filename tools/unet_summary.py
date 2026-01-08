import yaml
import torch
from torchinfo import summary

from architectures.models.unet3d import UNet3D


# ======================================================
# Config loader
# ======================================================
def load_config(config_path):
    """YAML Config File Loader"""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ======================================================
# Model builder (UNet3D)
# ======================================================
def build_model_from_config(cfg):
    params = cfg["model_parameters"]

    model = UNet3D(
        in_channels=params["in_channels"],
        num_classes=params["num_classes"],
        base_channels=params.get("base_channels", 32),
        use_batchnorm=params.get("use_batchnorm", True),
    )

    return model


# ======================================================
# Params / FLOPs summary
# ======================================================
def print_model_summary(config_path):
    # 1. Load config
    cfg = load_config(config_path)

    # 2. Build Model
    model = build_model_from_config(cfg)
    model.eval()

    # 3. Input Shape
    roi = cfg["sliding_window_inference"]["roi"]  # e.g. [128,128,128]
    in_channels = cfg["model_parameters"]["in_channels"]

    input_size = (1, in_channels, roi[0], roi[1], roi[2])

    print("\n====================================")
    print(" UNet3D Model Summary (Config)")
    print("====================================")

    stats = summary(
        model,
        input_size=input_size,
        depth=4,
        col_names=[
            "kernel_size",
            "output_size",
            "num_params",
            "mult_adds",
        ],
        row_settings=["var_names"],
        verbose=0,
    )

    # -----------------------------
    # Params / FLOPs (formatted)
    # -----------------------------
    total_params_m = stats.total_params / 1e6
    total_flops_g = (stats.total_mult_adds * 2) / 1e9

    print("\n====================================")
    print(" Model Complexity")
    print("====================================")
    print(f"Parameters : {total_params_m:.2f} M")
    print(f"FLOPs      : {total_flops_g:.2f} G")
    print("====================================\n")


# ======================================================
# Run
# ======================================================
if __name__ == "__main__":
    CONFIG_PATH = (
        "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/experiments/brats_2018_21/unet3d/ex1/config.yaml"
    )

    print_model_summary(CONFIG_PATH)