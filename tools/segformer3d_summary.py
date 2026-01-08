import os
import yaml
import torch
from torchinfo import summary
from architectures.models.segformer3d import SegFormer3D


def load_config(config_path):
    """YAML Config File Loader"""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model_from_config(cfg):
    """(cfg) → SegFormer3D 모델 생성"""

    params = cfg["model_parameters"]

    model = SegFormer3D(
        in_channels=params["in_channels"],
        sr_ratios=params["sr_ratios"],
        embed_dims=params["embed_dims"],
        patch_kernel_size=params["patch_kernel_size"],
        patch_stride=params["patch_stride"],
        patch_padding=params["patch_padding"],
        mlp_ratios=params["mlp_ratios"],
        num_heads=params["num_heads"],
        depths=params["depths"],
        decoder_head_embedding_dim=params["decoder_head_embedding_dim"],
        num_classes=params["num_classes"],
        decoder_dropout=params["decoder_dropout"],
    )

    return model


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
    print(" SegFormer3D Model Summary (Config)")
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
        verbose=0
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


if __name__ == "__main__":
    CONFIG_PATH = (
        "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/experiments/brats_2018/segformer3d_mlp/base/config.yaml"
    )
    print_model_summary(CONFIG_PATH)