"""
To select the architecture based on a config file we need to ensure
we import each of the architectures into this file. Once we have that
we can use a keyword from the config file to build the model.
"""
######################################################################
def build_architecture(config):
    model_name = config["model_name"]

    # SegFormer3D + MLP decoder (baseline)
    if model_name == "segformer3d":
        from architectures.models.segformer3d import build_segformer3d_model
        model = build_segformer3d_model(config)
        return model

    elif model_name == "segformer3d_old":
        from architectures.segformer3d import build_segformer3d_model
        model = build_segformer3d_model(config)
        return model

    # SegFormer3D + UNet decoder
    elif config["model_name"] == "segformer3d_unet":
        from architectures.models.segformer3d_unet import build_segformer3d_unet_model
        return build_segformer3d_unet_model(config)

    # UNet3D
    elif model_name == "unet3d":
        from .unet3d import build_unet3d_model
        model = build_unet3d_model(config)
        return model

    else:
        raise ValueError(
            f"Unknown model_name={model_name}. Modify build_architecture.py to support it."
        )
