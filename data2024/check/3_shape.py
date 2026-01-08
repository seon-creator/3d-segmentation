import h5py


def print_h5_data_shapes(h5_path):
    """
    Print shapes of all datasets stored in a single .h5 file.
    """

    print(f"\n[H5 FILE] {h5_path}\n")

    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            data = f[key]
            print(f"{key}: shape = {data.shape}, dtype = {data.dtype}")

        print("\n[ATTRIBUTES]")
        for k, v in f.attrs.items():
            print(f"{k}: {v}")


h5_path = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/BraTS2024_PED/BraTS-PED-00001-000.h5"

print_h5_data_shapes(h5_path)