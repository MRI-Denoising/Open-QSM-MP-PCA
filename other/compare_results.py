
from pathlib import Path
import numpy as np
from scipy.io import loadmat

base = Path("data/denoised")
folder1 = base / "magnitude"
folder3 = base / "magnitude_3core"

for noise in [10, 30, 50]:
    name = f"denoised_magnitude_{noise}_r3.mat"

    a = loadmat(folder1 / name)
    b = loadmat(folder3 / name)

    print(f"\n===== Noise {noise}% =====")

    keys_a = {k for k in a if not k.startswith("__")}
    keys_b = {k for k in b if not k.startswith("__")}

    print("Keys identical:", keys_a == keys_b)

    for key in sorted(keys_a & keys_b):
        x, y = a[key], b[key]

        if x.shape != y.shape:
            print(f"{key}: SHAPE DIFFERENT")
            continue

        if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            exact = np.array_equal(x, y, equal_nan=True)
            max_diff = np.max(np.abs(x - y)) if x.size else 0

            print(
                f"{key}: exact={exact}, "
                f"max_abs_diff={max_diff:.12g}"
            )
        else:
            print(f"{key}: exact={np.array_equal(x, y)}")
