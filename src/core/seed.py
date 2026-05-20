import os
import random
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = False if deterministic else True

        if deterministic:
            # 对 CUDA + CuBLAS 的严格确定性支持
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

            try:
                torch.use_deterministic_algorithms(True)
            except Exception as e:
                print(f"[seed] Warning: failed to enable full deterministic algorithms: {e}")
        else:
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass

    except ImportError:
        pass
