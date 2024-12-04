import random

import numpy as np
import torch


# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def str_to_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def create_experiment_file_name(target_file: str, version: str, file_format="csv") -> str:
    """실험 결과 파일 명을 생성하는 함수

    Args:
        target_file (str): 실험에 사용한 파일
        version (str): _description_
        file_format (str, optional): _description_. Defaults to "csv".

    Returns:
        str: _description_
    """
    prefix = target_file.split("_")[0]
    return prefix + "_" + version + "." + file_format
