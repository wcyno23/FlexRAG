from typing import Any
import os
import json
import pytz
from datetime import datetime
import sys
import string
import pathlib

def makedirs(path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path

def str_to_torch_dtype(dtype: str):
    import torch

    if dtype is None:
        return None
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unrecognized dtype: {dtype}")


def save_to_json(file: str, obj: Any):
    file_dir = os.path.dirname(file)
    os.makedirs(file_dir, exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def move_to_device(inputs, device):
    import torch

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def wrap_text(s):
    """Capitalize and add punctuation if there isn't."""
    s = s.strip()
    if not s[0].isupper():
        s = s[0].capitalize() + s[1:]
    if s[-1] not in string.punctuation:
        s += "."
    return s


def extract_file_name_and_extension(path: str):
    name, extension = os.path.splitext(path)
    return name, extension


class FileLogger:
    def __init__(self, log_file) -> None:
        self.log_file = log_file

    def log(self, metrics, **kwargs):
        with open(self.log_file, "a+") as f:
            # get current time
            tz = pytz.timezone("Asia/Shanghai")
            time = f"{'Time': <10}: {json.dumps(datetime.now(tz).strftime('%Y-%m-%d, %H:%M:%S'), ensure_ascii=False)}\n"
            print(time)
            command = f"{'Command': <10}: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}\n"
            print(command)
            metrics = f"{'Metrics': <10}: {json.dumps(metrics, ensure_ascii=False)}\n"
            msg = time + command

            for key, value in kwargs.items():
                x = f"{key: <10}: {json.dumps(value, ensure_ascii=False)}\n"
                print(x)
                msg += x
            msg += metrics
            print(metrics)
            f.write(str(msg) + "\n")
