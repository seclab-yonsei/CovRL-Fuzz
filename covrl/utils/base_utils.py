import os
import random
import signal
import torch
import numpy as np

from shutil import rmtree
from struct import iter_unpack, pack
from torch import cuda, manual_seed
from tqdm import tqdm
from transformers import AutoTokenizer


def score_to_label(score):
    label = 0
    if score < -0.5:
        label = 0
    elif score < 0:
        label = 1
    elif score <= 0.5:
        label = 2
    elif score <= 0.6:
        label = 3
    elif score <= 0.7:
        label = 4
    elif score <= 0.8:
        label = 5
    elif score <= 0.9:
        label = 6
    elif score <= 1.0:
        label = 7
    return label


label_to_score = {0: -1.0, 1: -0.5, 2: 0.5, 3: 0.6, 4: 0.7, 5: 0.8, 6: 0.9, 7: 1.0}


def extract_special_token_id(tokenizer_name, special_tokens=[";"]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    indices = {
        word: idx
        for word, idx in tokenizer.vocab.items()
        if any(token in word for token in special_tokens)
    }
    return sorted(list(indices.values()))


def setdir(dirpath: str, dirname: str = "", is_reset: bool = False):
    fullpath = os.path.join(dirpath, dirname)

    if is_reset and os.path.exists(fullpath):
        print(f"reset directory : {dirname}")
        rmtree(fullpath)

    os.makedirs(fullpath, exist_ok=True)
    return fullpath


def write(path: str, filename: str = None, content: str = None, mode: str = "wb"):
    filepath = os.path.join(path, filename) if filename else path

    # Convert content to bytes if it's a string
    if isinstance(content, str):
        content = content.encode()

    try:
        # Write content to the specified file
        with open(filepath, mode) as f:
            f.write(content)
        return os.path.abspath(filepath)
    except Exception as e:
        raise ValueError(f"An error occurred while writing to the file: {e}")


def read(filepath: str, mode: str = "rb"):
    with open(filepath, mode) as f:
        content = f.read()
    return content


def pool_map(pool, func, data):
    try:
        return tqdm(pool.imap_unordered(func, data), total=len(data))
    except KeyboardInterrupt:
        print("Process interrupted. Terminating pool...")
        pool.terminate()
        pool.join()
        print("Pool terminated.")
        # Optionally, forcefully kill the process group only if necessary
        try:
            os.killpg(os.getpid(), signal.SIGKILL)
        except Exception as e:
            print(f"Failed to kill process group: {e}")


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed_all(seed)


def hex_to_dec(hex_encoded):
    return [dec[0] for dec in iter_unpack("<H", hex_encoded)]


def dec_to_hex(token_ids):
    hex_encoded = b""
    for _id in token_ids:
        hex_encoded += pack("<H", _id)
    return hex_encoded


def get_device():
    # Determine if CUDA is available and set device accordingly
    is_cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda_available else "cpu")

    n_gpu = torch.cuda.device_count() if is_cuda_available else 0

    print(f"{device} ({n_gpu} GPUs)")
    return device


def load_testsuites(loadpath, format="", except_kw=[]):
    if isinstance(loadpath, str):
        loadpath = [os.path.abspath(loadpath)]
    elif isinstance(loadpath, list):
        loadpath = [os.path.abspath(path) for path in loadpath]

    files = []
    for path in loadpath:
        print(f"Using Dataset..{path}")
        for root, dir_names, file_names in tqdm(os.walk(path)):
            for f in file_names:
                file = os.path.join(root, f)
                if check_format(file, format) and not any(
                    kw in file for kw in except_kw
                ):
                    files.append(file)
    return files


def check_format(file, format):
    return file.endswith(format)


def is_error(target, stderr):
    stderr = stderr.strip()
    for target_err in target.keys():
        if target_err in stderr.decode():
            return target[target_err]
    return False
