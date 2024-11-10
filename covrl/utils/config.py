import json
from typing import NamedTuple


class Config(NamedTuple):
    testsuites: list = []
    save_dir: str = ""

    seed: int = 42
    n_samples: int = 32
    model_max_length: int = 1024
    mask_probability: float = 0.15
    total_batch_size: int = 512
    train_batch_size: int = 8
    eval_batch_size: int = 32
    learning_rate: float = 1e-4
    critic_epochs: int = 2
    actor_epochs: int = 1

    load_path: str = ""
    train_dataset_path: str = ""
    dev_dataset_path: str = ""
    target_interpreter: str = ""
    interpreter_path: str = ""

    alpha: float = 0.6
    beta: float = 1.0

    @classmethod
    def from_json(cls, file):  # load config from json file
        return cls(**json.load(open(file, "r")))
