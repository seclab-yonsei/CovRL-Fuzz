import pytest
import tempfile
import json
from covrl.utils.config import Config  # Replace with the actual module name


# Test default values for each attribute
@pytest.mark.parametrize(
    "attribute, expected_value",
    [
        ("testsuites", []),
        ("save_dir", ""),
        ("seed", 42),
        ("n_samples", 32),
        ("model_max_length", 1024),
        ("mask_probability", 0.15),
        ("total_batch_size", 512),
        ("train_batch_size", 8),
        ("eval_batch_size", 32),
        ("learning_rate", 1e-4),
        ("critic_epochs", 2),
        ("actor_epochs", 1),
        ("load_path", ""),
        ("train_dataset_path", ""),
        ("dev_dataset_path", ""),
        ("target_interpreter", ""),
        ("interpreter_path", ""),
        ("alpha", 0.6),
        ("beta", 1.0),
    ],
)
def test_config_default_values(attribute, expected_value):
    config = Config()
    assert (
        getattr(config, attribute) == expected_value
    ), f"Expected {attribute} to be {expected_value}"


# Test loading from JSON file
@pytest.mark.parametrize(
    "config_data",
    [
        {
            "testsuites": ["suite1", "suite2"],
            "save_dir": "/path/to/save",
            "seed": 123,
            "n_samples": 64,
            "model_max_length": 512,
            "mask_probability": 0.25,
            "total_batch_size": 1024,
            "train_batch_size": 16,
            "eval_batch_size": 64,
            "learning_rate": 5e-5,
            "critic_epochs": 4,
            "actor_epochs": 2,
            "load_path": "/path/to/load",
            "train_dataset_path": "/path/to/train",
            "dev_dataset_path": "/path/to/dev",
            "target_interpreter": "python",
            "interpreter_path": "/usr/bin/python",
            "alpha": 0.7,
            "beta": 1.2,
        }
    ],
)
def test_config_from_json(config_data):
    # Create a temporary JSON file with configuration data
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as temp_file:
        json.dump(config_data, temp_file)
        temp_file_path = temp_file.name

    # Load config from JSON file
    config = Config.from_json(temp_file_path)

    # Verify that each attribute matches the values in config_data
    for attribute, expected_value in config_data.items():
        assert (
            getattr(config, attribute) == expected_value
        ), f"Expected {attribute} to be {expected_value}"
