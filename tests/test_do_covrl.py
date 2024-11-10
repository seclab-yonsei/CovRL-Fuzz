import os
import tempfile

import pytest
from do_covrl import decode_data, mask_mutation
from transformers import AutoTokenizer


# Sample configuration class
class MockConfig:
    n_samples = 5


# Dummy model class with a simple inference method
class DummyModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")

    def inference(self, data, num_samples):
        # Just returns a sequence based on num_samples for predictable testing
        return data


@pytest.fixture
def conf():
    return MockConfig()


@pytest.fixture
def model():
    return DummyModel()


def test_mask_mutation(conf, model):
    with tempfile.TemporaryDirectory() as predict_path:
        # Create a temporary prediction file in the temporary directory
        predict_file = os.path.join(predict_path, "MLM_pred")

        # Write a sample binary file
        with open(predict_file, "wb") as f:
            f.write(b"\x00\x01\x00\x02\x00\x04")  # Binary data for testing

        # Run mask_mutation and capture the result
        result = mask_mutation(conf, model, predict_path)
        assert os.path.exists(predict_file), "Prediction file was not created."
        with open(predict_file, "rb") as f:
            assert (
                f.read() == b"\x00\x01\x00\x02\x00\x04"
            ), "File content does not match expected hex-encoded output."
        assert result == 6, f"Expected output length 6, got {result}"


def test_decode_data(model):
    with tempfile.TemporaryDirectory() as prediction_path:
        # Create a temporary prediction file for testing
        predict_file = os.path.join(prediction_path, "MLM_decoded")

        # Write a sample binary file with data for testing
        with open(predict_file, "wb") as f:
            f.write(
                b"\x18\x02\x94\x04C\x00\x9f\\C\x00\x03G\x144"
            )  # Binary data for testing

        # Run decode_data and capture the result
        result = decode_data(model, prediction_path)

        assert os.path.exists(predict_file), "Prediction file was not created."
        with open(predict_file, "rb") as f:
            assert (
                f.read() == b"def print_hello_world():"
            ), "Decoded file content does not match expected output."
        assert result == 24, f"Expected output length 3, got {result}"
