import os
import numpy as np
import pandas as pd
import pytest
import tempfile
from covrl.models.rewarding import Rewarding
from covrl.utils.map_target_error import ErrorType


class MockConf:
    alpha = 0.6
    target_interpreter = "mock_interpreter"
    interpreter_path = "/mock/path/to/interpreter"


@pytest.fixture
def conf():
    return MockConf()


@pytest.fixture
def rewarding(conf):
    with tempfile.TemporaryDirectory() as temp_dir:
        reward_instance = Rewarding(conf, save_dir=temp_dir, bitmap_size=10)
        yield reward_instance


@pytest.fixture
def dataset():
    return pd.DataFrame(
        {
            "is_orig": [False, True, False],
            "file_id": ["file1", "file2", "file3"],
            "data": ["data1", "data2", "data3"],
            "bitmap": [np.zeros(10), np.ones(10), np.zeros(10)],
            "reward": [0, 0, 0],
        }
    )


def test_set_dir(rewarding):
    with tempfile.TemporaryDirectory() as new_dir:
        rewarding.set_dir(new_dir)
        assert rewarding.save_dir == new_dir


def test_save_load_embedding(rewarding):
    rewarding.idf = [1] * 10
    rewarding.save_embedding()
    rewarding.idf = [0] * 10
    rewarding.load_embedding()
    assert rewarding.idf == [1] * 10


def test_update_idf(rewarding, dataset):
    rewarding.update_idf(dataset)
    assert np.all(np.array(rewarding.idf) != 0)


def test_check_validity(rewarding):
    param = ("./test_dir", "/mock_path", {"file_id": "testfile", "data": "content"})
    result = rewarding.check_validity(param)
    assert isinstance(result, tuple)
    assert result[0] is None or isinstance(result[0], dict)
    assert result[1] in [None, False, True, ErrorType.SYNTAX_ERROR.value]


def test_calculate_tfidf(rewarding):
    rewarding.idf = np.array([0.5, 1.2, 0.8, 0.3, 1.1, 0.0, 0.9, 1.5, 0.7, 1.3])
    data = np.ones(10)
    score = rewarding.calculate_tfidf(data)
    assert score > 0


def test_get_reward_with_errors(rewarding, dataset):
    # Set idf values so that rewards are more likely to be positive in normal cases
    rewarding.idf = np.array([0.5, 1.2, 0.8, 0.3, 1.1, 0.0, 0.9, 1.5, 0.7, 1.3])

    # Assign numpy arrays directly to the 'bitmap' column to avoid Series indexing issues
    dataset.at[0, "bitmap"] = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    dataset.at[1, "bitmap"] = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
    dataset.at[2, "bitmap"] = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1])

    # Simulate errors by manually setting negative reward values
    dataset.at[0, "reward"] = -1.0  # Major error, like syntaxError
    dataset.at[1, "reward"] = -0.5  # Minor error

    # Run the get_reward method
    updated_dataset = rewarding.get_reward(dataset)

    # Check that positive and negative rewards are both present
    assert (updated_dataset["reward"] > 0.5).any(), "No positive rewards found."
    assert (updated_dataset["reward"] < 0).any(), "No negative rewards found."
