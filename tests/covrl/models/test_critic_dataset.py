import pytest
import torch
from transformers import AutoTokenizer
from covrl.models.critic_dataset import CriticDataset


@pytest.fixture
def sample_dataset():
    return {
        "data": ["This is a test sentence.", "Another test sentence here."],
        "reward": [0.6, 0.4],
    }


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")


@pytest.fixture
def critic_dataset(sample_dataset, tokenizer):
    device = torch.device("cpu")
    return CriticDataset(device, sample_dataset, tokenizer)


def test_critic_dataset_length(critic_dataset, sample_dataset):
    assert len(critic_dataset) == len(
        sample_dataset["data"]
    ), "Dataset length does not match the expected value."


def test_critic_dataset_item_structure(critic_dataset):
    item = critic_dataset[0]

    assert isinstance(item, dict), "Returned item is not a dictionary."

    assert "input_ids" in item, "input_ids not found in returned item."
    assert "attention_mask" in item, "attention_mask not found in returned item."
    assert "labels" in item, "labels not found in returned item."


def test_critic_dataset_item_values(critic_dataset):
    item = critic_dataset[0]

    assert isinstance(item["input_ids"], torch.Tensor), "input_ids is not a tensor."
    assert isinstance(
        item["attention_mask"], torch.Tensor
    ), "attention_mask is not a tensor."
    assert isinstance(item["labels"], torch.Tensor), "labels is not a tensor."

    assert (
        item["input_ids"].shape == item["attention_mask"].shape
    ), "input_ids and attention_mask have different shapes."


def test_critic_dataset_label_conversion(critic_dataset):
    # Check if the score_to_label function works correctly
    item1 = critic_dataset[0]
    item2 = critic_dataset[1]

    assert item1["labels"].item() == 3, "Label for reward -0.6 should be 0."
    assert item2["labels"].item() == 2, "Label for reward 0.4 should be 2."


def test_critic_dataset_device(critic_dataset):
    item = critic_dataset[0]
    assert (
        item["input_ids"].device == critic_dataset.device
    ), "input_ids is not on the correct device."
    assert (
        item["attention_mask"].device == critic_dataset.device
    ), "attention_mask is not on the correct device."
    assert (
        item["labels"].device == critic_dataset.device
    ), "labels is not on the correct device."
