import pytest
import torch
from transformers import AutoTokenizer
from covrl.models.actor_dataset import ActorDataset


@pytest.fixture
def sample_dataset():
    return {"data": ["def add(a, b): return a + b", "def subtract(a, b): return a - b"]}


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")


@pytest.fixture
def actor_dataset(sample_dataset, tokenizer):
    device = torch.device("cpu")
    return ActorDataset(device, sample_dataset, tokenizer)


def test_actor_dataset_length(actor_dataset, sample_dataset):
    assert len(actor_dataset) == len(
        sample_dataset["data"]
    ), "Dataset length does not match the expected value."


def test_actor_dataset_item_structure(actor_dataset):
    item = actor_dataset[0]

    assert isinstance(item, dict), "Returned item is not a dictionary."

    assert "input_ids" in item, "input_ids not found in returned item."
    assert "attention_mask" in item, "attention_mask not found in returned item."
    assert "labels" in item, "labels not found in returned item."


def test_actor_dataset_item_values(actor_dataset):
    item = actor_dataset[0]

    assert isinstance(item["input_ids"], torch.Tensor), "input_ids is not a tensor."
    assert isinstance(
        item["attention_mask"], torch.Tensor
    ), "attention_mask is not a tensor."
    assert isinstance(item["labels"], torch.Tensor), "labels is not a tensor."

    assert (
        item["input_ids"].shape == item["attention_mask"].shape
    ), "input_ids and attention_mask have different shapes."


def test_actor_dataset_device(actor_dataset):
    # Verify that each tensor is on the specified device
    item = actor_dataset[0]
    assert (
        item["input_ids"].device == actor_dataset.device
    ), "input_ids is not on the correct device."
    assert (
        item["attention_mask"].device == actor_dataset.device
    ), "attention_mask is not on the correct device."
    assert (
        item["labels"].device == actor_dataset.device
    ), "labels is not on the correct device."
