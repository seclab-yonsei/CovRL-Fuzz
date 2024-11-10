import pytest
import torch
from covrl.models.inferencer import Inferencer


# Sample configuration class
class Config:
    target_interpreter = "v8"
    load_path = "Salesforce/codet5p-220m"
    mask_probability = 0.1
    total_batch_size = 8
    critic_epochs = 1
    actor_epochs = 1
    train_dataset_path = "./tests/resources/test_finetuner/train_dataset.json"
    train_batch_size = (1,)
    eval_batch_size = (1,)
    learning_rate = 1e-4


# Define configuration and sample paths
config = Config()
sample_model_path = "Salesforce/codet5p-220m"
predict_path = "dummy_path"  # Path for fine-tuning


@pytest.fixture
def inferencer():
    return Inferencer(config, model_path=sample_model_path, sample_method="contrastive")


# Test Model Loading
def test_load_model(inferencer):
    assert inferencer.model is not None, "Model was not loaded successfully."
    assert (
        inferencer.model.config.vocab_size == inferencer.tokenizer.vocab_size
    ), "Model and tokenizer vocab sizes do not match."


# Test Masking Function
@pytest.mark.parametrize(
    "input_ids, expected_mask_positions",
    [
        (
            [0, 1, 2, 3, 4, 5, 6],
            {3, 4},
        ),
    ],
)
def test_masking(inferencer, input_ids, expected_mask_positions):
    mask_dict, masked_input_ids, attention_mask = inferencer.masking(input_ids.copy())

    # Verify types and shapes
    assert isinstance(mask_dict, dict), "Mask dictionary is not created correctly."
    assert torch.is_tensor(masked_input_ids), "Masked input IDs are not a tensor."
    assert torch.is_tensor(attention_mask), "Attention mask is not a tensor."
    assert (
        masked_input_ids.shape == attention_mask.shape
    ), "Input IDs and attention mask shapes do not match."

    # Verify that the mask_dict keys correspond to positions in input_ids
    masked_positions = set(mask_dict.values())
    assert (
        masked_positions == expected_mask_positions
    ), f"Masked positions {masked_positions} do not match expected {expected_mask_positions}"

    # Check that the mask tokens are placed correctly in masked_input_ids
    for idx, pos in enumerate(masked_positions, 1):
        assert (
            masked_input_ids[0, pos] == inferencer.tokenizer.vocab_size - idx
        ), f"Position {pos} was not masked correctly."

    # Verify attention mask
    original_length = len(input_ids)
    assert torch.all(
        attention_mask[0, :original_length] == 1
    ), "Attention mask should be 1 for original tokens."
    assert torch.all(
        attention_mask[0, original_length:] == 0
    ), "Attention mask should be 0 for padding tokens."


# Test Sentence Splitting
@pytest.mark.parametrize("token_ids", [[0] * 100, [1] * 1000])
def test_split_sentences(inferencer, token_ids):
    splits = inferencer.split_sentences(token_ids)
    assert isinstance(splits, list), "Sentence splits should be a list."
    assert all(
        isinstance(split, list) for split in splits
    ), "Each split should be a list of token IDs."
    assert all(
        len(split) <= inferencer.split_length for split in splits
    ), "Each split exceeds the maximum split length."


# Test Reconstruction
@pytest.mark.parametrize(
    "token_ids, predictions, mask_dict, expected",
    [
        ([0, 1, 2, 3, 4, 5], [99, 12, 100, 11], {99: 1, 100: 4}, [0, 12, 2, 3, 11, 5]),
        (
            [10, 11, 12, 13],
            [99, 30, 10, 100, 51, 52, 55],
            {99: 0, 100: 2},
            [30, 10, 11, 51, 52, 55, 13],
        ),
        ([0, 1, 2, 3, 4, 5], [99, 100], {99: 1, 100: 4}, [0, 2, 3, 5]),
    ],
)
def test_reconstruct(inferencer, token_ids, predictions, mask_dict, expected):
    reconstructed = inferencer.reconstruct([token_ids], predictions, mask_dict)
    assert isinstance(
        reconstructed, list
    ), "Reconstructed output should be a list of token IDs."
    assert (
        len(reconstructed) <= inferencer.max_length
    ), "Reconstructed output exceeds max length."
    assert (
        reconstructed == expected
    ), f"Reconstructed output does not match expected output. Got {reconstructed}, expected {expected}."


# Test Inference
@pytest.mark.parametrize("inputs", [[1, 536, 1172, 67, 23711, 67, 18179, 13332, 3, 2]])
def test_inference(inferencer, inputs):
    result = inferencer.inference(inputs, num_samples=3)
    assert isinstance(result, list), "Inference output should be a list of token IDs."
    assert len(result) > 0, "Inference output should not be empty."
