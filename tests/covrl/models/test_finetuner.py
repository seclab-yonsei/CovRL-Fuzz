import os
import pytest
import torch
import pandas as pd
from covrl.models.finetuner import FineTuner, CriticModel


# Mock configuration for testing
class MockConfig:
    load_path = "Salesforce/codet5p-220m"
    train_batch_size = 2
    eval_batch_size = 2
    fp16 = False
    learning_rate = 1e-5
    alpha = 0.1
    mask_probability = 0.15
    target_interpreter = "mock_interpreter"


@pytest.fixture
def setup_finetuner():
    # Initialize temporary directories for model and dataset paths
    config = MockConfig()
    device = torch.device("cpu")
    base_path = "./tests/resources/test_finetuner"
    finetuner = FineTuner(
        config=config,
        device=device,
        actor_path="Salesforce/codet5p-220m",
        critic_path="Salesforce/codet5p-220m",
        train_dataset_path=os.path.join(base_path, "train_dataset.json"),
        save_dir="base_path",
    )
    yield finetuner


def test_load_actor(setup_finetuner):
    finetuner = setup_finetuner
    actor = finetuner.load_actor(finetuner.actor_path)
    assert actor is not None, "Actor model should be loaded successfully"


def test_load_critic(setup_finetuner):
    finetuner = setup_finetuner
    finetuner.critic = finetuner.load_critic()
    assert finetuner.critic is not None, "Critic model should be loaded successfully"


def test_prepare_dataset(setup_finetuner):
    finetuner = setup_finetuner
    mock_data = {
        "is_orig": [True],
        "file_id": ["test"],
        "data": ["sample data"],
        "reward": [0.5],
    }
    actor_dataset = finetuner.prepare_dataset(
        pd.DataFrame(mock_data), finetuner.tokenizer, option="actor"
    )
    assert len(actor_dataset) == 1, "Dataset should have one item"

    critic_dataset = finetuner.prepare_dataset(
        pd.DataFrame(mock_data), finetuner.tokenizer, option="critic"
    )
    assert len(critic_dataset) == 1, "Dataset should have one item"


def test_custom_loss(setup_finetuner):
    finetuner = setup_finetuner

    # Initialize a mock critic, previous and current actor models
    cur_actor = finetuner.load_actor(finetuner.actor_path)

    prev_actor = finetuner.load_actor(finetuner.actor_path)
    critic = finetuner.load_critic()

    # Create mock inputs for testing loss computation
    inputs = {
        "input_ids": torch.randint(0, 32000, (1, 14)),
        "attention_mask": torch.ones((1, 14), dtype=torch.long),
        "decoder_input_ids": torch.randint(0, 32000, (1, 10)),
    }
    with torch.no_grad():
        cur_outputs = cur_actor(**inputs, labels=inputs["decoder_input_ids"])

        # Compute the custom PPO-style loss
        loss = finetuner.compute_actor_loss(cur_outputs, prev_actor, critic, inputs)

    assert loss is not None, "Custom loss should be computed successfully"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"


def test_train_critic(setup_finetuner):
    finetuner = setup_finetuner
    mock_data = {
        "is_orig": [True],
        "file_id": ["test"],
        "data": ["sample data"],
        "reward": [0.5],
    }

    critic_dataset = finetuner.prepare_dataset(
        pd.DataFrame(mock_data), finetuner.tokenizer, option="critic"
    )
    assert len(critic_dataset) == 1, "Dataset should have one item"
    finetuner.critic_path = finetuner.train_critic(critic_dataset, epochs=1)
    assert os.path.exists(
        finetuner.critic_path
    ), "Trained critic model should be saved successfully"
    critic_model = CriticModel().from_pretrained(finetuner.critic_path)
    assert (
        critic_model is not None
    ), "Failed to load the pretrained critic model from the specified path."


def test_finetune_actor(setup_finetuner):
    finetuner = setup_finetuner
    mock_data = {
        "is_orig": [True, True],
        "file_id": ["test1", "test1"],
        "data": ["sample data", "sample data"],
        "reward": [0.5, 0.0],
    }
    actor_dataset = finetuner.prepare_dataset(
        pd.DataFrame(mock_data), finetuner.tokenizer, option="actor"
    )

    finetuner.actor_path = finetuner.finetune_actor(actor_dataset, epochs=1)
    assert os.path.exists(
        finetuner.actor_path
    ), "Fine-tuned actor model should be saved successfully"
    test_actor = finetuner.load_actor(finetuner.actor_path)
    assert (
        test_actor is not None
    ), "Failed to load the pretrained actor model from the specified path."

# TODO: Add preprocess test