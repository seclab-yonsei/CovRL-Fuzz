import pytest
import torch
from covrl.models.critic import (
    CriticModel,
)

@pytest.fixture
def model():
    # Initialize the model with a small configuration for testing
    model_path = "Salesforce/codet5p-220m"
    num_labels = 8
    return CriticModel(model_path=model_path, num_labels=num_labels)


def test_forward_inference(model):
    # Prepare dummy input data for inference
    batch_size = 2
    sequence_length = 10
    input_ids = torch.randint(
        0, 32100, (batch_size, sequence_length)
    )
    attention_mask = torch.ones((batch_size, sequence_length))

    # Run forward pass without labels (inference mode)
    logits = model(input_ids=input_ids, attention_mask=attention_mask)

    # Check output shape
    assert logits.shape == (
        batch_size,
        model.classifier[-1].out_features,
    ), f"Expected logits shape {(batch_size, model.classifier[-1].out_features)}, but got {logits.shape}"


def test_forward_training(model):
    # Prepare dummy input data with labels for training
    batch_size = 2
    sequence_length = 10
    num_labels = model.classifier[-1].out_features

    input_ids = torch.randint(0, 32100, (batch_size, sequence_length))
    attention_mask = torch.ones((batch_size, sequence_length))
    labels = torch.randint(0, num_labels, (batch_size,))  # Random labels

    # Run forward pass with labels (training mode)
    loss, logits = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )

    # Check output shapes
    assert (
        loss.shape == ()
    ), "Expected scalar loss, but got tensor of shape {loss.shape}"
    assert logits.shape == (
        batch_size,
        num_labels,
    ), f"Expected logits shape {(batch_size, num_labels)}, but got {logits.shape}"


def test_loss_calculation(model):
    # Prepare dummy logits and labels for loss calculation
    batch_size = 2
    num_labels = model.classifier[-1].out_features
    logits = torch.randn(batch_size, num_labels, requires_grad=True)
    labels = torch.randint(0, num_labels, (batch_size,))

    # Calculate loss
    loss = model.loss_fct(logits, labels)

    # Check loss properties
    assert loss.item() >= 0, "Expected non-negative loss value"
    assert loss.requires_grad, "Expected loss to require gradients"
