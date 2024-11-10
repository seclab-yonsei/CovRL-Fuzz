import torch.nn as nn
from transformers import T5Config, T5EncoderModel, T5PreTrainedModel


class CriticModel(T5PreTrainedModel):
    def __init__(self, model_path="", num_labels=8, dropout_rate=0.1):
        """
        Initializes the CriticModel for classification tasks using a T5 encoder.

        Args:
            model_path (str): Path or model identifier for the pretrained T5 model.
            num_labels (int): The number of output classes for classification.
            dropout_rate (float): Dropout rate applied before the classification layer.
        """

        config = T5Config()
        super().__init__(config)
        self.backbone = T5EncoderModel(config)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(
                self.backbone.config.d_model, num_labels
            ),  # d_model is the hidden size of the encoder
        )

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.

        Args:
            input_ids (Tensor): Tensor of input token IDs of shape (batch_size, sequence_length).
            attention_mask (Tensor): Attention mask tensor indicating which tokens should be attended to.
            labels (Tensor, optional): Ground truth labels for supervised learning. If provided, loss is computed.

        Returns:
            If labels are provided, returns a tuple (loss, logits).
            If labels are not provided, returns only logits.
        """
        # Pass inputs through the encoder to get hidden states
        encoder_outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_states = (
            encoder_outputs.last_hidden_state
        )  # (batch_size, sequence_length, d_model)

        # Use the hidden state of the first token (<s>) for classification
        cls_token_state = hidden_states[:, 0, :]  # (batch_size, d_model)

        logits = self.classifier(cls_token_state)  # (batch_size, num_labels)

        # If labels are provided, calculate and return the loss along with logits
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return loss, logits

        return logits
