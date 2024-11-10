import numpy as np
import torch
from covrl.models.actor_dataset import ActorDataset
from covrl.utils.base_utils import score_to_label
from torch.nn.utils.rnn import pad_sequence

class CriticDataset(ActorDataset):
    def __init__(
        self,
        device,
        dataset,
        tokenizer,
        mask_probability=0.15,
        poisson_lambda=3.0,
        return_tensors="pt",
        model_vocab_size=0,
    ):
        super().__init__(
            device,
            dataset,
            tokenizer,
            mask_probability,
            poisson_lambda,
            return_tensors,
            model_vocab_size,
        )
        self.device = device
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.max_length = self.tokenizer.model_max_length
        self.mask_probability = mask_probability
        self.poisson_lambda = poisson_lambda

        self.critic_vocab_size = model_vocab_size

    def __len__(self):
        return len(self.dataset["data"])

    def __getitem__(self, idx):
        data = self.dataset["data"][idx]
        encoded_ids = self.tokenizer(
            data, return_tensors=self.return_tensors, truncation=True
        )
        masked_encoded_ids, mask_ids, attention_mask = self.mask_tokens(
            encoded_ids["input_ids"]
        )

        # Convert score to label
        labels = torch.tensor(
            score_to_label(self.dataset["reward"][idx]), dtype=torch.long
        )

        return {
            "input_ids": torch.cat((masked_encoded_ids, mask_ids), dim=-1),
            "attention_mask": torch.cat(
                (attention_mask, torch.ones_like(mask_ids)), dim=-1
            ),
            "labels": labels,
        }

class CriticDataCollator:
    def __call__(self, features):
        input_ids = [item["input_ids"] for item in features]
        attention_masks = [item["attention_mask"] for item in features]
        labels = [item["labels"] for item in features]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels_padded = torch.stack(labels)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_masks_padded,
            "labels": labels_padded,
        }