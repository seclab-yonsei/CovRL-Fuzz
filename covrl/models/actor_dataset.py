import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ActorDataset(Dataset):
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
        self.device = device
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.max_length = tokenizer.model_max_length
        self.mask_probability = mask_probability
        self.poisson_lambda = poisson_lambda
        self.critic_vocab_size = model_vocab_size

    def __len__(self):
        return len(self.dataset["data"])

    def random_spans_noise_mask(self, length):
        num_noise_tokens = max(
            1, min(length - 1, int(round(length * self.mask_probability)))
        )
        num_noise_spans = max(1, int(round(num_noise_tokens / self.poisson_lambda)))

        def random_segmentation(total_length, num_segments):
            # Segments the total length randomly
            splits = (
                np.cumsum(
                    np.random.multinomial(
                        total_length - num_segments, [1.0] * num_segments
                    )
                )
                + 1
            )
            return np.diff(np.insert(splits, 0, 0))

        noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = random_segmentation(
            length - num_noise_tokens, num_noise_spans
        )

        # Interleave noise and non-noise spans
        span_lengths = np.empty((num_noise_spans * 2,), dtype=int)
        span_lengths[0::2], span_lengths[1::2] = (
            nonnoise_span_lengths,
            noise_span_lengths,
        )
        mask = np.zeros(length, dtype=bool)
        mask[np.cumsum(span_lengths)[:-1]] = 1
        return np.cumsum(mask) % 2 == 1

    def create_sentinel_ids(self, mask_indices):
        mask_indices[:, -1] = 1
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]
        sentinel_ids = np.where(
            start_indices != 0,
            len(self.tokenizer) - np.cumsum(start_indices, axis=-1),
            0,
        )
        return sentinel_ids - mask_indices + start_indices

    def filter_input_ids(self, input_ids, sentinel_ids, pad_token=0):
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full >= 0]
        labels = np.concatenate([input_ids.copy(), [self.tokenizer.eos_token_id] * 2])
        attention_mask = np.ones(input_ids.shape[0], dtype=int)
        return input_ids, attention_mask, labels

    def pad_tokens(self, token_ids, pad_id=None):
        pad_id = pad_id or self.tokenizer.pad_token_id
        padded_tokens = np.pad(token_ids, (0, max(0, 1)), constant_values=pad_id)
        return padded_tokens[: self.max_length]

    def mask_tokens(self, inputs):
        inputs = inputs.squeeze()[: self.max_length]  # Limit to max length
        is_token = ~(inputs == self.tokenizer.eos_token_id)
        mask_indices = np.asarray(
            [self.random_spans_noise_mask(is_token.sum().item() + 1)]
        )

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        input_ids, attention_mask, _ = self.filter_input_ids(inputs, input_ids_sentinel)

        labels_sentinel = self.create_sentinel_ids((~mask_indices).astype(np.int8))
        _, decoder_attention_mask, labels = self.filter_input_ids(
            inputs, labels_sentinel
        )

        input_ids = torch.tensor(
            self.pad_tokens(input_ids), dtype=torch.long, device=self.device
        )
        attention_mask = torch.tensor(
            self.pad_tokens(attention_mask, pad_id=0),
            dtype=torch.long,
            device=self.device,
        )
        labels = torch.tensor(
            self.pad_tokens(labels, pad_id=0), dtype=torch.long, device=self.device
        )
        return input_ids, labels, attention_mask

    def __getitem__(self, idx):
        data = self.dataset["data"][idx]
        encoded_ids = self.tokenizer(
            data, return_tensors=self.return_tensors, truncation=True
        )
        masked_encoded_ids, labels, attention_mask = self.mask_tokens(
            encoded_ids["input_ids"]
        )

        return {
            "input_ids": masked_encoded_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": labels,
            "labels": labels,
        }

class ActorDataCollator:
    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        decoder_input_ids = [f["decoder_input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        decoder_input_ids_padded = pad_sequence(decoder_input_ids, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "decoder_input_ids": decoder_input_ids_padded,
            "labels": labels_padded,
        }