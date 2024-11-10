import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from covrl.models.finetuner import FineTuner
from covrl.utils.base_utils import *



class Inferencer:
    def __init__(
        self, conf, model_path, critic_path=None, sample_method="greedy", save_dir="./"
    ):
        self.config = conf
        self.model_path = model_path
        self.device = self._get_device()
        self.tokenizer = self._initialize_tokenizer(conf.load_path)

        self.load_model(self.model_path)
        self._initialize_tokens()

        self.sample_method = sample_method
        self.finetuner = self._initialize_finetuner(conf, critic_path, save_dir)

    def _initialize_tokenizer(self, load_path):
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        print(f"vocab_size: {tokenizer.vocab_size}")
        return tokenizer

    def _initialize_finetuner(self, conf, critic_path, save_dir):
        return FineTuner(
            config=conf,
            actor_path=self.model_path,
            critic_path=critic_path,
            device=self.device,
            save_period=1,
            train_dataset_path=conf.train_dataset_path,
            save_dir=save_dir,
        )

    def _initialize_tokens(self):
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.PADDING = self.tokenizer.pad_token_id
        self.UNKNOWN_TOKEN = self.tokenizer.unk_token_id
        self.MASK_TOKEN = self.tokenizer.mask_token_id
        self.max_length = self.tokenizer.model_max_length
        self.split_length = self.max_length - 3
        self.max_pred_length = round(self.max_length * self.config.mask_probability)

    def _get_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        print(
            f"Current CUDA device: {torch.cuda.current_device() if device.type == 'cuda' else 'N/A'}"
        )
        print(
            f"Count of available GPUs: {torch.cuda.device_count() if device.type == 'cuda' else 0}"
        )
        return device

    def load_model(self, loadpath):
        model = T5ForConditionalGeneration.from_pretrained(loadpath)
        model.eval()
        model.to(self.device)
        self.model = model
        print("Model loaded successfully.")

    def finetune(self, predict_path):
        print("Start fine-tuning...")
        is_first = self.finetuner.preprocess(predict_path)
        self.finetuner.train_critic(epochs=self.config.critic_epochs)
        if not is_first:
            self.model_path = self.finetuner.finetune_actor(
                epochs=self.config.actor_epochs
            )
        self.load_model(self.model_path)
        return self.model_path

    def masking(self, input_ids):
        """Apply masking to input IDs."""
        input_ids, mask_positions = self._mask_unknowns(input_ids)
        mask_dict = {
            self.tokenizer.vocab_size - i: pos
            for i, pos in enumerate(mask_positions, 1)
        }
        padded_input_ids, attention_mask = self._pad_input_ids(input_ids)

        return mask_dict, padded_input_ids, attention_mask

    def _mask_unknowns(self, input_ids):
        """Replace unknown tokens with mask tokens and identify masked positions."""
        mask_positions = []
        mask_cnt = 0
        for i, token in enumerate(input_ids):
            if token in {self.MASK_TOKEN, self.UNKNOWN_TOKEN}:
                mask_positions.append(i)
                mask_cnt += 1
                input_ids[i] = self.tokenizer.vocab_size - mask_cnt
        return input_ids, mask_positions

    def _pad_input_ids(self, input_ids):
        """Pad input IDs and create an attention mask."""
        padded_input_ids = input_ids + [self.PADDING]
        attention_mask = [1] * len(input_ids) + [0]
        return (
            torch.tensor([padded_input_ids], dtype=torch.long).to(self.device),
            torch.tensor([attention_mask], dtype=torch.long).to(self.device),
        )

    def split_sentences(self, token_ids):
        """Split token IDs into chunks based on max input length."""
        return [
            token_ids[i : i + self.split_length]
            for i in range(0, len(token_ids), self.split_length)
        ]

    def reconstruct(self, token_ids, predictions, mask_dict):
        """Reconstruct input by replacing masked positions with predicted tokens."""
        input_ids = token_ids[0]
        result_dict = {mask: (pos, []) for mask, pos in mask_dict.items()}

        # Fill result_dict with predictions
        prev_mask_id = None
        for pred in predictions:
            if pred in mask_dict:
                prev_mask_id = pred
            elif (
                pred > (self.tokenizer.vocab_size - 100)
                or pred == self.tokenizer.eos_token_id
            ):
                prev_mask_id = None
            elif prev_mask_id is not None:
                result_dict[prev_mask_id][1].append(pred)

        # Reconstruct the final input with predictions
        new_inputs = []
        prev_pos = 0
        for pos, preds in sorted(result_dict.values()):
            new_inputs.extend(input_ids[prev_pos:pos] + preds)
            prev_pos = pos + 1

        # Add remaining tokens after the last mask, up to the first padding token
        new_inputs.extend(
            input_ids[
                prev_pos : next(
                    (
                        i
                        for i, id in enumerate(input_ids[prev_pos:])
                        if id == self.PADDING
                    ),
                    len(input_ids),
                )
            ]
        )
        return new_inputs

    def inference(self, input_ids, num_samples=3):
        """Generate predictions for given input token IDs."""
        token_lists = self.split_sentences(input_ids)
        results = []

        for token_ids in token_lists:
            if len(token_ids) < 5:
                results.extend(token_ids)
                continue
            mask_dict, input_ids, attention_mask = self.masking(token_ids)
            if mask_dict:
                outputs = self._generate_predictions(
                    input_ids, attention_mask, num_samples
                )
                result = self.reconstruct(input_ids.tolist(), outputs, mask_dict)
                results.extend(result)
        return results

    def _generate_predictions(self, input_ids, attention_mask, num_samples):
        """Generate predictions using the specified sampling method."""
        if self.sample_method == "contrastive":
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                penalty_alpha=0.6,
                top_k=num_samples,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                min_length=1,
                max_length=self.max_pred_length,
            )
        else:
            outputs = self.model.generate(
                input_ids=input_ids,
                top_k=num_samples,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                max_length=self.max_pred_length,
            )
        return outputs.tolist()[0]
