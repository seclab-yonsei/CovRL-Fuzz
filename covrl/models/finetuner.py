from torch import nn
import pandas as pd
from transformers import (AutoTokenizer, T5ForConditionalGeneration)

from covrl.models.actor_dataset import ActorDataset, ActorDataCollator
from covrl.models.critic import CriticModel
from covrl.models.critic_dataset import CriticDataset, CriticDataCollator
from covrl.models.rewarding import Rewarding
from covrl.utils.base_utils import *
from covrl.utils.map_target_error import map_target_error


pd.set_option("mode.chained_assignment", None)
afl_showmap_path = os.path.abspath("../AFL/afl-showmap")
from transformers import Trainer, TrainingArguments


class ActorTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):

        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")

        outputs = model(input_ids=input_ids, labels=labels)

        loss = self.compute_loss_func(outputs, inputs, num_items_in_batch)
        return (loss, outputs) if return_outputs else loss


class FineTuner:
    def __init__(
        self,
        config,
        device,
        actor_path,
        critic_path=None,
        save_period=1,
        train_dataset_path=None,
        save_dir="./",
    ):
        self.config = config
        self.target = map_target_error(self.config.target_interpreter)
        self.device = device
        self.save_dir = save_dir
        self.training_args = TrainingArguments(
            output_dir="model_output",
            overwrite_output_dir=True,
            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            fp16=False,  # Use FP16 if supported
            learning_rate=config.learning_rate,
            eval_strategy="no",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=False,
        )
        self.actor_path = actor_path
        self.critic_path = critic_path
        self.tokenizer = AutoTokenizer.from_pretrained(config.load_path)
        self.train_dataset_path = train_dataset_path
        self.save_period = save_period
        self.reward_object = Rewarding(self.config, save_dir=self.save_dir)
        self.train_dataset = self.preprocess_dataset(self.train_dataset_path)
        self.mutation_dataset = pd.DataFrame([], columns=["is_orig", "file_id", "data"])
        self.dataset = None
        self.critic = None
        self.actor = None
        self.scaler = None  # No need with Hugging Face Trainer

    def load_critic(self):
        if self.critic:
            pass
        elif self.critic_path:
            self.critic = CriticModel(self.critic_path)
        else:
            self.critic = CriticModel(self.actor_path)
        print("Critic model loaded successfully")
        return self.critic

    def load_actor(self, filepath):
        model = T5ForConditionalGeneration.from_pretrained(filepath)
        model.to(self.device)
        print("Actor model loaded successfully")
        return model

    def save_model(self, model, filename):
        filepath = os.path.join(self.save_dir, filename)
        model.save_pretrained(filepath)
        return filepath

    def prepare_dataset(self, df, tokenizer, option="critic"):
        if option == "critic":
            dataset = CriticDataset(
                device=self.device,
                dataset=df,
                tokenizer=tokenizer,
                mask_probability=self.config.mask_probability,
                model_vocab_size=self.tokenizer.vocab_size,
            )
        else:
            dataset = ActorDataset(
                device=self.device,
                dataset=df,
                tokenizer=tokenizer,
                mask_probability=self.config.mask_probability,
                model_vocab_size=self.tokenizer.vocab_size,
            )
        return dataset

    def train_critic(self, critic_dataset=None, epochs=1):
        self.critic = self.load_critic()
        self.critic.to(self.device)

        if not critic_dataset:
            critic_dataset = self.prepare_dataset(
                self.dataset, self.tokenizer, option="critic"
            )

        self.training_args.num_train_epochs = epochs

        trainer = Trainer(
            model=self.critic,
            args=self.training_args,
            train_dataset=critic_dataset,
            data_collator=CriticDataCollator()
        )
        print("Training critic model...")
        trainer.train()
        self.critic_path = self.save_model(self.critic, f"critic_final")
        return self.critic_path

    def compute_actor_loss(self, cur_outputs, prev_actor, critic, inputs):
        # Set critic and previous actor to evaluation mode
        critic.eval()
        prev_actor.eval()

        # Calculate critic scores based on current actor's predictions
        cur_predictions = cur_outputs.logits.argmax(dim=-1)
        critic_inputs = {
            "input_ids": torch.cat((inputs["input_ids"], cur_predictions), dim=1),
            "attention_mask": torch.cat(
                (
                    inputs["attention_mask"],
                    torch.ones_like(cur_predictions, dtype=torch.long),
                ),
                dim=1,
            ),
        }

        with torch.no_grad():
            prev_outputs = prev_actor(**inputs)
            critic_scores = critic(**critic_inputs)

        # Convert scores to reward signal
        pred_labels = torch.argmax(critic_scores, dim=-1)
        reward = torch.tensor([label_to_score[label.item()] for label in pred_labels])
        reward = reward.unsqueeze(1).unsqueeze(2)

        log_prob_fct = nn.LogSoftmax(dim=-1)
        cur_log_probs = log_prob_fct(cur_outputs.logits)
        prev_log_probs = log_prob_fct(prev_outputs.logits)

        ratio = torch.exp(cur_log_probs - prev_log_probs)

        # Clip the ratio to stay within bounds, e.g., [0.8, 1.2]
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        ppo_loss = -torch.min(ratio * reward, clipped_ratio * reward).mean()

        # Final adjusted loss
        final_loss = ppo_loss + cur_outputs.loss.mean()
        return final_loss

    def finetune_actor(self, actor_dataset=None, epochs=1):
        self.actor = self.load_actor(self.actor_path)
        self.prev_actor = self.load_actor(self.actor_path)
        self.critic = self.load_critic() if self.critic_path else self.actor_path

        self.prev_actor.eval()  # Actor in evaluation mode
        self.critic.eval()  # Critic in evaluation mode

        if not actor_dataset:
            actor_dataset = self.prepare_dataset(
                self.dataset, self.tokenizer, option="actor"
            )
        self.training_args.num_train_epochs = epochs

        # Custom loss function to incorporate critic score
        def custom_loss(model, inputs, num_items_in_batch=None):
            return self.compute_actor_loss(model, self.prev_actor, self.critic, inputs)

        # Set up Trainer with custom loss function
        trainer = ActorTrainer(
            model=self.actor,
            args=self.training_args,
            train_dataset=actor_dataset,
            compute_loss_func=custom_loss,
            data_collator=ActorDataCollator()
        )

        print("Fine-tuning actor model with critic scoring...")
        trainer.train()
        self.actor_path = self.save_model(self.actor, "actor_final")
        return self.actor_path

    def preprocess_dataset(self, dataset_path):
        df = pd.read_json(dataset_path)
        df = df.dropna()
        df["file_id"] = -1
        df["is_orig"] = False
        return df

    def preprocess(self, dir_path):
        self.reward_object.set_dir(dir_path)

        def decode(data):
            return self.tokenizer.decode(data, skip_special_tokens=True)

        def load_files(load_path):
            files = load_testsuites(
                load_path,
                except_kw=[
                    "fuzzer_stats",
                    "MLM_pred",
                    "MLM_decoded",
                    "MLM_Record",
                    "fuzz_bitmap",
                    "plot_data",
                    ".cur_input",
                    ".synced",
                    "tmp",
                    "decoded",
                    ".state",
                    "idf_embedding.bin",
                    "hangs",
                    "README.txt",
                ],
            )
            files = sorted(
                files,
                key=lambda x: int(os.path.basename(x).split(",")[0].split(":")[-1]),
            )

            dataset = []
            for file in files:
                data = {}
                filename = os.path.basename(file)
                data["is_orig"] = "orig" in filename
                data["file_id"] = filename.split(",")[0].split(":")[-1]
                if not self.mutation_dataset[
                    self.mutation_dataset["file_id"] == data["file_id"]
                ].empty:
                    continue

                with open(file, "rb") as reader:
                    try:
                        texts = reader.read()
                    except Exception:
                        print("File not found.")
                        continue
                data["data"] = decode(hex_to_dec(texts))
                if data["data"]:
                    dataset.append(data)

            df = pd.DataFrame(dataset, columns=["is_orig", "file_id", "data"])
            return df

        dataset = load_files(dir_path)
        if self.mutation_dataset.empty:
            is_first = True
            self.mutation_dataset = dataset
            self.mutation_dataset = self.reward_object.update(
                self.mutation_dataset, is_update_idf=True
            )
            self.dataset = self.reward_object.update(
                self.train_dataset.sample(
                    min(len(self.mutation_dataset) * 4, len(self.train_dataset)), ignore_index=True
                ),
                is_update_idf=False,
            )
        else:
            is_first = False
            self.mutation_dataset = pd.concat(
                [self.mutation_dataset, dataset], ignore_index=True
            )
            self.mutation_dataset = self.reward_object.update(
                self.mutation_dataset, is_update_idf=True
            )

            train_dataset = self.reward_object.update(
                self.train_dataset.sample(len(self.mutation_dataset) * 4, len(self.train_dataset), ignore_index=True),
                is_update_idf=False,
            )
            self.dataset = pd.concat(
                [train_dataset, self.mutation_dataset], ignore_index=True
            )
        return is_first
