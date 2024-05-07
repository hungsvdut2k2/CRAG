import json
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from sklearn.utils import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)

from src.arguments import DatasetArguments, ModelArguments, TrainingProcessArguments
from src.data_process import preprocess_function
from src.metrics import compute_metrics
from src.trainers import get_trainer


class TrainingPipeline:
    def __init__(
        self,
        training_arguments: TrainingProcessArguments,
        dataset_arguments: DatasetArguments,
        model_arguments: ModelArguments,
    ):
        self.training_arguments = training_arguments
        self.dataset_arguments = dataset_arguments
        self.model_arguments = model_arguments

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_arguments.model_name)
        self.label2id = json.load(open(self.dataset_arguments.label_file_path))
        self.id2label = {value: key for key, value in self.label2id.items()}

    def _load_dataset(self):
        dataset = DatasetDict.from_csv(
            {
                "train": os.path.join(
                    self.dataset_arguments.data_directory, "train_dataset.csv"
                ),
                "valid": os.path.join(
                    self.dataset_arguments.data_directory, "valid_dataset.csv"
                ),
                "test": os.path.join(
                    self.dataset_arguments.data_directory, "test_dataset.csv"
                ),
            }
        )

        dataset = dataset.shuffle(seed=self.training_arguments.seed)
        preprocess_function_with_tokenizer = partial(
            preprocess_function, tokenizer=self.tokenizer
        )

        dataset = dataset.map(
            preprocess_function_with_tokenizer,
            batched=True,
            num_proc=self.dataset_arguments.num_process,
            remove_columns=["text"],
        )

        return dataset

    def _model_init(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_arguments.model_name,
            num_labels=self.model_arguments.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        return model

    def _calculate_class_weight(self):
        train_df = pd.read_csv(
            os.path.join(self.arguments.data_directory, "train_dataset.csv")
        )
        train_labels = train_df["label"].tolist()
        class_weight = compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )

        return torch.tensor(class_weight, dtype=torch.float).to(self.arguments.device)

    def _train(self):
        dataset = self._load_dataset()
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        compute_metrics_with_id2label = partial(compute_metrics, id2label=self.id2label)

        training_arguments = TrainingArguments(
            output_dir=self.training_arguments.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.training_arguments.learning_rate,
            weight_decay=self.training_arguments.weight_decay,
            num_train_epochs=self.training_arguments.num_epochs,
            warmup_steps=self.training_arguments.warmup_steps,
            logging_dir="./logs",
            logging_steps=10,
            per_device_train_batch_size=self.training_arguments.batch_size,
            per_device_eval_batch_size=self.training_arguments.batch_size,
            push_to_hub=False,
            seed=42,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1-score",
            gradient_accumulation_steps=self.training_arguments.gradient_accumulation_steps,
            label_smoothing_factor=self.training_arguments.label_smoothing_factor,
        )

        if self.training_arguments.trainer_type == "base":
            trainer = get_trainer(
                trainer_type=self.training_arguments.trainer_type,
                model_init=self._model_init,
                args=training_arguments,
                train_dataset=dataset["train"],
                eval_dataset=dataset["valid"],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics_with_id2label,
            )

        # using Focal Loss Trainer
        elif self.training_arguments.trainer_type == "focal_loss":
            if not self.training_arguments.alpha:
                self.training_arguments.alpha = self._calculate_class_weight()
            else:
                self.training_arguments.alpha = torch.tensor(
                    [self.training_arguments.alpha_1, self.training_arguments.alpha_2],
                    dtype=torch.float,
                ).to(self.training_arguments.device)
            trainer = get_trainer(
                trainer_type=self.training_arguments.trainer_type,
                alpha=self.training_arguments.alpha,
                gamma=self.training_arguments.gamma,
                model_init=self._model_init,
                args=training_arguments,
                train_dataset=dataset["train"],
                eval_dataset=dataset["valid"],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics_with_id2label,
                num_classes=max(dataset["train"]["label"]) + 1,
            )

        trainer.train()
        evaluate_result = trainer.evaluate()
        test_result = trainer.predict(test_dataset=dataset["test"])
        return evaluate_result["eval_f1-score"], test_result.metrics["test_f1-score"]

    def train(self):
        return self._train()
