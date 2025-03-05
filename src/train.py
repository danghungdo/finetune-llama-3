import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer
from scipy.stats import pearsonr
from loguru import logger
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from src.inference import predict
import numpy as np


class MyTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.args.device
            )
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss of the model on the given inputs.
        """
        labels = inputs.pop("labels").long()

        ouputs = model(**inputs)

        logits = ouputs.get("logits")

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, ouputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    try:
        predictions = np.argmax(logits, axis=1)
        pearson, _ = pearsonr(predictions, labels)
        return {"pearson": pearson}
    except Exception as e:
        logger.warning(f"Error computing pearson correlation: {e}")
        return {"pearson": None}


def train_model(peft_model, dataset, category_map, tokenizer, config):
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluation_strategy=config["evaluation_strategy"],
        save_strategy=config["save_strategy"],
        load_best_model_at_end=config["load_best_model_at_end"],
    )
    # shuffle and split data
    X = dataset.pop("query").to_frame()
    y = dataset.pop("target").to_frame()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )

    # compute class weights
    class_weights = (1 / y_train.value_counts(normalize=True)).to_list()
    class_weights = torch.tensor(class_weights)
    class_weights = class_weights / class_weights.sum()
    # Create huggingface datasets
    dataset_train = Dataset.from_pandas(
        pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    )
    dataset_val = Dataset.from_pandas(
        pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
    )
    dataset_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    dataset = DatasetDict(
        {
            "train": dataset_train,
            "eval": dataset_val,
        }
    )

    collate_fnc = DataCollatorWithPadding(tokenizer)
    max_len = config["max_len"]

    def process_llama(examples):
        return tokenizer(examples["query"], truncation=True, max_length=max_len)

    tokenized_datasets = dataset.map(process_llama, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("target", "label")
    tokenized_datasets.set_format("torch")

    peft_model.config.pad_token_id = tokenizer.pad_token_id
    peft_model.config.use_cache = False
    peft_model.config.pretraining_tp = 1
    
    trainer = MyTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        tokenizer=tokenizer,
        data_collator=collate_fnc,
        compute_metrics=compute_metrics,
        class_weights=class_weights,

    )

    trainer.train()

    # test on test set and print metrics
    predictions = predict(peft_model, dataset_test, tokenizer, max_len)
    predictions = [category_map[p] for p in predictions]
    y_test = dataset_test["target"].tolist()
    y_test = [category_map[p] for p in y_test]
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Classification report:")
    print(classification_report(y_test, predictions))
    print("Accuracy score:")
    print(accuracy_score(y_test, predictions))
