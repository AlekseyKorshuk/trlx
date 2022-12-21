from typing import Optional

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, AutoModel, \
    AutoConfig, PreTrainedModel, HfArgumentParser
import json
from reward_model import GPTRewardModel
from trainer import PairwiseTrainer
from dataset import PairwiseDataset, data_collator
import deepspeed
from dataclasses import dataclass, field
import os
import sys


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def prepare_data(data_args, tokenizer, max_seq_length):
    dataset = load_dataset(data_args.dataset_name)
    train_dataset = PairwiseDataset(dataset["train"], tokenizer, max_length=max_seq_length)
    if "validation" in dataset:
        val_dataset = PairwiseDataset(dataset["validation"], tokenizer, max_length=max_seq_length)
    else:
        train_size = int((1 - data_args.validation_split_percentage) * len(train_dataset))
        train_dataset, val_dataset = random_split(train_dataset, [train_size, len(train_dataset) - train_size])
    return train_dataset, val_dataset


def load_model(model_name_or_path):
    model = GPTRewardModel(model_name_or_path)
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.5 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)
    # load_checkpoint = False
    # if load_checkpoint:
    #     model.load_state_dict(torch.load('ckpts/single_context_pairwise/model_fp16.pt'))
    return model


def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = load_tokenizer(
        tokenizer_name=model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    )
    model = load_model(
        model_name_or_path=model_args.model_name_or_path,
    )
    train_dataset, val_dataset = prepare_data(data_args, tokenizer, model.config.max_position_embeddings)

    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    trainer.train()

    trainer.push_to_hub()


if __name__ == '__main__':
    main()
