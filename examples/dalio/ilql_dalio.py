from time import time

import ray
import wandb
from datasets import load_dataset
from transformers import pipeline

import trlx
import yaml
from typing import List, Dict
import os
from trlx.data.configs import TRLConfig
import os
from typing import Callable, Iterable, List, Optional, Tuple
import torch
import torch.nn.functional as F

from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ilql_model import AccelerateILQLModel
from trlx.pipeline.offline_pipeline import ILQLRolloutStorage
from trlx.utils import set_seed
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator

default_config = yaml.safe_load(open("configs/ilql_config.yml"))


class DalioOrchestrator(OfflineOrchestrator):

    def make_experience(self, samples, rewards):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the model
        """
        if self.model.tokenizer:
            input_ids = self.model.tokenize(samples)
        else:
            input_ids = samples

        input_ids = list(map(torch.as_tensor, input_ids))

        states_ixs, actions_ixs = [], []
        dones = []
        for s, s_tok in zip(samples, input_ids):
            # split samples on (prompts, continuations) on a given substring `split_token`
            if self.split_token:
                prompt_str_len = s.index(self.split_token) + len(self.split_token)
                prompt_tok_len = len(self.model.tokenizer(s[:prompt_str_len]).input_ids)
            # else assume that the prompt is a bos token
            else:
                prompt_tok_len = 1

            # indices of continuations, to mask prompts in loss computation
            a_ixs = torch.arange(prompt_tok_len + 1, len(s_tok) - 1)
            # same continuations but for value computation, with the premise to eventually support interleaved dialog
            s_ixs = torch.arange(prompt_tok_len + 1, len(s_tok))
            # mask continuation's ending
            terminals = torch.ones_like(s_ixs)
            terminals[-1] = 0

            actions_ixs.append(a_ixs)
            states_ixs.append(s_ixs)
            dones.append(terminals)

        if self.model.tokenizer:
            prompt = self.model.tokenizer.decode(input_ids[0][: states_ixs[0][1]])
            response = self.model.tokenizer.decode(input_ids[0][states_ixs[0][1]:])
            print("[Sample example]")
            print("Prompt: ", prompt)
            print("Response: ", response)

        print(f"[Mean reward] {torch.Tensor(rewards).mean():.2f}")
        print(
            f"[Mean sample length] {torch.mean(torch.Tensor(list(map(len, input_ids)))):.2f}"
        )

        returns = torch.as_tensor(rewards, dtype=torch.float)
        returns = (returns - returns.mean()) / (returns.std() + 1e-30)

        rewards = [torch.zeros(x.shape[0]) for x in actions_ixs]
        for rs, G in zip(rewards, returns):
            rs[-1] = G

        attention_mask = [torch.ones(x.shape[0], dtype=int) for x in input_ids]

        self.model.store = ILQLRolloutStorage(
            input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones
        )


class DalioModel(AccelerateILQLModel):

    def evaluate(self):
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        stats = {}
        all_samples = []
        generate_time = time()
        input_texts = []
        output_texts = []

        self.generate_kwargs = {
            # "max_new_tokens": 64,
            "eos_token_id": 50118,
            "pad_token_id": 50118,
        }

        for prompts in self.eval_dataloader:
            if isinstance(prompts, torch.Tensor):
                input_ids = prompts
                samples = self.generate(prompts)
            else:
                input_ids = prompts["input_ids"]
                samples = self.generate(**prompts)

            if isinstance(samples, tuple):
                samples, *_ = samples

            for prompt, sample in zip(input_ids, samples):
                input_texts.append(prompt)
                output_texts.append(sample[len(prompt):])

            pad_token = self.tokenizer.eos_token_id if self.tokenizer else 0
            all_samples.append(
                F.pad(
                    samples,
                    (0, self.max_length - samples.shape[1]),
                    value=pad_token,
                )
            )
        stats["generate_time"] = time() - generate_time

        samples = self.accelerator.gather(torch.vstack(all_samples))
        input_texts = self.accelerator.gather(torch.vstack(input_texts))
        output_texts = self.accelerator.gather(torch.vstack(output_texts))

        if self.accelerator.is_main_process:
            if self.tokenizer:
                samples = self.tokenizer.batch_decode(samples, skip_special_tokens=True)
                input_texts = self.tokenizer.batch_decode(input_texts, skip_special_tokens=True)
                output_texts = self.tokenizer.batch_decode(output_texts, skip_special_tokens=True)

            if isinstance(samples[0], str):
                columns_data = [samples]
            else:
                columns_data = [samples.tolist()]
            columns = ["samples"]

            # in online setting, compute the reward for validation
            if self.reward_fn:
                rewards = torch.as_tensor(self.reward_fn(samples), dtype=torch.float)
                mean_reward = rewards.mean()
                columns.append("reward")
                columns_data.append(rewards)
                stats["mean_reward"] = mean_reward
                print(f"{mean_reward=}")

            # additionally log any other metrics
            if self.metric_fn:
                metric_time = time()
                metrics = self.metric_fn(samples)
                stats["metric_time"] = time() - metric_time

                mean_metrics = {
                    f"metrics/{k}": torch.as_tensor(xs).mean(-1)
                    for k, xs in metrics.items()
                }

                stats.update(mean_metrics)

                for metric, values in metrics.items():
                    columns.append(metric)
                    columns_data.append(values)

            rows = list(zip(*columns_data))
            print(rows[0])
            print(list(zip(*[input_texts, output_texts])))
            if not ray.is_initialized():
                stats["samples"] = wandb.Table(columns=columns, rows=rows)
                stats["table"] = wandb.Table(columns=["input_text", "generated_response"],
                                             rows=list(zip(*[input_texts, output_texts])))

        return stats


def main(hparams={}):
    model_path = "facebook/opt-125m"
    logit_mask = None
    metric_fn = None
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("ChaiML/dalio_scored_responses_v1")
    train_dataset = dataset["train"]
    split_token = "<|endoftext|>"
    samples = [
        input_text + split_token + output_text for input_text, output_text in
        zip(train_dataset["input_text"], train_dataset["output_text"])
    ]
    rewards = train_dataset["score"]

    eval_prompts = dataset["train"]["input_text"][:8]

    if len(samples) != len(rewards):
        raise ValueError(
            f"Number of samples {len(samples)} should match the number of rewards {len(rewards)}"
        )

    if config is None:
        config = TRLConfig.load_yaml("examples/dalio/ilql_config.yml")
    set_seed(config.train.seed)

    if model_path:
        config.model.model_path = model_path

    model = DalioModel(
        config=config,
        logit_mask=logit_mask,
        metric_fn=metric_fn,
    )

    model.tokenizer.eos_token_id = 50118
    model.tokenizer.pad_token_id = 50118
    model.tokenizer.bos_token_id = 50118

    batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
    if eval_prompts is None:
        eval_prompts = [model.tokenizer.bos_token] * batch_size
    eval_pipeline = get_pipeline(config.train.pipeline)(
        eval_prompts, model.tokenizer
    )

    orch = DalioOrchestrator(
        model, split_token=split_token
    )
    orch.make_experience(samples, rewards)
    model.add_eval_pipeline(eval_pipeline)

    model.learn()


if __name__ == "__main__":
    main()
