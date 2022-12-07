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

from trlx.data.configs import TRLConfig
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


def main(hparams={}):
    model_path = "facebook/opt-1.3b"
    logit_mask = None
    metric_fn = None
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("ChaiML/dalio_scored_responses_v1")
    train_dataset = dataset["train"]
    split_token = "[SEP]"
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

    model = get_model(config.model.model_type)(
        config=config,
        logit_mask=logit_mask,
        metric_fn=metric_fn,
    )

    model.tokenizer.eos_token_id = 50118

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
