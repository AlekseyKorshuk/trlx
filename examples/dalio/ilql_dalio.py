from datasets import load_dataset
from transformers import pipeline

import trlx
import yaml
from typing import List, Dict
import os
from trlx.data.configs import TRLConfig
import os
from typing import Callable, Iterable, List, Optional, Tuple

from trlx.data.configs import TRLConfig
from trlx.utils import set_seed
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline

default_config = yaml.safe_load(open("configs/ilql_config.yml"))


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

    batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
    if eval_prompts is None:
        eval_prompts = [model.tokenizer.bos_token] * batch_size
    eval_pipeline = get_pipeline(config.train.pipeline)(
        eval_prompts, model.tokenizer
    )

    orch = get_orchestrator(config.train.orchestrator)(
        model, split_token=split_token
    )
    orch.make_experience(samples, rewards)
    model.add_eval_pipeline(eval_pipeline)


if __name__ == "__main__":
    main()
