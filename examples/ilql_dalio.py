from datasets import load_dataset
from transformers import pipeline

import trlx
import yaml
from typing import List, Dict
import os
from trlx.data.configs import TRLConfig


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


default_config = yaml.safe_load(open("configs/ilql_config.yml"))


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    # sentiment_fn = pipeline(
    #     "sentiment-analysis",
    #     "lvwerra/distilbert-imdb",
    #     top_k=2,
    #     truncation=True,
    #     batch_size=256,
    #     device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    # )

    # def metric_fn(samples: List[str]) -> Dict[str, List[float]]:
    #     sentiments = list(map(get_positive_score, sentiment_fn(samples)))
    #     return {"sentiments": sentiments}

    dataset = load_dataset("ChaiML/dalio_scored_responses_v1")
    train_dataset = dataset["train"]
    split_token = "[SEP]"
    texts = [
        input_text + split_token + output_text for input_text, output_text in
        zip(train_dataset["input_text"], train_dataset["output_text"])
    ]
    labels = train_dataset["score"]

    eval_prompts = dataset["train"]["input_text"][:8]

    trlx.train(
        "facebook/opt-1.3b",
        dataset=(texts, labels),
        eval_prompts=eval_prompts,
        # metric_fn=metric_fn,
        config=config,
        split_token=split_token,
    )


if __name__ == "__main__":
    main()
