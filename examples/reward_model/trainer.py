from transformers import Trainer
import torch


class PairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        rewards = model(**inputs)
        rewards_chunked = rewards.view((2, -1))
        chosen_rewards = rewards_chunked[0]
        rejected_rewards = rewards_chunked[1]
        # compute pairwise loss
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        return (loss, rewards) if return_outputs else loss
