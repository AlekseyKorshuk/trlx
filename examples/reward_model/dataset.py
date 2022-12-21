from torch.utils.data import Dataset
import torch


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []

        for pair in pairs:
            prompt = pair["prompt"] if "prompt" in pair else ""
            chosen, rejected = pair["chosen"], pair["rejected"]
            tok_chosen = tokenizer(prompt + chosen + "<|endoftext|>", return_tensors="pt")
            tok_rejected = tokenizer(rejected + "<|endoftext|>", return_tensors="pt")
            # Reject data with num tokens > max_length
            if tok_chosen.shape[-1] <= max_length and tok_rejected.shape[-1] <= max_length:
                chosen_encodings_dict = tokenizer(chosen + '<|endoftext|>', truncation=True,
                                                  max_length=max_length, padding="max_length", return_tensors="pt")
                rejected_encodings_dict = tokenizer(rejected + '<|endoftext|>', truncation=True,
                                                    max_length=max_length, padding="max_length", return_tensors="pt")
                self.chosen_input_ids.append(chosen_encodings_dict['input_ids'])
                self.chosen_attn_masks.append(chosen_encodings_dict['attention_mask'])
                self.rejected_input_ids.append(rejected_encodings_dict['input_ids'])
                self.rejected_attn_masks.append(rejected_encodings_dict['attention_mask'])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return self.chosen_input_ids[idx], self.chosen_attn_masks[idx], self.rejected_input_ids[idx], \
               self.rejected_attn_masks[idx]


def data_collator(data):
    return {'input_ids': torch.cat([f[0] for f in data] + [f[2] for f in data]),
            'attention_mask': torch.cat([f[1] for f in data] + [f[3] for f in data])}
