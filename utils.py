import torch
from torch.utils.data import Dataset
import pandas as pd

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the data from the CSV.
    Assumes the CSV has columns 'review' and 'labels'.
    The 'labels' column should be a comma-separated string of binary values, e.g., "1,0,0,1".
    """
    texts = df["review"].tolist()
    labels = df["labels"].apply(lambda x: [int(i) for i in x.split(",")]).tolist()
    return texts, labels

class CustomReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float)
        }
