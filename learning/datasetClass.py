import pandas as pd
import torch
from torch.utils.data import Dataset

tensor = torch.tensor


class ConversationDataset(Dataset):
    def __init__(self, data_file_path: str, tokenizer, max_length=512):
        """
        Args:
            data_file_path (string): Path to the csv file with conversations.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer to convert text to tokens.
            max_length (int): Maximum length of tokenized sequences.
        """
        super().__init__()
        self.data = pd.read_csv(data_file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        reply = self.data.iloc[idx, 2]
        question = self.data.iloc[idx, 1]
        input_encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        response_encoding = self.tokenizer(
            reply,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # sample = {
        #     'input_ids': input_encoding['input_ids'].flatten(),
        #     'attention_mask': input_encoding['attention_mask'].flatten(),
        #     'response_ids': response_encoding['input_ids'].flatten(),
        #     'response_attention_mask': response_encoding['attention_mask'].flatten()
        # }
        input_ids = input_encoding['input_ids'].flatten()
        response_ids = response_encoding['input_ids'].flatten()
        return input_ids, response_ids
