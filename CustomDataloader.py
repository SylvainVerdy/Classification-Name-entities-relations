from torch.utils.data import Dataset, DataLoader
import torch
from tokenizers import Tokenizer
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):
    """My customdataset class"""

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[int(idx),-1]
        #text = tokenizer.encode_plus(text, truncation=True, max_length=100, padding='max_length', add_special_tokens=True)
        #text = tokenizer.encode(text)
        text = tokenizer.encode_plus(text)
        label =  self.df.iloc[int(idx),1]
        return torch.tensor(text['input_ids']), torch.tensor(text['attention_mask']), torch.tensor(label) # torch.tensor(text.ids), torch.tensor(text.attention_mask), torch.tensor(label)
      