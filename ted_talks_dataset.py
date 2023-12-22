import torch
from torch.utils.data import Dataset
from pathlib import Path
from os import PathLike
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence


class TedTalksEnglishFrenchDataset(Dataset):
    def __init__(self, english_file_path: Path | PathLike, french_file_path: Path | PathLike,
                 english_tokenizer: Tokenizer, french_tokenizer: Tokenizer, english_start_token_id: int,
                 english_end_token_id: int, english_pad_token_id: int, french_start_token_id: int,
                 french_end_token_id: int, french_pad_token_id: int):
        with (open(english_file_path) as english_file, open(french_file_path) as french_file):
            self.english_sequences = [e.strip() for e in english_file]
            self.french_sequences = [f.strip() for f in french_file]
        self.english_tokenizer = english_tokenizer
        self.french_tokenizer = french_tokenizer

        self.english_start_token_id = english_start_token_id
        self.english_end_token_id = english_end_token_id
        self.english_pad_token_id = english_pad_token_id

        self.french_start_token_id = french_start_token_id
        self.french_end_token_id = french_end_token_id
        self.french_pad_token_id = french_pad_token_id

        self.encoded_english_sequences = self.english_tokenizer.encode_batch(self.english_sequences,
                                                                             add_special_tokens=False)
        self.encoded_french_sequences = self.french_tokenizer.encode_batch(self.french_sequences,
                                                                           add_special_tokens=False)

    def __len__(self):
        return len(self.encoded_english_sequences)

    def __getitem__(self, index: int):
        return torch.tensor([self.english_start_token_id, *self.encoded_english_sequences[index].ids,
                             self.english_end_token_id]), torch.tensor(
            [self.french_start_token_id, *self.encoded_french_sequences[index].ids, self.french_end_token_id])

    def collate(self, item_list: list[torch.Tensor, torch.Tensor]) \
            -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.BoolTensor]:
        english_list = [i[0] for i in item_list]
        french_list = [i[1] for i in item_list]

        padded_english = pad_sequence(english_list, batch_first=True, padding_value=self.english_pad_token_id)
        padded_french = pad_sequence(french_list, batch_first=True, padding_value=self.french_pad_token_id)

        english_padding_mask: torch.BoolTensor = padded_english != self.english_pad_token_id
        french_padding_mask: torch.BoolTensor = padded_french != self.french_pad_token_id

        return padded_english, padded_french, english_padding_mask, french_padding_mask
