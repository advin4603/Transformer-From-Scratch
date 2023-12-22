from lightning_transformer import LightningTransformer
from transformer import Transformer
from train_tokenizer import START_TOKEN, END_TOKEN, PAD_TOKEN
from ted_talks_dataset import TedTalksEnglishFrenchDataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar
import torch

seed_everything(42)

english_tokenizer = Tokenizer.from_file("english_tokenizer.json")
french_tokenizer = Tokenizer.from_file("french_tokenizer.json")

english_start_token_id = english_tokenizer.token_to_id(START_TOKEN)
english_end_token_id = english_tokenizer.token_to_id(END_TOKEN)
english_pad_token_id = english_tokenizer.token_to_id(PAD_TOKEN)

french_start_token_id = french_tokenizer.token_to_id(START_TOKEN)
french_end_token_id = french_tokenizer.token_to_id(END_TOKEN)
french_pad_token_id = french_tokenizer.token_to_id(PAD_TOKEN)

train_set = TedTalksEnglishFrenchDataset("ted-talks-corpus/train.en", "ted-talks-corpus/train.fr", english_tokenizer,
                                         french_tokenizer, english_start_token_id, english_end_token_id,
                                         english_pad_token_id, french_start_token_id, french_end_token_id,
                                         french_pad_token_id)

dev_set = TedTalksEnglishFrenchDataset("ted-talks-corpus/dev.en", "ted-talks-corpus/dev.fr", english_tokenizer,
                                       french_tokenizer, english_start_token_id, english_end_token_id,
                                       english_pad_token_id, french_start_token_id, french_end_token_id,
                                       french_pad_token_id)

BATCH_SIZE = 24
train_loader = DataLoader(train_set, BATCH_SIZE,
                          shuffle=True, collate_fn=train_set.collate)
dev_loader = DataLoader(dev_set, BATCH_SIZE,
                        shuffle=False, collate_fn=dev_set.collate)

EMBEDDING_DIMENSIONS = 300
MAX_LENGTH = 2048
HEADS = 6
N = 2
FEED_FORWARD_DIMENSIONS = (600,)
FEED_FORWARD_DROPOUT = 0.5
ATTENTION_DROPOUT = 0.2

model = Transformer(english_tokenizer.get_vocab_size(), french_tokenizer.get_vocab_size(), EMBEDDING_DIMENSIONS,
                    MAX_LENGTH, HEADS, FEED_FORWARD_DIMENSIONS, ATTENTION_DROPOUT, FEED_FORWARD_DROPOUT, N, HEADS,
                    HEADS, FEED_FORWARD_DIMENSIONS, ATTENTION_DROPOUT, ATTENTION_DROPOUT, FEED_FORWARD_DROPOUT, N)

LEARNING_RATE = 2e-4
lightning_model = LightningTransformer(
    model, LEARNING_RATE, ignore_index=french_pad_token_id)

MAX_EPOCHS = 25
trainer = Trainer(default_root_dir="transformer_logs",
                  max_epochs=MAX_EPOCHS, callbacks=[RichProgressBar()])

trainer.fit(model=lightning_model, train_dataloaders=train_loader,
            val_dataloaders=dev_loader)

torch.save(model, "translator2.pt")
