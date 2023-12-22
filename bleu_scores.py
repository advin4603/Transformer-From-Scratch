from transformer import Transformer
import torch
from tokenizers import Tokenizer
from train_tokenizer import START_TOKEN, END_TOKEN, PAD_TOKEN
from rich.progress import track
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

english_tokenizer = Tokenizer.from_file("english_tokenizer.json")
french_tokenizer: Tokenizer = Tokenizer.from_file("french_tokenizer.json")

english_start_token_id = english_tokenizer.token_to_id(START_TOKEN)
english_end_token_id = english_tokenizer.token_to_id(END_TOKEN)
english_pad_token_id = english_tokenizer.token_to_id(PAD_TOKEN)

french_start_token_id = french_tokenizer.token_to_id(START_TOKEN)
french_end_token_id = french_tokenizer.token_to_id(END_TOKEN)
french_pad_token_id = french_tokenizer.token_to_id(PAD_TOKEN)

device = "cuda" if torch.cuda.is_available() else "cpu"

model: Transformer = torch.load("translator.pt").to(device)
model.eval()

print(model.embedding_module)

# Create a SmoothingFunction for NLTK BLEU
smoother = SmoothingFunction().method4

for file_name, length in [("train", 30000), ("dev", 887), ("test", 1305)]:
    with open(f"ted-talks-corpus/{file_name}.en") as english_file, open(f"ted-talks-corpus/{file_name}.fr") as french_file, open(f"{file_name}_bleu.txt", "w") as out, torch.no_grad():
        for english_line, french_line in track(zip(english_file, french_file), description=f"Calculating {file_name}", total=length):
            english_line, french_line = english_line.strip(), french_line.strip()
            encoded_english = [english_start_token_id, *
                               english_tokenizer.encode(english_line).ids, english_end_token_id]
            encoded_french = [french_start_token_id]
            prediction = model.beam_search(
                encoded_english, encoded_french, device, 10, french_end_token_id, max_results=1)[0][0]
            predicted_french_line = french_tokenizer.decode(prediction).strip()
            bleu = sentence_bleu(
                [french_line.split()], predicted_french_line.split(), smoothing_function=smoother)
            print(
                f"{english_line} | {predicted_french_line} | {french_line} | {bleu}", file=out)

