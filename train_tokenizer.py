from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import argparse
from pathlib import Path

START_TOKEN = "<{START}>"
END_TOKEN = "<{END}>"
PAD_TOKEN = "<{PAD}>"


def train_tokenizer(train_paths: list[Path], vocabulary_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(vocab_size=vocabulary_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=[START_TOKEN, END_TOKEN, PAD_TOKEN])

    tokenizer.train([str(p) for p in train_paths], trainer=trainer)
    return tokenizer


def exists_path_type(arg):
    path = Path(arg)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{arg} does not exist.")
    return path


def path_type(arg):
    return Path(arg)


def main():
    parser = argparse.ArgumentParser(description="Train a byte level BPE")

    # Define the 'train_corpus_paths' argument as a required positional argument.
    # The nargs='+' allows for at least one argument, which will be stored as a list.
    parser.add_argument(
        'train_corpus_paths',
        type=exists_path_type,
        nargs='+',
        help="List of at least one file paths for training."
    )

    parser.add_argument(
        'vocabulary_size',
        type=int,
        help="Specify the vocabulary size as an integer."
    )

    parser.add_argument(
        '--output-path',
        type=path_type,
        help="Optional output path for the tokenizer file. If a folder is provided, 'tokenizer.json' will be appended."
    )

    args = parser.parse_args()

    # Process the output path.
    if args.output_path is None:
        # If no output path is provided, use the first path from train_corpus_paths.
        output_path = Path("tokenizer.json")
    elif args.output_path.is_dir():
        # If a directory is provided, append 'tokenizer.json' to it.
        output_path = args.output_path / 'tokenizer.json'
    else:
        # Otherwise, use the provided path as is.
        output_path = args.output_path

    train_corpus_paths = args.train_corpus_paths
    tokenizer = train_tokenizer(train_corpus_paths, args.vocabulary_size)
    tokenizer.save(str(output_path), pretty=True)


if __name__ == "__main__":
    main()
