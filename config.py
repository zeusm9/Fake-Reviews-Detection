import pathlib

# directories
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "op_spam_v1.4"
CLEANED_DIR = pathlib.Path(__file__).resolve().parent / "data" / "cleaned"
FAKE_CORPUS = CLEANED_DIR / "fake" / "fake.txt"
TRUTH_CORPUS = CLEANED_DIR / "truth" / "truth.txt"
LIAR_CORPUS = pathlib.Path(__file__).resolve().parent / "data" / "liar" / "train.tsv"
LIAR_FAKE = CLEANED_DIR / "fake" / "fake_liar.txt"
LIAR_TRUTH = CLEANED_DIR / "truth" / "truth_liar.txt"
