import config as c
import glob
import string
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from os import walk

nltk.download('stopwords')


def read_file(inputfile: str):
    with open(inputfile, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if f]


def write_file(outputfile: str, review):
    with open(outputfile, 'a', encoding='utf-8') as f:
        f.writelines(review + '\n')


def clean_text(inputext):
    lmtzr = WordNetLemmatizer()
    tokens = word_tokenize(inputext.lower())
    stops = set(stopwords.words('english')) | set(string.punctuation)
    return [lmtzr.lemmatize(word) for word in tokens if word and word not in stops and not word.isdigit() and word.isalpha()]


def load_dataset(datapath):
    fake = []
    truth = []
    for (dirpath, dirnames, filenames) in walk(datapath):
        if "deceptive" in dirpath:
            fake.extend(os.path.join(dirpath, filename) for filename in filenames)
        else:
            truth.extend(os.path.join(dirpath, filename) for filename in filenames)
    return fake, truth


def preprocess(filepaths):
    for filepath in filepaths:
        f = read_file(filepath)

        for line in f:
            cleaned = clean_text(line)
            if "deceptive" in filepath:
                folder = "fake"
            else:
                folder = "truth"
            write_file(str(c.CLEANED_DIR / folder / str(folder + ".txt")), ' '.join(cleaned))


def main():
    fake, truth = load_dataset(c.DATA_DIR)
    preprocess(fake)
    preprocess(truth)


if __name__ == "__main__":
    main()
