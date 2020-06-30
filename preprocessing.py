import config as c
import string
import nltk
import os
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from os import walk

nltk.download('stopwords')


def read_file(inputfile: str):
    """
        Read file
        :param inputfile: input path
        :return:
    """
    with open(inputfile, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if f]


def write_file(outputfile: str, reviews):
    """
        Write file
        :param outputfile: output path
        :param reviews: list of reviews
        :return:
    """
    with open(outputfile, 'a', encoding='utf-8') as f:
        f.writelines(reviews + '\n')


def read_tsv(inputfile: str):
    """args command line
        Read tsv file
        :param inputfile: input path
        :return list of fake documents, list of truth documents:
    """
    fake = []
    truth = []
    with open(inputfile, 'r', encoding='utf-8') as f:
        rd = csv.reader(f, delimiter="\t", quotechar='"')
        for line in rd:
            if line[1] == "false":
                cleaned_line = clean_text(line[2])
                fake.append(cleaned_line)
            elif line[1] == "true":
                cleaned_line = clean_text(line[2])
                truth.append(cleaned_line)
        return fake,truth


def clean_text(inputext):
    """
        Clean text removing stop words, punctuation, non alphanumeric characters and digits
        :param inputext: line of text
        :return: list of cleaned tokens
    """
    lmtzr = WordNetLemmatizer()
    tokens = word_tokenize(inputext.lower())
    stops = set(stopwords.words('english')) | set(string.punctuation)
    return [lmtzr.lemmatize(word) for word in tokens if
            word and word not in stops and not word.isdigit() and word.isalpha()]


def load_dataset(datapath):
    """
        Find path of files
        :param datapath: path of dataset
        :return: fake file paths, truth file paths
    """
    fake = []
    truth = []
    for (dirpath, dirnames, filenames) in walk(datapath):
        if "deceptive" in dirpath:
            fake.extend(os.path.join(dirpath, filename) for filename in filenames)
        else:
            truth.extend(os.path.join(dirpath, filename) for filename in filenames)
    return fake, truth


def preprocess(filepaths):
    """
        Clean and write file
        :param filepaths: list of file paths
        :return:
    """
    for filepath in filepaths:
        f = read_file(filepath)

        for line in f:
            cleaned = clean_text(line)
            if "deceptive" in filepath:
                folder = "fake"
            else:
                folder = "truth"
            write_file(str(c.CLEANED_DIR / folder / str(folder + ".txt")), ' '.join(cleaned))

def preprocess_liar(fake,truth,outpu_fake,output_truth):
    """
        Preprocess liar dataset
        :param fake: list of fake documents
        :param truth: list of truth documents
        :param outpu_fake: output path for fake
        :param output_truth: output path for truth
        :return:
    """
    for line in fake:
        write_file(str(outpu_fake), ' '.join(line))
    for line in truth:
        write_file(str(output_truth),' '.join(line))


def main():
    fake, truth = load_dataset(c.DATA_DIR)
    preprocess(fake)
    preprocess(truth)


if __name__ == "__main__":
    main()
