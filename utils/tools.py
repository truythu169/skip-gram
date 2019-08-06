import numpy as np
from collections import Counter
import random
import pickle
from nltk.corpus import stopwords


def preprocess(text):
    # Replace punctuation with tokens so we can use them in our model
    # text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]

    return trimmed_words


def get_train_words(int_words):
    # implementation of subsampling
    threshold = 1e-5
    word_counts = Counter(int_words)
    total_count = len(int_words)
    freqs = {word: count / total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts}
    train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

    return train_words


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict....
    """
    # Load stop words
    stop_words = stopwords.words('english')

    # dict for contexts
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_cont = {ii: word for ii, word in enumerate(sorted_vocab)}
    cont_to_int = {word: ii for ii, word in int_to_cont.items()}

    # dict for words
    non_stopwords_words = [word for word in words if word.lower() not in stop_words]
    word_counts = Counter(non_stopwords_words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return [vocab_to_int, int_to_vocab, cont_to_int, int_to_cont]


def save_dict_to_file(dict, filename):
    """ Save dictionary to file """
    output = open(filename, 'wb')
    pickle.dump(dict, output, pickle.HIGHEST_PROTOCOL)
    output.close()


def load_dict_from_file(filename):
    """ Load dictionary to file """
    input = open(filename, 'rb')
    dict = pickle.load(input)
    input.close()
    return dict


def label_binarizer(labels, n_class):
    """ Convert dense labels array to sparse labels matrix """
    n_records = len(labels)
    labels_b = np.zeros((n_records, n_class))
    labels_b[np.arange(n_records), labels] = 1

    return  labels_b
