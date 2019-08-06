import utils.tools as utils
import numpy as np


class Dataset:

    # Constructor
    def __init__(self, data_file, dict_path):
        # read data from file
        print('Reading file: ', data_file)
        with open(data_file) as f:
            text = f.read()

        # about our data
        words = utils.preprocess(text)

        # word to int and int to word dictionaries, convert words to list of int
        # 0: vocab_to_int, 1: int_to_vocab, 2: cont_to_int, 3: int_to_cont
        dicts = utils.create_lookup_tables(words)
        int_words = [dicts[2][word] for word in words]

        # subsampling
        train_words = utils.get_train_words(int_words)

        # set class attributes
        self.n_vocab = len(dicts[1])
        self.n_context = len(dicts[3])
        self.vocab_to_int = dicts[0]
        self.int_to_vocab = dicts[1]
        self.cont_to_int = dicts[2]
        self.int_to_cont = dicts[3]
        self.words = train_words

        # Save dictionaries
        utils.save_dict_to_file(dicts[0], dict_path + '/vocab_to_int.dict')
        utils.save_dict_to_file(dicts[1], dict_path + '/int_to_vocab.dict')

        print("Total words: {}".format(len(words)))
        print("Unique words: {}".format(self.n_vocab))
        print("Unique context: {}".format(self.n_context))
        print("Data Prepared!")

    def convert_context_to_word(self, context_id):
        word = self.int_to_cont[context_id]
        if word in self.vocab_to_int:
            return self.vocab_to_int[word]
        else:
            return False

    def get_target(self, words, idx, window_size=5):
        """ Get a list of words in a window around an index. """
        R = np.random.randint(1, window_size + 1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_words = set(words[start:idx] + words[idx + 1:stop + 1])

        return list(target_words)

    def get_batches(self, batch_size, window_size=5):
        """ Create a generator of word batches as a tuple (inputs, targets) """
        n_batches = len(self.words) // batch_size

        # only full batches
        words = self.words[:n_batches * batch_size]

        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx:idx + batch_size]
            for ii in range(len(batch)):
                batch_x = self.convert_context_to_word(batch[ii])
                if batch_x != False:
                    batch_y = self.get_target(batch, ii, window_size)
                    y.extend(batch_y)
                    x.extend([batch_x] * len(batch_y))
            yield x, y