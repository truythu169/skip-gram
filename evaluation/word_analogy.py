from embedding import Embedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class WordAnalogy:

    def __init__(self, filename='datasets/word_analogy/google_analogy.txt'):
        with open(filename, "r") as f:
            L = f.read().splitlines()

        # Simple 4 word analogy questions with categories
        questions = []
        answers = []
        category = []
        cat = None
        for l in L:
            l = l.lower()
            if l.startswith(":"):
                cat = l.split()[1]
            else:
                words = l.split()
                questions.append(words[0:3])
                answers.append(words[3])
                category.append(cat)

        syntactic = set([c for c in set(category) if c.startswith("gram")])
        category_high_level = []
        for cat in category:
            category_high_level.append("syntactic" if cat in syntactic else "semantic")

        self.X = np.array(questions)
        self.y = np.array(answers)
        self.category = np.array(category)
        self.category_high_level = np.array(category_high_level)

    def get_data_by_category(self, cat, high_level_category=False):
        if high_level_category:
            data_indexes = np.where(self.category_high_level == cat)[0]
        else:
            data_indexes = np.where(self.category == cat)[0]
        return self.X[data_indexes], self.y[data_indexes]

    def evaluate(self, embedding, high_level_category=False):
        # Categories list
        if high_level_category:
            cat_list = set(self.category_high_level)
        else:
            cat_list = set(self.category)

        # Devide data into categories
        X = {}
        labels = {}
        skip_lines = 0
        for cat in cat_list:
            X_cat, y_cat = self.get_data_by_category(cat, high_level_category)
            skipped_labels = []
            skipped_X = []

            # convert all words to int and skip words not exist in vocab
            for i in range(len(X_cat)):
                x = X_cat[i]
                y = y_cat[i]

                if embedding.in_vocabs(x) and embedding.in_vocab(y):
                    skipped_X.append(embedding.indexes(x))
                    skipped_labels.append(embedding.index(y))
                else:
                    skip_lines += 1

            X[cat] = skipped_X
            labels[cat] = skipped_labels
        print('Skipped {} lines.'.format(skip_lines))

        # Predict answer vector
        predictions = {}
        for cat in cat_list:
            X_cat, y_cat = X[cat], labels[cat]
            pred_vectors = []

            for x in X_cat:
                x = embedding.vectors(x)
                pred_vector = x[1] - x[0] + x[2]
                pred_vectors.append(pred_vector)

            # Get cosine similarity of predicted answer to all words in vocab
            pred_vectors = np.array(pred_vectors)
            distance_matrix = cosine_similarity(pred_vectors, embedding.e)

            # Remove words that were originally in the query
            for i in range(len(X_cat)):
                distance_matrix[i][X_cat[i]] = 0

            # Get nearest word
            pred_indexes = np.argmax(distance_matrix, axis=1)

            # accuracy
            acc = np.mean(pred_indexes == y_cat)

            # result
            print("Category: %-30s, accuracy: %f" % (cat, acc))
            predictions[cat] = acc

        # overrall
        total_count = 0
        acc = 0
        for cat in cat_list:
            cat_count = len(X[cat])
            acc += cat_count * predictions.get(cat)
            total_count += cat_count
        predictions['all'] = acc / total_count
        print("All Category accuracy: %f" % (acc / total_count))

        return predictions


if __name__ == "__main__":
    word_analogy = WordAnalogy()
    embedding = Embedding.from_file('../output/300dim/embedding.txt')
    result = word_analogy.evaluate(embedding)
    print(result)
