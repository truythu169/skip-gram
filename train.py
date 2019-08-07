import time
import numpy as np
import tensorflow as tf
import utils.tools as utils
import os


class SkipGram:

    # Constructor
    def __init__(self, data_path, n_embedding):
        self.n_embedding = n_embedding
        self.embedding = np.array([])
        self.ident = '-embedding={}'.format(n_embedding)
        self.data_path = data_path

        # create output directory
        self.output_dictionary = 'output/{}dim'.format(n_embedding)
        if not os.path.exists(self.output_dictionary):
            os.makedirs(self.output_dictionary)

        # read dictionaries
        self.int_to_vocab = utils.load_dict_from_file(data_path + '/dict/int_to_vocab.dict')
        self.int_to_cont = utils.load_dict_from_file(data_path + '/dict/int_to_cont.dict')
        self.n_vocab = len(self.int_to_vocab)
        self.n_context = len(self.int_to_cont)

    def train(self, n_sampled=200, epochs=1, batch_size=10000):
        self.ident += '-{}-{}'.format(n_sampled, epochs)

        # computation graph
        train_graph = tf.Graph()

        with train_graph.as_default():
            # training data
            dataset = tf.data.experimental.make_csv_dataset(self.data_path + '/data.csv',
                                                            batch_size=batch_size,
                                                            column_names=['input', 'output'],
                                                            header=False,
                                                            num_epochs=epochs)
            datum = dataset.make_one_shot_iterator().get_next()
            inputs, labels = datum['input'], datum['output']

            # embedding layer
            embedding = tf.Variable(tf.random_uniform((self.n_vocab, self.n_embedding), -1, 1))
            embed = tf.nn.embedding_lookup(embedding, inputs)

            # softmax layer
            softmax_w = tf.Variable(tf.truncated_normal((self.n_context, self.n_embedding)))
            softmax_b = tf.Variable(tf.zeros(self.n_context))

            # Calculate the loss using negative sampling
            labels = tf.reshape(labels, [-1, 1])
            loss = tf.nn.sampled_softmax_loss(
                weights=softmax_w,
                biases=softmax_b,
                labels=labels,
                inputs=embed,
                num_sampled=n_sampled,
                num_classes=self.n_context)

            cost = tf.reduce_mean(loss)
            optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session(graph=train_graph) as sess:
            iteration = 1
            loss = 0
            sess.run(tf.global_variables_initializer())

            try:
                start = time.time()
                while True:
                    train_loss, _ = sess.run([cost, optimizer])
                    loss += train_loss

                    if iteration % 100 == 0:
                        end = time.time()
                        print("Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(loss / 100),
                              "{:.4f} sec/ 1.000.000 sample".format((end - start)))
                        loss = 0
                        start = time.time()

                    iteration += 1
            except tf.errors.OutOfRangeError:
                print("End of dataset")

            # export embedding matrix
            self.embedding = embedding.eval()

    def export_embedding(self):
        # write embedding result to file
        output = open(self.output_dictionary + '/embedding{}.txt'.format(self.ident), 'w')
        for i in range(self.embedding.shape[0]):
            text = self.int_to_vocab[i]
            for j in self.embedding[i]:
                text += ' %f' % j
            text += '\n'
            output.write(text)

        output.close()


if __name__ == "__main__":
    skip_gram = SkipGram('data/processed data', n_embedding=100)
    skip_gram.train(n_sampled=200, epochs=10, batch_size=10000)
    skip_gram.export_embedding()
