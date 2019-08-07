import time
import numpy as np
import tensorflow as tf
import utils.tools as utils
from utils.raw_dataset import Dataset
import os


class SkipGram:

    # Constructor
    def __init__(self, filename, n_embedding):
        self.n_embedding = n_embedding

        self.output_dictionary = 'output/{}dim'.format(n_embedding)
        if not os.path.exists(self.output_dictionary):
            os.makedirs(self.output_dictionary)
            os.makedirs(self.output_dictionary + '/dict')
            os.makedirs(self.output_dictionary + '/checkpoints')

        data = Dataset(filename, self.output_dictionary + '/dict')

        self.data = data
        self.n_vocab = data.n_vocab
        self.n_context = data.n_context
        self.embedding = np.array([])

    def train(self, n_sampled=200, epochs=10, batch_size=1000, window_size=5, eval_mode=False):
        # computation graph
        train_graph = tf.Graph()

        with train_graph.as_default():
            # placeholders
            inputs = tf.placeholder(tf.int32, [None], name='inputs')
            labels = tf.placeholder(tf.int32, [None, None], name='labels')

            # embedding layer
            embedding = tf.Variable(tf.random_uniform((self.n_vocab, self.n_embedding), -1, 1))
            embed = tf.nn.embedding_lookup(embedding, inputs)

            # softmax layer
            softmax_w = tf.Variable(tf.truncated_normal((self.n_context, self.n_embedding)))
            softmax_b = tf.Variable(tf.zeros(self.n_context))

            # Calculate the loss using negative sampling
            loss = tf.nn.sampled_softmax_loss(
                weights=softmax_w,
                biases=softmax_b,
                labels=labels,
                inputs=embed,
                num_sampled=n_sampled,
                num_classes=self.n_context)

            # True loss
            with tf.device("/device:CPU:0"):
                logits = tf.matmul(embed, tf.transpose(softmax_w)) + softmax_b
                true_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                true_cost = tf.reduce_mean(true_loss)

            cost = tf.reduce_mean(loss)
            optimizer = tf.train.AdamOptimizer().minimize(cost)

            saver = tf.train.Saver()

        with tf.Session(graph=train_graph) as sess:
            iteration = 1
            loss = 0
            sess.run(tf.global_variables_initializer())
            loss_his = []

            for e in range(1, epochs + 1):
                batches = self.data.get_batches(batch_size, window_size)
                start = time.time()
                for x, y in batches:
                    feed = {inputs: x,
                            labels: np.array(y)[:, None]}

                    train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

                    loss += train_loss

                    if iteration % 100 == 0:
                        end = time.time()
                        print("Epoch {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(loss / 100),
                              "{:.4f} sec/batch".format((end - start) / 100))
                        if eval_mode:
                            binary_labels = utils.label_binarizer(y, self.n_vocab)
                            true_cost_ = sess.run(true_cost, feed_dict={inputs: x, labels: binary_labels})
                            print("True cost: {}".format(true_cost_))
                            loss_his.append(true_cost_)
                        loss = 0
                        start = time.time()

                    iteration += 1

            # export embedding matrix
            self.embedding = embedding.eval()

            # save model to files
            saver.save(sess, self.output_dictionary + "/checkpoints/text8.ckpt")

        # Save loss history
        if eval_mode:
            with open('loss/loss-hist-{}.txt'.format(self.n_embedding), 'w') as f:
                for item in loss_his:
                    f.write(str(item)+'\n')

    def export_embedding(self):
        # write embedding result to file
        output = open(self.output_dictionary + '/embedding.txt', 'w')
        for i in range(self.embedding.shape[0]):
            text = self.data.int_to_vocab[i]
            for j in self.embedding[i]:
                text += ' %f' % j
            text += '\n'
            output.write(text)

        output.close()


if __name__ == "__main__":
    skip_gram = SkipGram('data/text8', n_embedding=300)
    skip_gram.train(n_sampled=200, epochs=10, batch_size=1000, window_size=5, eval_mode=False)
    skip_gram.export_embedding()
