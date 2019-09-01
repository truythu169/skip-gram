import numpy as np
import tensorflow as tf
import utils.tools as utils


class Model:

    # Constructor
    def __init__(self, model_path, sample_path, n_files, n_train_sample=1000):
        # Load parameters
        self.embedding = utils.load_pkl(model_path + '/embedding.pkl')
        self.softmax_w = utils.load_pkl(model_path + '/softmax_w.pkl')
        self.softmax_b = utils.load_pkl(model_path + '/softmax_b.pkl')
        self.n_vocab = self.embedding.shape[0]
        self.n_embedding = self.embedding.shape[1]
        self.n_context = self.softmax_w.shape[0]

        # paths
        self.data_path = model_path
        self.output_path = model_path + '/output'
        self.sample_path = sample_path
        self.n_files = n_files

        # sample data
        self.n_train_sample = n_train_sample

    def set_weights(self, embedding, softmax_w, softmax_b):
        self.embedding = embedding
        self.softmax_w = softmax_w
        self.softmax_b = softmax_b

    def train(self, word, context, n_neg_sample=200, epochs=10, update_weigh=True):
        print('Start training...')
        prob, losses = self._train_sample(word, context, n_neg_sample, epochs, update_weigh)
        print('Finished!')
        return prob, losses

    def _new_training_sample(self):
        words, contexts = utils.sample_learning_data(self.sample_path, self.n_files, self.n_train_sample)
        self.n_train_sample = self.n_train_sample
        self.words = words
        self.contexts = contexts

    def _get_sample_data(self, word, context):
        words, contexts = utils.sample_learning_data(self.sample_path, self.n_files, self.n_train_sample)

        # sampling sample from train data
        words = words + [word]
        contexts = contexts + [context]

        return words, contexts

    def _train_sample(self, word, context, n_sampled=200, epochs=10, update_weigh=False):
        # create samples training set
        words = []
        contexts = []
        for e in range(epochs):
            w, c = self._get_sample_data(word, context)
            words.append(w)
            contexts.append(c)

        # computation graph
        train_graph = tf.Graph()
        with train_graph.as_default():
            # training data
            # input placeholders
            inputs = tf.placeholder(tf.int32, [None], name='inputs')
            labels = tf.placeholder(tf.int32, [None, None], name='labels')

            # embedding layer
            embedding = tf.get_variable("embedding", initializer=self.embedding)
            embed = tf.nn.embedding_lookup(embedding, inputs)

            # softmax layer
            softmax_w = tf.get_variable("softmax_w", initializer=self.softmax_w)
            softmax_b = tf.get_variable("softmax_b", initializer=self.softmax_b)

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

            # conditional probability of word given contexts
            mul = tf.matmul(embed, tf.transpose(softmax_w))
            logits = tf.reshape(tf.exp(mul + softmax_b), [-1])
            sum_logits = tf.reduce_sum(logits)
            prob = tf.gather(logits, tf.reshape(labels, [-1])) / sum_logits

        # Run optimizer
        with tf.Session(graph=train_graph) as sess:
            losses = []
            sess.run(tf.global_variables_initializer())

            # train weights
            for e in range(1, epochs + 1):
                feed = {inputs: words[e-1],
                        labels: np.array(contexts[e-1])[:, None]}

                train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                losses.append(train_loss)

                feed = {inputs: [word], labels: [[context]]}
                p = sess.run(prob, feed_dict=feed)
                print(p)

            # estimate conditional probability of word given contexts
            feed = {inputs: [word], labels: [[context]]}
            p = sess.run(prob, feed_dict=feed)

            # update weights
            if update_weigh:
                self.embedding = embedding.eval()
                self.softmax_w = softmax_w.eval()
                self.softmax_b = softmax_b.eval()

        return p, losses
