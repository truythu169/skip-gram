import numpy as np
import utils.tools as utils
import utils.math as math


class Model:

    def __init__(self, word_size=1000, context_size=1000, embed_size=100):
        self.E = np.random.rand(word_size, embed_size)
        self.C = np.random.rand(embed_size, context_size)
        self.b = np.random.rand(context_size)
        self.V = word_size
        self.K = embed_size
        self.V_dash = context_size

    def load_model(self, data_path):
        self.E = utils.load_pkl(data_path + '/embedding.pkl')
        self.C = utils.load_pkl(data_path + '/softmax_w.pkl')
        self.b = utils.load_pkl(data_path + '/softmax_b.pkl')
        self.V = self.E.shape[0]
        self.K = self.E.shape[1]
        self.V_dash = self.C.shape[0]

    def train(self, w, c, alpha=0.025):
        # forward propagation
        e = self.E[w]  # K dimensions vector
        z = np.dot(e, self.C.T) + self.b
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z)
        y = exp_z / sum_exp_z

        # compute loss
        loss = - np.log(y[c])

        # back propagation
        dz = exp_z / sum_exp_z
        dz[c] -= 1
        dC = np.dot(dz.reshape(-1, 1), e.reshape(1, -1))
        db = dz
        de = np.dot(dz.reshape(1, -1), self.C)

        # update weights
        self.E[w] -= alpha * de.reshape(-1)
        self.C -= alpha * dC
        self.b -= alpha * db

        # compute loss
        return loss, y[c]

    def train_neg(self, w, c, neg, loss_log=False, alpha=0.025):
        # indicate for true label
        indicate = - np.ones(len(neg) + 1)
        indicate[0] = 1

        # forward propagation
        e = self.E[w] # K dimensions vector
        labels = [c] + neg
        z = np.dot(e, self.C[labels].T) * indicate
        z_sigmoid = math.sigmoid(z)

        # back propagation
        db = (z_sigmoid - 1) * indicate # (neg_size + 1) dimensions vector
        dC = np.dot(db.reshape(-1, 1), e.reshape(1, -1)) # (neg_size + 1) x K
        dE = np.dot(db, self.C[labels]).reshape(-1)

        # update weights
        self.b[labels] = self.b[labels] - alpha * db
        self.C[labels] = self.C[labels] - alpha * dC
        self.E[w] = e - alpha * dE

        # compute loss
        if loss_log:
            loss = - np.sum(np.log(z_sigmoid))
            # probability
            z = np.dot(e, self.C.T)
            y = math.softmax(z)
            prob = y[c]
            return loss, prob

    def prob_snml(self, word, context):
        # forward propagation
        word_vector = self.E[word]
        context_vectors = 1


