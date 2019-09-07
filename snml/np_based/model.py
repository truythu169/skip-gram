import numpy as np
import utils.tools as utils
import utils.math as math
import math as ma


class Model:

    def __init__(self, data_path, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.E = utils.load_pkl(data_path + 'embedding.pkl')
        self.C = utils.load_pkl(data_path + 'softmax_w.pkl')
        self.b = utils.load_pkl(data_path + 'softmax_b.pkl')
        self.V = self.E.shape[0]
        self.K = self.E.shape[1]
        self.V_dash = self.C.shape[0]
        self.data_path = data_path

        # adam optimizer initialize
        self.t = 256040
        self.t_default = 256040
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = learning_rate
        self.epsilon = epsilon
        self.beta1_t = beta1 ** self.t
        self.beta2_t = beta2 ** self.t

        # default
        self.E_default = self.E.copy()
        self.C_default = self.C.copy()
        self.b_default = self.b.copy()

    def reset(self):
        self.E = self.E_default.copy()
        self.C = self.C_default.copy()
        self.b = self.b_default.copy()

        # adam optimizer initialize
        self.t = self.t_default
        self.beta1_t = self.beta1 ** self.t
        self.beta2_t = self.beta2 ** self.t

    def train_adam(self, w, c, epochs=20):
        # initialize things
        self.mE_t = np.zeros(self.K)
        self.mC_t = np.zeros((self.V_dash, self.K))
        self.mb_t = np.zeros(self.V_dash)
        self.vE_t = np.zeros(self.K)
        self.vC_t = np.zeros((self.V_dash, self.K))
        self.vb_t = np.zeros(self.V_dash)

        prob = 0
        losses = []
        for i in range(epochs):
            loss, prob = self._train_adam(w, c)
            losses.append(loss)

        return prob, losses

    def _train_adam(self, w, c):
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
        dE = np.dot(dz.reshape(1, -1), self.C).reshape(-1)

        # adam step
        self.t = self.t + 1
        self.beta1_t = self.beta1_t * self.beta1
        self.beta2_t = self.beta2_t * self.beta2

        # adam things
        lr = self.lr * ma.sqrt(1 - self.beta2_t) / (1 - self.beta1_t)
        mE = self.beta1 * self.mE_t + (1 - self.beta1) * dE
        mC = self.beta1 * self.mC_t + (1 - self.beta1) * dC
        mb = self.beta1 * self.mb_t + (1 - self.beta1) * db
        vE = self.beta2 * self.vE_t + (1 - self.beta2) * dE * dE
        vC = self.beta2 * self.vC_t + (1 - self.beta2) * dC * dC
        vb = self.beta2 * self.vb_t + (1 - self.beta2) * db * db

        # update weights
        self.E[w] -= lr * mE / (np.sqrt(vE + self.epsilon))
        self.C -= lr * mC / (np.sqrt(vC + self.epsilon))
        self.b -= lr * mb / (np.sqrt(vb + self.epsilon))

        # save status
        self.mE_t = mE
        self.mC_t = mC
        self.mb_t = mb
        self.vE_t = vE
        self.vC_t = vC
        self.vb_t = vb

        # compute loss
        return loss, y[c]

    def train_neg_adam(self, w, c, epochs=20, neg_size=200):
        # initialize things
        self.mE_t = np.zeros(self.K)
        self.mC_t = np.zeros((neg_size + 1, self.K))
        self.mb_t = np.zeros(neg_size + 1)
        self.vE_t = np.zeros(self.K)
        self.vC_t = np.zeros((neg_size + 1, self.K))
        self.vb_t = np.zeros(neg_size + 1)

        losses = []
        for i in range(epochs - 1):
            neg = utils.sample_negative(neg_size, {c}, vocab_size=self.V_dash)
            loss, prob = self._train_neg_adam(w, c, neg)
            losses.append(loss)

        neg = utils.sample_negative(neg_size, {c}, vocab_size=self.V_dash)
        loss, prob = self._train_neg_adam(w, c, neg, get_prob=True)
        losses.append(loss)

        return prob, losses

    def _train_neg_adam(self, w, c, neg, get_prob=False):
        # forward propagation
        e = self.E[w] # K dimensions vector
        labels = [c] + neg
        z = np.dot(e, self.C[labels].T) + self.b[labels]
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z)

        # back propagation
        dz = exp_z / sum_exp_z
        dz[0] -= 1 # for true label
        dz = dz / 10000
        dC = np.dot(dz.reshape(-1, 1), e.reshape(1, -1))
        db = dz
        dE = np.dot(dz.reshape(1, -1), self.C[labels]).reshape(-1)

        # adam step
        self.t = self.t + 1
        self.beta1_t = self.beta1_t * self.beta1
        self.beta2_t = self.beta2_t * self.beta2

        # adam things
        lr = self.lr * ma.sqrt(1 - self.beta2_t) / (1 - self.beta1_t)
        mE = self.beta1 * self.mE_t + (1 - self.beta1) * dE
        mC = self.beta1 * self.mC_t + (1 - self.beta1) * dC
        mb = self.beta1 * self.mb_t + (1 - self.beta1) * db
        vE = self.beta2 * self.vE_t + (1 - self.beta2) * dE * dE
        vC = self.beta2 * self.vC_t + (1 - self.beta2) * dC * dC
        vb = self.beta2 * self.vb_t + (1 - self.beta2) * db * db

        # update weights
        self.E[w] -= lr * mE / (np.sqrt(vE + self.epsilon))
        self.C[labels] -= lr * mC / (np.sqrt(vC + self.epsilon))
        self.b[labels] -= lr * mb / (np.sqrt(vb + self.epsilon))

        # save status
        self.mE_t = mE
        self.mC_t = mC
        self.mb_t = mb
        self.vE_t = vE
        self.vC_t = vC
        self.vb_t = vb

        # compute loss
        loss = - np.log(exp_z[0] / sum_exp_z)

        # probability
        prob = None
        if get_prob:
            z = np.dot(e, self.C.T)
            y = math.softmax(z)
            prob = y[c]

        return loss, prob

