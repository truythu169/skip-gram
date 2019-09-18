import numpy as np
import utils.tools as utils
import utils.math as math
import math as ma
import time


class Model:

    def __init__(self, data_path, context_distribution_file, n_context_sample,
                 learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
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

        # initialize things
        self.mE_t = np.zeros((self.V, self.K))
        self.mC_t = np.zeros((self.V_dash, self.K))
        self.mb_t = np.zeros(self.V_dash)
        self.vE_t = np.zeros((self.V, self.K))
        self.vC_t = np.zeros((self.V_dash, self.K))
        self.vb_t = np.zeros(self.V_dash)

        # sampling snml
        self.sample_contexts, self.sample_contexts_prob = utils.sample_contexts(context_distribution_file,
                                                                                n_context_sample)
        self.n_context_sample = n_context_sample

    def get_prob(self, word, context):
        # forward propagation
        e = self.E[word]  # K dimensions vector
        z = np.dot(e, self.C.T) + self.b
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z)
        y = exp_z / sum_exp_z

        return y[context]

    def get_neg_prob(self, word, context, neg_size=200):
        neg = utils.sample_negative(neg_size, {context}, vocab_size=self.V_dash)

        # forward propagation
        e = self.E[word]  # K dimensions vector
        labels = [context] + neg
        z = np.dot(e, self.C[labels].T) + self.b[labels]
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z)
        prob = exp_z[0] / sum_exp_z

        return prob

    def snml_length(self, word, context, epochs=20):
        prob_sum = 0
        iteration = 0

        # Update all other context
        for c in range(self.V_dash):
            if c != context:
                iteration += 1
                prob, losses = self.train_neg_adam(word, c, epochs)

                prob_sum += prob

        # Update true context and save weights
        prob, losses = self.train_neg_adam(word, context, epochs, update_weights=True)
        prob_sum += prob
        snml_length = - np.log(prob / prob_sum)
        return snml_length

    def snml_length_sampling(self, word, context, epochs=20):
        prob_sum = 0
        iteration = 0

        # Update all other context
        for i in range(self.n_context_sample):
            c = self.sample_contexts[i]
            c_prob = self.sample_contexts_prob[i]

            iteration += 1
            prob, losses = self.train_neg_adam(word, c, epochs)
            prob_sum += prob / c_prob
        prob_sum = prob_sum / self.n_context_sample

        # Update true context and save weights
        prob, losses = self.train_neg_adam(word, context, epochs, update_weights=True)
        snml_length = - np.log(prob / prob_sum)

        return snml_length, prob_sum

    def _copy_weights(self, w):
        # copy weights to train
        self.e_train = self.E[w].copy()
        self.C_train = self.C.copy()
        self.b_train = self.b.copy()

        self.me_train = self.mE_t[w].copy()
        self.mC_train = self.mC_t.copy()
        self.mb_train = self.mb_t.copy()

        self.ve_train = self.vE_t[w].copy()
        self.vC_train = self.vC_t.copy()
        self.vb_train = self.vb_t.copy()

        self.t_train = self.t
        self.beta1_train = self.beta1_t
        self.beta2_train = self.beta2_t

    def _update_weights(self, w):
        # update training things back
        self.E[w] = self.e_train
        self.C = self.C_train
        self.b = self.b_train

        self.mE_t[w] = self.me_train
        self.mC_t = self.mC_train
        self.mb_t = self.mb_train

        self.vE_t[w] = self.ve_train
        self.vC_t = self.vC_train
        self.vb_t = self.vb_train

        self.t = self.t_train
        self.beta1_t = self.beta1_train
        self.beta2_t = self.beta2_train

    def train_adam(self, w, c, epochs=20, update_weights=False):
        self._copy_weights(w)

        prob = 0
        losses = []
        for i in range(epochs):
            loss, prob = self._train_adam(w, c)
            losses.append(loss)

        if update_weights:
            self._update_weights(w)

        return prob, losses

    def _train_adam(self, w, c):
        # forward propagation
        e = self.e_train
        z = np.dot(e, self.C_train.T) + self.b_train
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z)
        y = exp_z / sum_exp_z

        # compute loss
        loss = - np.log(y[c])

        # back propagation
        dz = exp_z / sum_exp_z
        dz[c] -= 1
        dz = dz / 10000
        dC = np.dot(dz.reshape(-1, 1), e.reshape(1, -1))
        db = dz
        dE = np.dot(dz.reshape(1, -1), self.C_train).reshape(-1)

        # adam step
        self.t_train = self.t_train + 1
        self.beta1_train = self.beta1_train * self.beta1
        self.beta2_train = self.beta2_train * self.beta2

        # adam things
        lr = self.lr * ma.sqrt(1 - self.beta2_train) / (1 - self.beta1_train)
        mE = self.beta1 * self.me_train + (1 - self.beta1) * dE
        mC = self.beta1 * self.mC_train + (1 - self.beta1) * dC
        mb = self.beta1 * self.mb_train + (1 - self.beta1) * db
        vE = self.beta2 * self.ve_train + (1 - self.beta2) * dE * dE
        vC = self.beta2 * self.vC_train + (1 - self.beta2) * dC * dC
        vb = self.beta2 * self.vb_train + (1 - self.beta2) * db * db

        # update weights
        self.e_train -= lr * mE / (np.sqrt(vE + self.epsilon))
        self.C_train -= lr * mC / (np.sqrt(vC + self.epsilon))
        self.b_train -= lr * mb / (np.sqrt(vb + self.epsilon))

        # save status
        self.me_train = mE
        self.mC_train = mC
        self.mb_train = mb
        self.ve_train = vE
        self.vC_train = vC
        self.vb_train = vb

        # compute loss
        return loss, y[c]

    def train_neg_adam(self, w, c, epochs=20, neg_size=200, update_weights=False):
        self._copy_weights(w)

        prob = 0
        losses = []
        for i in range(epochs):
            neg = utils.sample_negative(neg_size, {c}, vocab_size=self.V_dash)
            prob, loss = self._train_neg_adam(w, c, neg)
            losses.append(loss)

        if update_weights:
            self._update_weights(w)

        return prob, losses

    def _train_neg_adam(self, w, c, neg):
        # forward propagation
        e = self.e_train # K dimensions vector
        labels = [c] + neg
        z = np.dot(e, self.C_train[labels].T) + self.b_train[labels]
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z)

        # back propagation
        dz = exp_z / sum_exp_z
        dz[0] -= 1 # for true label
        dz = dz / 10000
        dC = np.dot(dz.reshape(-1, 1), e.reshape(1, -1))
        db = dz
        dE = np.dot(dz.reshape(1, -1), self.C_train[labels]).reshape(-1)

        # adam step
        self.t_train = self.t + 1
        self.beta1_train = self.beta1_train * self.beta1
        self.beta2_train = self.beta2_train * self.beta2

        # adam things
        lr = self.lr * ma.sqrt(1 - self.beta2_t) / (1 - self.beta1_t)
        mE = self.beta1 * self.me_train + (1 - self.beta1) * dE
        mC = self.beta1 * self.mC_train[labels] + (1 - self.beta1) * dC
        mb = self.beta1 * self.mb_train[labels] + (1 - self.beta1) * db
        vE = self.beta2 * self.ve_train + (1 - self.beta2) * dE * dE
        vC = self.beta2 * self.vC_train[labels] + (1 - self.beta2) * dC * dC
        vb = self.beta2 * self.vb_train[labels] + (1 - self.beta2) * db * db

        # update weights
        self.e_train -= lr * mE / (np.sqrt(vE + self.epsilon))
        self.C_train[labels] -= lr * mC / (np.sqrt(vC + self.epsilon))
        self.b_train[labels] -= lr * mb / (np.sqrt(vb + self.epsilon))

        # save status
        self.me_train = mE
        self.mC_train[labels] = mC
        self.mb_train[labels] = mb
        self.ve_train = vE
        self.vC_train[labels] = vC
        self.vb_train[labels] = vb

        # compute loss
        prob = exp_z[0] / sum_exp_z
        loss = - np.log(prob)

        return prob, loss

