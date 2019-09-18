from snml.np_based.model import Model
import time
import numpy as np
import utils.tools as utils
import math
from multiprocessing import Pool


t = 256040
beta1 = 0.9
beta2 = 0.999
lr = 0.001
epsilon = 1e-08


def train_neg_adam(c, epochs, neg_size, e, C_train, b_train, me_train, mC_train, mb_train,
                   ve_train, vC_train, vb_train, t_train, beta1_train, beta2_train, V_dash, lr):
    # start epochs
    for i in range(epochs):
        neg = utils.sample_negative(neg_size, {c}, vocab_size=V_dash)

        # forward propagation
        labels = [c] + neg
        z = np.dot(e, C_train[labels].T) + b_train[labels]
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z)

        # back propagation
        dz = exp_z / sum_exp_z
        dz[0] -= 1  # for true label
        dz = dz / 100000
        dC = np.dot(dz.reshape(-1, 1), e.reshape(1, -1))
        db = dz
        dE = np.dot(dz.reshape(1, -1), C_train[labels]).reshape(-1)

        # adam step
        t_train = t_train + 1
        beta1_train = beta1_train * beta1
        beta2_train = beta2_train * beta2

        # adam things
        lr = lr * math.sqrt(1 - beta2_train) / (1 - beta1_train)
        mE = beta1 * me_train + (1 - beta1) * dE
        mC = beta1 * mC_train[labels] + (1 - beta1) * dC
        mb = beta1 * mb_train[labels] + (1 - beta1) * db
        vE = beta2 * ve_train + (1 - beta2) * dE * dE
        vC = beta2 * vC_train[labels] + (1 - beta2) * dC * dC
        vb = beta2 * vb_train[labels] + (1 - beta2) * db * db

        # update weights
        e -= lr * mE / (np.sqrt(vE + epsilon))
        C_train[labels] -= lr * mC / (np.sqrt(vC + epsilon))
        b_train[labels] -= lr * mb / (np.sqrt(vb + epsilon))

        # save status
        me_train = mE
        mC_train[labels] = mC
        mb_train[labels] = mb
        ve_train = vE
        vC_train[labels] = vC
        vb_train[labels] = vb

    # get probability
    neg = utils.sample_negative(neg_size, {c}, vocab_size=V_dash)
    labels = [c] + neg
    z = np.dot(e, C_train[labels].T) + b_train[labels]
    exp_z = np.exp(z)
    prob = exp_z[0] / np.sum(exp_z)

    return c


def product_train(args):
    return train_neg_adam(*args)


if __name__ == "__main__":
    # E = utils.load_pkl('../models/100dim/' + 'embedding.pkl')
    # C = utils.load_pkl('../models/100dim/' + 'softmax_w.pkl')
    # b = utils.load_pkl('../models/100dim/' + 'softmax_b.pkl')
    # V = E.shape[0]
    # K = E.shape[1]
    # V_dash = C.shape[0]
    #
    # # adam optimizer initialize
    # beta1_t = beta1 ** t
    # beta2_t = beta2 ** t
    #
    # # initialize things
    # mE_t = np.zeros((V, K))
    # mC_t = np.zeros((V_dash, K))
    # mb_t = np.zeros(V_dash)
    # vE_t = np.zeros((V, K))
    # vC_t = np.zeros((V_dash, K))
    # vb_t = np.zeros(V_dash)
    #
    # start = time.time()
    # job_args = [(i, 20, 200, E[8229], C, b, mE_t[8229], mC_t, mb_t, vE_t[8229], vC_t, vb_t, t, beta1_t, beta2_t,
    #              V_dash, lr) for i in range(3000)]
    # end = time.time()
    # print("Create data in {:.4f} sec".format(end - start))
    #
    # p = Pool(7)
    # start = time.time()
    # results = p.map(product_train, job_args)
    # print(results)
    # end = time.time()
    # print("Multiprocessing in {:.4f} sec".format(end - start))


    # start = time.time()
    # result = 0
    # for param in job_args:
    #     r = product_train(param)
    #     print(t)
    #     result += r
    # print(result)
    # end = time.time()
    # print("Single processing in {:.4f} sec".format(end - start))

    model = Model('../models/100dim/',
                  '../context_distribution.pkl', n_context_sample=3000)

    start = time.time()
    snml_length = model.snml_length_sampling_multiprocess(8229, 9023, neg_size=3000)
    end = time.time()
    print("Multiprocessing in {:.4f} sec".format(end - start))
    print(snml_length)

    start = time.time()
    snml_length = model.snml_length_sampling(8229, 9023, neg_size=3000)
    end = time.time()
    print("Single process in {:.4f} sec".format(end - start))
    print(snml_length)



