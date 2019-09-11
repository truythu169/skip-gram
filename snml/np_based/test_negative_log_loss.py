from snml.np_based.model import Model
import utils.tools as utils
from matplotlib import pyplot as plt
import time
import numpy as np


if __name__ == "__main__":
    # get samples from training data
    samples = []
    for i in range(100):
        samples.extend(utils.sample_learning_data('../../../data/processed data/split/', 12802, 100))

    model1 = Model('../models/50dim/', '../context_distribution.pkl', n_context_sample=6000)
    model2 = Model('../models/100dim/', '../context_distribution.pkl', n_context_sample=6000)
    model3 = Model('../models/150dim/', '../context_distribution.pkl', n_context_sample=6000)
    model4 = Model('../models/200dim/', '../context_distribution.pkl', n_context_sample=6000)
    models = [model1, model2, model3, model4]

    # get negative log of samples
    loss = [0., 0., 0., 0.]
    for sample in samples:
        for i in range(4):
            neg_log = -np.log(models[i].get_neg_prob(sample[0], sample[1]))
            loss[i] += neg_log

    print(loss)
