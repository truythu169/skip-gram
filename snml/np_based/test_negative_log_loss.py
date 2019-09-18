from snml.np_based.model import Model
import utils.tools as utils
from matplotlib import pyplot as plt
import time
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # get samples from training data
    words = []
    contexts = []
    loop_no = 100
    for i in range(loop_no):
        ws, ctx = utils.sample_learning_data('../../../data/processed data/split/', 12802, 100)
        words.extend(ws)
        contexts.extend(ctx)

    model1 = Model('../models/50dim/', '../context_distribution.pkl', n_context_sample=6000)
    model2 = Model('../models/100dim/', '../context_distribution.pkl', n_context_sample=6000)
    model3 = Model('../models/150dim/', '../context_distribution.pkl', n_context_sample=6000)
    model4 = Model('../models/200dim/', '../context_distribution.pkl', n_context_sample=6000)
    models = [model1, model2, model3, model4]

    losses = [[], [], [], []]
    for i in range(len(words)):
        w = words[i]
        c = contexts[i]

        for j in range(4):
            neg_log = -np.log(models[j].get_prob(w, c))
            losses[j].append(neg_log)

    print(np.sum(losses[0]))
    print(np.sum(losses[1]))
    print(np.sum(losses[2]))
    print(np.sum(losses[3]))
