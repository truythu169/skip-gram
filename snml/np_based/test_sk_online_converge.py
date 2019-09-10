from snml.np_based.model import Model
import utils.tools as utils
from matplotlib import pyplot as plt
import time
import numpy as np


if __name__ == "__main__":
    model = Model('../models/150dim/', '../context_distribution.pkl', n_context_sample=600)
    p_sum = []
    start = time.time()
    print(model.get_prob(8229, 9023))
    p, losses = model.train_adam(8229, 9023)
    print(p)
    for i in range(1000):
        model.reset()
        p, losses = model.train_neg_adam(8229, 9023)
        p_sum.append(p)
    end = time.time()
    print("100 loop in {:.4f} sec".format(end - start))
    plt.hist(p_sum, bins=20)
    plt.show()
    # print('Mean: {} \nMin: {} \nMax: {} \nstd: {}'.format(np.mean(p_sum), min(p_sum), max(p_sum), np.std(p_sum)))
    #
    # model1 = Model('../../../output/convergence_test/20epochs/50dim/1/')
    # model2 = Model('../../../output/convergence_test/20epochs/50dim/2/')
    # model3 = Model('../../../output/convergence_test/20epochs/50dim/3/')
