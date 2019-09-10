from snml.np_based.model import Model
import random
import time
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', default=50, type=int)
    parser.add_argument('--snml_train_file', default='../../../data/processed data/scope.csv', type=str)
    parser.add_argument('--output_path', default='models/100dim/output/', type=str)
    parser.add_argument('--context_distribution_file', default='../context_distribution.pkl', type=str)
    args = parser.parse_args()

    # read snml train file
    data = np.genfromtxt(args.snml_train_file, delimiter=',').astype(int)
    data = data[:100]

    # training
    model_path = '../models/{}dim/'.format(args.dim)
    model = Model(model_path, args.context_distribution_file, n_context_sample=600)
    leng1 = 0
    iteration = 0
    start = time.time()
    for datum in data:
        w = datum[0]
        c = datum[1]

        iteration += 1
        if iteration % 5 == 0:
            end = time.time()
            print("Iteration: {}, ".format(iteration),
                  "{:.4f} sec".format(end - start))
            start = time.time()

        snml_length, prob_sum = model.snml_length_sampling(w, c)
        leng1 += snml_length
    print(leng1)
