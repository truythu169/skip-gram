from snml.np_based.model import Model
import time
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snml_train_file', default='../../../data/processed data/scope.csv', type=str)
    parser.add_argument('--output_path', default='models/100dim/output/', type=str)
    parser.add_argument('--context_distribution_file', default='../context_distribution.pkl', type=str)
    args = parser.parse_args()

    # read snml train file
    data = np.genfromtxt(args.snml_train_file, delimiter=',').astype(int)

    # 50 dim
    model = Model('../models/50dim/', args.context_distribution_file, n_context_sample=600)
    total_length = 0
    start = time.time()
    for i in range(5):
        w = data[i][0]
        c = data[i][1]

        snml_length, prob_sum = model.snml_length_sampling(w, c)
        total_length += snml_length
    end = time.time()

    print('Length: {}, in {:.4f} sec'.format(total_length, (end - start)))

    total_length = 0
    start = time.time()
    for i in range(5):
        w = data[i][0]
        c = data[i][1]

        snml_length, prob_sum = model.snml_length_sampling(w, c)
        total_length += snml_length
    end = time.time()

    print('Length: {}, in {:.4f} sec'.format(total_length, (end - start)))

    # # 100 dim
    # model = Model('../models/100dim/', args.context_distribution_file, n_context_sample=600)
    # total_length = 0
    # for i in range(5):
    #     w = data[i][0]
    #     c = data[i][1]
    #
    #     snml_length, prob_sum = model.snml_length_sampling(w, c)
    #     total_length += snml_length
    # print(total_length)
    #
    # # 150 dim
    # model = Model('../models/150dim/', args.context_distribution_file, n_context_sample=600)
    # total_length = 0
    # for i in range(5):
    #     w = data[i][0]
    #     c = data[i][1]
    #
    #     snml_length, prob_sum = model.snml_length_sampling(w, c)
    #     total_length += snml_length
    # print(total_length)
    #
    # # 200 dim
    # model = Model('../models/200dim/', args.context_distribution_file, n_context_sample=600)
    # total_length = 0
    # for i in range(5):
    #     w = data[i][0]
    #     c = data[i][1]
    #
    #     snml_length, prob_sum = model.snml_length_sampling(w, c)
    #     total_length += snml_length
    # print(total_length)
