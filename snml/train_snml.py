from snml.tf_based.model import Model
import time
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/150dim/', type=str)
    parser.add_argument('--sample_path', default='../../data/processed data/split/', type=str)
    parser.add_argument('--snml_train_file', default='../../data/processed data/scope.csv', type=str)
    parser.add_argument('--output_path', default='models/150dim/output/', type=str)
    parser.add_argument('--context_distribution_file', default='context_distribution.pkl', type=str)
    args = parser.parse_args()

    # read snml train file
    data = np.genfromtxt(args.snml_train_file, delimiter=',').astype(int)
    print(data)

    model = Model(args.model_path, args.sample_path, args.output_path, args.context_distribution_file,
                  n_train_sample=10000, n_context_sample=400)
    snml_length = model.snml_length(data[0][0], data[0][1], epochs=10)
    print(snml_length)
    snml_length = model.snml_length(data[0][0], data[0][1], epochs=10)
    print(snml_length)
    snml_length = model.snml_length(data[0][0], data[0][1], epochs=10)
    print(snml_length)

    # 50 dim
    # p_sum = 0
    # for i in range(100):
    #     p = model.train(data[0][0], data[0][1], epochs=20, update_weigh=False)
    #     p_sum += p
    #
    # print('Average 50 dim:', p_sum / 100)
    #
    # # 100 dim
    # model.change_model('models/100dim/')
    # p_sum = 0
    # for i in range(100):
    #     p = model.train(data[0][0], data[0][1], epochs=20, update_weigh=False)
    #     p_sum += p
    #
    # print('Average 100 dim:', p_sum / 100)
    #
    # # 150 dim
    # model.change_model('models/150dim/')
    # p_sum = 0
    # for i in range(100):
    #     p = model.train(data[0][0], data[0][1], epochs=20, update_weigh=False)
    #     p_sum += p
    #
    # print('Average 150 dim:', p_sum / 100)
    #
    # # 200 dim
    # model.change_model('models/200dim/')
    # p_sum = 0
    # for i in range(100):
    #     p = model.train(data[0][0], data[0][1], epochs=20, update_weigh=False)
    #     p_sum += p
    #
    # print('Average 200 dim:', p_sum / 100)
