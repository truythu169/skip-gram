from snml.tf_based.model import Model
import time
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/100dim/', type=str)
    parser.add_argument('--sample_path', default='../../data/processed data/split/', type=str)
    parser.add_argument('--snml_train_file', default='../../data/processed data/split/scope.csv', type=str)
    parser.add_argument('--output_path', default='models/100dim/output/', type=str)
    args = parser.parse_args()

    # read snml train file
    # data = np.genfromtxt(args.snml_train_file, delimiter=',').astype(int)
    # print(data)

    model = Model(args.model_path, args.sample_path, args.output_path, n_train_sample=10000)
    snml_length = model.snml_length(93, 1172, epochs=10)
    print(snml_length)

