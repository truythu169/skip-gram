from tf_based.skip_gram import SkipGram
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../../data/processed data/', type=str)
    parser.add_argument('--output_path', default='../output/', type=str)
    parser.add_argument('--n_embedding', default=50, type=int)
    parser.add_argument('--n_sampled', default=200, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=10000, type=int)
    parser.add_argument('--print_step', default=1000, type=int)
    args = parser.parse_args()

    input_path = '../data/processed data/'
    output_path = '../output/1/'

    dimension_list = [25, 75, 125, 175, 225, 275]
    # dimension_list = [50, 100, 150, 200, 250, 300]

    for dimension in dimension_list:
        skip_gram = SkipGram(input_path, output_path, n_embedding=dimension)
        skip_gram.train(n_sampled=200, epochs=20, batch_size=10000, print_step=1000)
        skip_gram.export_embedding()
