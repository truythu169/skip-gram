from tf_based.skip_gram import SkipGram
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../../data/processed data/', type=str)
    parser.add_argument('--output_path', default='../output/', type=str)
    args = parser.parse_args()

    skip_gram = SkipGram(args.input_path, args.output_path, n_embedding=5)
    skip_gram.train(n_sampled=200, epochs=1, batch_size=100, print_step=10)
    skip_gram.export_embedding()
    skip_gram.export_model()
