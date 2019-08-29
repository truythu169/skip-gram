from tf_based.skip_gram import SkipGram

if __name__ == "__main__":
    skip_gram = SkipGram('../data/processed data', n_embedding=25)
    skip_gram.train(n_sampled=200, epochs=10, batch_size=10000)
    skip_gram.export_embedding()
    skip_gram.export_model()
