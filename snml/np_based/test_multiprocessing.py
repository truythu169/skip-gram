from snml.np_based.model import Model
import time


if __name__ == "__main__":

    model = Model('../models/100dim/',
                  '../context_distribution.pkl')

    start = time.time()
    snml_length = model.snml_length_sampling_multiprocess(8229, 9023, neg_size=3000, n_context_sample=200)
    end = time.time()
    print("Multiprocessing in {:.4f} sec".format(end - start))
    print(snml_length)

    start = time.time()
    snml_length = model.snml_length_sampling(8229, 9023, neg_size=3000, n_context_sample=200)
    end = time.time()
    print("Single process in {:.4f} sec".format(end - start))
    print(snml_length)



