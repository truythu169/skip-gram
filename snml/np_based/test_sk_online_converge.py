from snml.np_based.model import Model
from sklearn.metrics import mean_absolute_error
import numpy as np


if __name__ == "__main__":
    # read snml train file
    data = np.genfromtxt('../../../data/processed data/scope.csv', delimiter=',').astype(int)

    # unique data
    str_data = set()
    for datum in data:
        str_data.add(str(datum[0]) + ',' + str(datum[1]))

    # full data
    model_full = Model('../../../output/convergence_test/3000samples/31epochs/full/50dim/',
                       '../context_distribution.pkl', n_context_sample=600)
    model_snml = Model('../../../output/convergence_test/3000samples/31epochs/snml/50dim/',
                       '../context_distribution.pkl', n_context_sample=600)

    p_full = []
    p_snml = []
    percent_error = 0
    n_sample = 100

    for i in range(n_sample):
        datum = str_data.pop()
        w, c = datum.split(',')
        w = int(w)
        c = int(c)

        ps = model_snml.train(w, c, epochs=31, neg_size=3000, update_weights=True)
        pf = model_full.get_neg_prob(w, c, neg_size=3000)

        p_full.append(pf)
        p_snml.append(ps)
        percent_error += abs(ps - pf) / pf

        if i % 100 == 0:
            print('{} th loop'.format(i))

    print('MAE: ', mean_absolute_error(p_snml, p_full))
    print('Mean percent error: ', (percent_error * 100 / n_sample))
