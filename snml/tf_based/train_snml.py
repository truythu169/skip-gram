from snml.tf_based.model import Model

if __name__ == "__main__":
    model = Model('../models/100dim', '../../../data/processed data/split', 12802, 10000)
    p, loss = model.train(93, 1172, epochs=20)

    print(p)
    print(loss)

