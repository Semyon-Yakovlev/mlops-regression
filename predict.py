import fire
import joblib


def predict():
    X_test = joblib.load("data/X_test.h5")
    model = joblib.load("data/model.h5")
    return model.forward(X_test).view(-1).detach().numpy()


if __name__ == "__main__":
    fire.Fire(predict)
