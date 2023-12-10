import fire
import joblib


def predict():
    X_test = joblib.load("X_test.h5")
    model = joblib.load("model.h5")
    return model.forward(X_test).view(-1).detach().numpy()


if __name__ == "__main__":
    fire.Fire(predict)
