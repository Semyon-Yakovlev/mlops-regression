import fire
import joblib
from dvc.api import DVCFileSystem


def predict():
    fs = DVCFileSystem("https://github.com/Semyon-Yakovlev/MLOPS/")
    with fs.open("data/X_test.h5") as f:
        X_test = joblib.load(f)
    with fs.open("data/model.h5") as f:
        model = joblib.load(f)
    return model.forward(X_test).view(-1).detach().numpy()


if __name__ == "__main__":
    fire.Fire(predict)
