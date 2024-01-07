from dvc.api import DVCFileSystem
from fire import Fire
from joblib import load
from pandas import DataFrame


def predict():
    fs = DVCFileSystem("https://github.com/Semyon-Yakovlev/MLOPS/")
    with fs.open("data/X_test.h5") as file:
        X_test = load(file)
    with fs.open("models/model.h5") as file:
        model = load(file)
    data = DataFrame(
        X_test.detach().numpy(), columns=["carat", "depth", "table", "x", "y", "z"]
    )
    data["predict"] = model.forward(X_test).view(-1).detach().numpy()
    return data.to_csv("data/predict_diamonds.csv")


if __name__ == "__main__":
    Fire(predict)
