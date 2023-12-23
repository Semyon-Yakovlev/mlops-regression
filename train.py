import subprocess

import hydra
import joblib
import mlflow
import numpy as np
import pandas as pd
import toml
import torch
from dvc.api import DVCFileSystem
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_percentage_error
from torcheval.metrics import R2Score


class DiamondsDataset(torch.utils.data.Dataset):
    def __init__(self, path_file):
        self.df = pd.read_csv(path_file, index_col=0)
        self.X = torch.from_numpy(
            self.df.drop(columns=["price", "cut", "color", "clarity"]).values
        ).to(dtype=torch.float32)
        self.y = torch.from_numpy(self.df["price"].values).to(dtype=torch.float32)

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_model(cfg: DictConfig):
    config_path = "mlflow_config.toml"
    config = toml.load(config_path)
    mlflow.set_tracking_uri(
        f"http://{config['server']['host']}:{config['server']['port']}"
    )
    git_commit_id = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode("utf-8")
    )
    with mlflow.start_run():
        mlflow.log_param("git_commit_id", git_commit_id)
        mlflow.log_param("batch_size", cfg["params"].batch_size)
        mlflow.log_param("learning_rate", cfg["params"].learning_rate)
        mlflow.log_param("epochs", cfg["params"].epochs)
        fs = DVCFileSystem("https://github.com/Semyon-Yakovlev/MLOPS/")
        with fs.open("data/diamonds.csv") as f:
            diamonds = DiamondsDataset(f)
        train_size = int(0.8 * len(diamonds))
        test_size = len(diamonds) - train_size

        train, test = torch.utils.data.random_split(diamonds, [train_size, test_size])
        data_train = torch.utils.data.DataLoader(
            train, batch_size=cfg["params"].batch_size
        )
        joblib.dump(diamonds.X[test.indices], "data/X_test.h5")

        X_features = diamonds.X.size()[1]
        model = torch.nn.Sequential(
            torch.nn.Linear(X_features, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["params"].learning_rate)
        for i in range(cfg["params"].epochs):
            losses = []
            mape = []
            r2score = []
            for X, y in data_train:
                # Compute prediction and loss
                pred = model(X).view(-1)
                loss = loss_fn(pred, y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().numpy())
                mape.append(mean_absolute_percentage_error(pred.detach().numpy(), y))
                r2score.append(R2Score().update(pred, y).compute())
            mlflow.log_metric("MAPE_train", np.mean(mape), step=i)
            mlflow.log_metric("R2Score_train", np.mean(r2score), step=i)
            mlflow.log_metric("MSELoss_train", np.mean(losses), step=i)
        joblib.dump(model, "data/model.h5")


if __name__ == "__main__":
    train_model()
