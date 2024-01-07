from subprocess import check_output

from dvc.api import DVCFileSystem
from joblib import dump
from mlflow import log_metric, log_param, set_tracking_uri, start_run
from numpy import mean
from omegaconf import DictConfig
from pandas import read_csv
from sklearn.metrics import mean_absolute_percentage_error
from toml import load
from torch import float32, from_numpy, nn, optim, utils
from torcheval.metrics import R2Score

from hydra import main
from models.model import model_spec


class DiamondsDataset(utils.data.Dataset):
    def __init__(self, path_file):
        self.df = read_csv(path_file, index_col=0)
        self.X = from_numpy(
            self.df.drop(columns=["price", "cut", "color", "clarity"]).values
        ).to(dtype=float32)
        self.y = from_numpy(self.df["price"].values).to(dtype=float32)

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@main(version_base=None, config_path="./hydra", config_name="config")
def train_model(cfg: DictConfig):
    config_path = "mlflow_config.toml"
    config = load(config_path)
    set_tracking_uri(f"http://{config['server']['host']}:{config['server']['port']}")
    git_commit_id = (
        check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    )
    with start_run():
        log_param("git_commit_id", git_commit_id)
        log_param("batch_size", cfg["params"].batch_size)
        log_param("learning_rate", cfg["params"].learning_rate)
        log_param("epochs", cfg["params"].epochs)
        fs = DVCFileSystem("https://github.com/Semyon-Yakovlev/MLOPS/")
        with fs.open("data/diamonds.csv") as file:
            diamonds = DiamondsDataset(file)
        train_size = int(0.8 * len(diamonds))
        test_size = len(diamonds) - train_size

        train, test = utils.data.random_split(diamonds, [train_size, test_size])
        data_train = utils.data.DataLoader(train, batch_size=cfg["params"].batch_size)
        dump(diamonds.X[test.indices], "data/X_test.h5")

        X_features = diamonds.X.size()[1]
        model = model_spec(X_features)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg["params"].learning_rate)
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
            log_metric("MAPE_train", mean(mape), step=i)
            log_metric("R2Score_train", mean(r2score), step=i)
            log_metric("MSELoss_train", mean(losses), step=i)
        dump(model, "models/model.h5")


if __name__ == "__main__":
    train_model()
