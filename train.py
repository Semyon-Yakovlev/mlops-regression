import fire
import joblib
import numpy as np
import pandas as pd
import torch


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


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X).view(-1)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
    return np.mean(losses)


def train_model(batch_size, learning_rate, epochs):
    diamonds = DiamondsDataset("diamonds.csv")
    train_size = int(0.8 * len(diamonds))
    test_size = len(diamonds) - train_size

    train, test = torch.utils.data.random_split(diamonds, [train_size, test_size])
    joblib.dump(diamonds.X[test.indices], "X_test.h5")

    X_features = diamonds.X.size()[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(X_features, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)
    )
    data_train = torch.utils.data.DataLoader(train, batch_size=batch_size)
    loss = torch.nn.MSELoss()
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        losses.append(train_loop(data_train, model, loss, optimizer))
    joblib.dump(model, "model.h5")


if __name__ == "__main__":
    fire.Fire(train_model)