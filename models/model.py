from torch import nn


def model_spec(X_features):
    model = nn.Sequential(
        nn.Linear(X_features, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    return model
