import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import make_dataset


def prepare_combined_data(csv_path, config):
    """
    Fit preprocessing on combined train + validation data (2017-2021)
    and apply the fitted preprocessing to the 2022 test set.
    """
    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")

    combined_rows = [
        row for row in rows
        if int(row["year"]) in config.train_years
        or int(row["year"]) == config.validation_year
    ]

    test_rows = [
        row for row in rows
        if int(row["year"]) == config.test_year
    ]

    # Fit categorical encoders on combined 2017-2021 data.
    # ID 0 is reserved for categories unseen in the combined training data.
    category_maps = {}
    for name in config.categorical_features:
        values = sorted({row[name] for row in combined_rows})
        category_maps[name] = {
            value: index + 1
            for index, value in enumerate(values)
        }

    # Fit continuous-feature scaling on combined 2017-2021 data.
    combined_continuous = torch.tensor(
        [
            [float(row[name]) for name in config.continuous_features]
            for row in combined_rows
        ],
        dtype=torch.float32
    )

    means = combined_continuous.mean(dim=0)
    stds = combined_continuous.std(dim=0).clamp_min(1e-8)

    # Fit target scaling on combined 2017-2021 data.
    combined_target = torch.tensor(
        [float(row["resale_price"]) for row in combined_rows],
        dtype=torch.float32
    )

    target_mean = combined_target.mean().item()
    target_std = combined_target.std(correction=0).clamp_min(1e-8).item()

    return {
        "combined": make_dataset(
            combined_rows,
            category_maps,
            means,
            stds,
            config
        ),
        "test": make_dataset(
            test_rows,
            category_maps,
            means,
            stds,
            config
        ),
        "cardinalities": [
            len(category_maps[name]) + 1
            for name in config.categorical_features
        ],
        "continuous_mean": means,
        "continuous_std": stds,
        "target_mean": target_mean,
        "target_std": target_std
    }


@torch.no_grad()
def calculate_rmse(model, dataset, batch_size):
    model.eval()
    predictions = []
    targets = []

    for categorical, continuous, target in DataLoader(
        dataset,
        batch_size=batch_size
    ):
        predictions.append(model.predict_price(categorical, continuous))
        targets.append(target)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    return (predictions - targets).square().mean().sqrt().item()


@torch.no_grad()
def evaluate_regression(model, dataset, batch_size):
    model.eval()
    predictions = []
    targets = []

    for categorical, continuous, target in DataLoader(
        dataset,
        batch_size=batch_size
    ):
        predictions.append(model.predict_price(categorical, continuous))
        targets.append(target)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    rmse = (predictions - targets).square().mean().sqrt().item()

    ss_res = (targets - predictions).square().sum()
    ss_tot = (targets - targets.mean()).square().sum()
    r2 = (1 - ss_res / ss_tot).item()

    return rmse, r2
