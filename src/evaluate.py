"""
Evaluation of the selected model on the two samples that matter: the 189 months
it is estimated on, and the sealed 2021-2024 block it is scored on once.

The test block takes no part in fitting or in selecting the configuration, so
the out-of-sample figures are an out-of-sample result rather than a selection
statistic.

Sanity check: if the out-of-sample R2 falls outside 0.35-0.45 the run stops
instead of writing results.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from build_features import RESULTS
from train_gb import train

R2_SANITY_BAND = (0.35, 0.45)


def r2(y_true, y_pred):
    """1 - SSE/SST, with SST taken around the mean of the evaluated sample."""
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - sse / sst


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(data=None):
    """Fit (if needed) and score in-sample and out-of-sample."""
    data = data or train()
    model = data["model"]

    in_sample = model.predict(data["x_fit"])
    out_sample = model.predict(data["x_test"])

    metrics = {
        "r2_is": r2(data["y_fit"], in_sample),
        "rmse_is": rmse(data["y_fit"], in_sample),
        "r2_oos": r2(data["y_test"], out_sample),
        "rmse_oos": rmse(data["y_test"], out_sample),
    }
    metrics["gap"] = metrics["r2_is"] - metrics["r2_oos"]

    low, high = R2_SANITY_BAND
    if not (low <= metrics["r2_oos"] <= high):
        raise ValueError(
            f"Sanity check failed: out-of-sample R2 = {metrics['r2_oos']:.4f}, "
            f"outside the expected band [{low}, {high}]. Stopping instead of publishing."
        )
    return metrics, data


def metrics_table(metrics):
    """One row per sample, as shown in the README."""
    return pd.DataFrame(
        [
            {
                "Sample": "in-sample (189 estimation months)",
                "R2": round(metrics["r2_is"], 4),
                "RMSE": round(metrics["rmse_is"], 4),
            },
            {
                "Sample": "out-of-sample (sealed 2021-2024 block)",
                "R2": round(metrics["r2_oos"], 4),
                "RMSE": round(metrics["rmse_oos"], 4),
            },
        ]
    )


def main():
    metrics, _ = evaluate()
    table = metrics_table(metrics)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "model_metrics.csv"
    table.to_csv(out, index=False)

    print("Gradient Boosting.")
    print(table.to_string(index=False))
    print(f"  gap in R2: {metrics['gap']:.4f}")
    print(f"\nWrote {out.relative_to(Path(__file__).resolve().parents[1])}")


if __name__ == "__main__":
    main()
