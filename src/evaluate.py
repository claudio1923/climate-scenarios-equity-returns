"""
Out-of-sample evaluation on the sealed test block (Split == "test", 2021-2024).

The thesis numbers are MATLAB numbers. scikit-learn will not reproduce them
exactly: the two implementations differ in split search, tie breaking and leaf
fitting. Both are reported side by side and never mixed.

Sanity check: if the replicated out-of-sample R2 falls outside 0.35-0.45 the
run stops instead of writing results.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from build_features import RESULTS
from train_gb import train

# Thesis values, from oos_model_comparison.csv / gb_final_metrics.csv.
THESIS_GB_R2_OOS = 0.406
THESIS_GB_RMSE_OOS = 5.108

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
            f"Sanity check failed: replicated out-of-sample R2 = {metrics['r2_oos']:.4f}, "
            f"outside the expected band [{low}, {high}]. Stopping instead of publishing."
        )
    return metrics, data


def comparison_table(metrics):
    """The replication-vs-thesis table shown in the README."""
    return pd.DataFrame(
        [
            {
                "Metric": "R2 out-of-sample",
                "Thesis (MATLAB)": THESIS_GB_R2_OOS,
                "Replication (Python)": round(metrics["r2_oos"], 4),
            },
            {
                "Metric": "RMSE out-of-sample",
                "Thesis (MATLAB)": THESIS_GB_RMSE_OOS,
                "Replication (Python)": round(metrics["rmse_oos"], 4),
            },
            {
                "Metric": "R2 in-sample",
                "Thesis (MATLAB)": 0.7065,
                "Replication (Python)": round(metrics["r2_is"], 4),
            },
            {
                "Metric": "RMSE in-sample",
                "Thesis (MATLAB)": 3.8182,
                "Replication (Python)": round(metrics["rmse_is"], 4),
            },
        ]
    )


def main():
    metrics, _ = evaluate()
    table = comparison_table(metrics)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "replication_vs_thesis_metrics.csv"
    table.to_csv(out, index=False)

    print("Sealed test block (2021-2024), Gradient Boosting.")
    print(table.to_string(index=False))
    print(f"\nWrote {out.relative_to(Path(__file__).resolve().parents[1])}")


if __name__ == "__main__":
    main()
