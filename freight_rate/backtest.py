import pandas as pd

from freight_rate import data
from freight_rate.config import ARTIFACTS_DIR, BACKTEST_MONTHS, TARGET, TRAIN_PATH
from freight_rate.evaluation import rolling_backtest
from freight_rate.models import (
    GradientBoostingModel,
    LogLinearModel,
    RatePerMileBaseline,
    valid_label_mask,
)

CANDIDATES = {
    "rate_per_mile_baseline": RatePerMileBaseline,
    "log_linear": LogLinearModel,
    "lightgbm_no_trend": lambda: GradientBoostingModel(detrend=False),
    "lightgbm": GradientBoostingModel,
}


def main() -> None:
    raw = data.load(TRAIN_PATH)
    frame = data.clean(raw, data.weight_medians_by_equipment(raw))
    clean_reference = valid_label_mask(frame, frame[TARGET])
    print(f"Corrupted labels flagged in development data: {(~clean_reference).sum()} of {len(frame)}")

    results = rolling_backtest(frame, CANDIDATES, BACKTEST_MONTHS, clean_reference)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    results.to_csv(ARTIFACTS_DIR / "backtest_folds.csv", index=False)

    summary = results.groupby(["subset", "model"])[["MAE", "RMSE", "MAPE", "MedAPE"]].mean().round(2)
    summary.to_csv(ARTIFACTS_DIR / "backtest_summary.csv")
    with pd.option_context("display.width", 120):
        print(results.round(2).to_string(index=False))
        print(summary)


if __name__ == "__main__":
    main()
