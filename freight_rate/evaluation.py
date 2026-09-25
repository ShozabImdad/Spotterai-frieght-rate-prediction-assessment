from typing import Callable

import numpy as np
import pandas as pd

from freight_rate.config import TARGET
from freight_rate.models import valid_label_mask


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(np.mean(np.abs(error) / actual) * 100),
        "MedAPE": float(np.median(np.abs(error) / actual) * 100),
    }


def rolling_backtest(
    frame: pd.DataFrame,
    candidates: dict[str, Callable[[], object]],
    months: list[str],
    clean_reference: np.ndarray,
) -> pd.DataFrame:
    """Expanding-window evaluation: train on all loads before a month, test on that month.

    Label screening runs inside each training window only. `clean_reference`
    marks labels screened on the full development set and is used purely to
    report accuracy on the rows that are not corrupted.
    """
    period = frame["date"].dt.to_period("M").astype(str)
    rows = []
    for month in months:
        train = frame[period < month]
        test = frame[period == month]
        keep = valid_label_mask(train, train[TARGET])
        clean_train = train[keep]
        test_clean = clean_reference[test.index]
        for name, make_model in candidates.items():
            model = make_model().fit(clean_train, clean_train[TARGET])
            predicted = model.predict(test)
            actual = test[TARGET].to_numpy()
            for subset, mask in (("all", np.ones(len(test), dtype=bool)), ("clean", test_clean)):
                rows.append(
                    {"month": month, "model": name, "subset": subset, "n": int(mask.sum()),
                     **metrics(actual[mask], predicted[mask])}
                )
    return pd.DataFrame(rows)
