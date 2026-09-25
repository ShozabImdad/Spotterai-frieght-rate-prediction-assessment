from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from freight_rate import features
from freight_rate.config import LABEL_OUTLIER_THRESHOLD, SEED

LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "verbose": -1,
}


class RatePerMileBaseline:
    """Median rate per mile by equipment, scaled by distance."""

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "RatePerMileBaseline":
        rate_per_mile = target / frame["distance"]
        self.medians_ = rate_per_mile.groupby(frame["equipment"]).median()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (frame["equipment"].map(self.medians_) * frame["distance"]).to_numpy()


class LogLinearModel:
    """Multiplicative pricing model: linear regression on log rate with a time trend."""

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        design = pd.DataFrame(index=frame.index)
        design["years_elapsed"] = features.years_elapsed(frame["date"])
        design["log_distance"] = np.log(frame["distance"])
        design["log_distance_sq"] = design["log_distance"] ** 2
        design["log_market_index"] = np.log(frame["market_index"])
        design["quote_signal"] = frame["quote_signal"]
        design["weight"] = frame["weight"] / 10_000
        for equipment in features.EQUIPMENT[1:]:
            design[equipment] = (frame["equipment"] == equipment).astype(float)
        return design

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "LogLinearModel":
        self.model_ = LinearRegression().fit(self._design(frame), np.log(target))
        return self

    @property
    def annual_trend(self) -> float:
        return float(self.model_.coef_[0])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model_.predict(self._design(frame)))


@dataclass
class GradientBoostingModel:
    """LightGBM on log rate, optionally on top of a log-linear time trend.

    Rates drift upward over the year independently of market_index. Trees cannot
    extrapolate that drift past the training window, so the trend is estimated
    by the log-linear model, removed from the target, and added back at
    prediction time.
    """

    params: dict = field(default_factory=lambda: dict(LGBM_PARAMS))
    detrend: bool = True

    def _trend(self, frame: pd.DataFrame) -> np.ndarray:
        return self.annual_trend_ * features.years_elapsed(frame["date"]).to_numpy()

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "GradientBoostingModel":
        self.annual_trend_ = LogLinearModel().fit(frame, target).annual_trend if self.detrend else 0.0
        log_target = np.log(target.to_numpy()) - self._trend(frame)
        self.model_ = lgb.LGBMRegressor(**self.params).fit(features.build(frame), log_target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model_.predict(features.build(frame)) + self._trend(frame))


def valid_label_mask(frame: pd.DataFrame, target: pd.Series, n_splits: int = 5) -> np.ndarray:
    """Flag labels consistent with the rest of the data.

    Each row is scored by a Huber-loss model that never saw it; rows whose log
    residual exceeds the threshold are treated as corrupted labels.
    """
    X = features.build(frame)
    y = np.log(target.to_numpy())
    params = {**LGBM_PARAMS, "objective": "huber", "alpha": 0.1, "n_estimators": 600, "learning_rate": 0.05}
    predictions = np.empty(len(y))
    for train_idx, test_idx in KFold(n_splits, shuffle=True, random_state=SEED).split(X):
        model = lgb.LGBMRegressor(**params).fit(X.iloc[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X.iloc[test_idx])
    return np.abs(y - predictions) <= LABEL_OUTLIER_THRESHOLD
