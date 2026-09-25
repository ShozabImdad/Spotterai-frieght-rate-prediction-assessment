import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]

# Calendar features beyond day of week are excluded on purpose: training covers
# January to October only, so month or day-of-year splits cannot extrapolate to
# November and December. Seasonality reaches the model through market_index.
FEATURES = [
    "distance",
    "log_distance",
    "equipment",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "day_of_week",
]


TREND_ORIGIN = pd.Timestamp("2025-01-01")


def years_elapsed(dates: pd.Series) -> pd.Series:
    return (dates - TREND_ORIGIN).dt.days / 365.25


def build(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    features["distance"] = frame["distance"]
    features["log_distance"] = np.log(frame["distance"])
    features["equipment"] = pd.Categorical(frame["equipment"], categories=EQUIPMENT)
    for column in ("weight", "market_index", "quote_signal"):
        features[column] = frame[column]
    for column in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        features[column] = frame[column]
    features["day_of_week"] = frame["date"].dt.dayofweek
    return features[FEATURES]
