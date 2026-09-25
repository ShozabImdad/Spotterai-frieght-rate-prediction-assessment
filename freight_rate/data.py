from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def clean(frame: pd.DataFrame, weight_medians: pd.Series) -> pd.DataFrame:
    """Repair feature-level quality issues shared by training and scoring data."""
    result = frame.copy()

    # Negative weights share the positive range and show no rate bias, so they are sign errors.
    result["weight"] = result["weight"].abs()
    result["weight"] = result["weight"].fillna(result["equipment"].map(weight_medians))

    # market_index is almost entirely a daily market level (R^2 ~0.98 on date alone).
    daily_index = result.groupby("date")["market_index"].transform("mean")
    result["market_index"] = result["market_index"].fillna(daily_index)
    return result


def weight_medians_by_equipment(frame: pd.DataFrame) -> pd.Series:
    return frame.assign(weight=frame["weight"].abs()).groupby("equipment")["weight"].median()


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            parts.append(
                frame[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
            )
    return pd.concat(parts).drop_duplicates("city").set_index("city")
