import json

import numpy as np
import pandas as pd

from freight_rate import data
from freight_rate.config import (
    ARTIFACTS_DIR,
    DECEMBER_PATH,
    PREDICTIONS_PATH,
    TARGET,
    TEMPLATE_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
)
from freight_rate.models import GradientBoostingModel, valid_label_mask

DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def enrich_december(december: pd.DataFrame, validation: pd.DataFrame, coordinates: pd.DataFrame) -> pd.DataFrame:
    """Derive the model inputs the chart file does not provide.

    Coordinates are fixed per city. market_index is a daily market level, taken
    as the mean over all validation loads on the same date. quote_signal has no
    daily structure, so the December median for the same equipment is used.
    """
    frame = december.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for side in ("pickup", "delivery"):
        frame[f"{side}_lat"] = frame[side].map(coordinates["lat"])
        frame[f"{side}_lon"] = frame[side].map(coordinates["lon"])

    daily_index = validation.groupby("date")["market_index"].mean()
    frame["market_index"] = frame["date"].map(daily_index)

    in_december = validation["date"].dt.month == 12
    quote_medians = validation[in_december].groupby("equipment")["quote_signal"].median()
    frame["quote_signal"] = frame["equipment"].map(quote_medians)

    if frame.isna().any().any():
        raise ValueError("December inputs could not be fully enriched")
    return frame


def main() -> None:
    raw_train = data.load(TRAIN_PATH)
    raw_validation = data.load(VALIDATION_PATH)
    weight_medians = data.weight_medians_by_equipment(raw_train)
    train = data.clean(raw_train, weight_medians)
    validation = data.clean(raw_validation, weight_medians)

    keep = valid_label_mask(train, train[TARGET])
    model = GradientBoostingModel().fit(train[keep], train.loc[keep, TARGET])

    template = pd.read_csv(TEMPLATE_PATH)
    predicted = pd.Series(model.predict(validation), index=validation["load_id"])
    template["predicted_rate"] = template["load_id"].map(predicted).round(2)
    if template["predicted_rate"].isna().any():
        raise ValueError("Some template load_ids have no prediction")
    template[["load_id", "predicted_rate"]].to_csv(PREDICTIONS_PATH, index=False)

    december = pd.read_csv(DECEMBER_PATH)
    coordinates = data.city_coordinates(raw_train, raw_validation)
    enriched = enrich_december(december.drop(columns="predicted_rate"), validation, coordinates)
    december["predicted_rate"] = np.round(model.predict(enriched), 2)
    december[DECEMBER_COLUMNS].to_csv(DECEMBER_PATH, index=False)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    summary = {
        "training_rows": int(len(train)),
        "corrupted_labels_removed": int((~keep).sum()),
        "annual_trend_log": round(model.annual_trend_, 4),
        "validation_prediction_stats": template["predicted_rate"].describe().round(2).to_dict(),
    }
    (ARTIFACTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {PREDICTIONS_PATH.name} and filled {DECEMBER_PATH.name}")


if __name__ == "__main__":
    main()
