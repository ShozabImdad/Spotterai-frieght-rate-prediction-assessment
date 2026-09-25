# Freight Rate Prediction

Predicts the posted rate of a truckload from its lane, equipment, weight, date and market signals.

## Results

Expanding-window backtest on the development data (train on every load before a month, test on that month; August, September and October 2025):

| Model | MAE ($) | RMSE ($) | MAPE (%) | Median APE (%) |
|---|---:|---:|---:|---:|
| Rate-per-mile baseline | 181.41 | 270.13 | 8.13 | 6.82 |
| Log-linear regression with trend | 64.11 | 89.15 | 2.94 | 2.47 |
| LightGBM without trend | 82.27 | 108.87 | 3.67 | 3.36 |
| **LightGBM with trend (final)** | **50.82** | **76.28** | **2.18** | **1.77** |

Metrics above exclude the ~1.4% of labels identified as corrupted. Across all loads the final model scores MAE 104.64, MAPE 4.66%. Full per-fold numbers are in `artifacts/`.

## Approach

**Validation split.** `train_test.csv` covers January to October 2025 and the loads to predict cover November and December. A random split would leak future market conditions into training, so every evaluation is out-of-time: an expanding window that trains on all months before the test month.

**Data quality.**
- Negative weights (~0.6% of rows) fall in the same 5,000-47,500 lb range as positive weights once the sign is removed, and their rates show no bias, so they are treated as sign errors (absolute value).
- Missing weights are filled with the median for the same equipment type.
- Missing `market_index` values are filled with the mean for the same date; the date alone explains ~98% of its variance.
- About 1.4% of `posted_rate` labels are roughly 3-6x off from comparable loads in either direction. Each label is scored by a Huber-loss model trained on the other folds, and rows with a log residual above 0.3 are dropped from training. Clean residuals sit within about +/-0.12, so the threshold is not sensitive.
- Short lanes are floored at 70 miles, so their distance can exceed the straight-line distance several times over. This is consistent across files and left as is.
- Eight cities appear only in the prediction set. City identities are not used as features; latitude and longitude generalise to them.

**Time trend.** At the same `market_index`, rates rise by roughly 8% a year. Tree models cannot extrapolate this past the training window, which showed up as a consistent ~3.7% under-prediction on later months. The trend is estimated by a log-linear model, removed from the target before fitting LightGBM, and added back at prediction time.

**Model.** LightGBM regression on log rate, using distance, equipment, weight, market index, quote signal, coordinates and day of week. Month and day-of-year are excluded because the training data never sees November or December.

**December chart.** The chart inputs contain only lane, equipment, weight and date. Coordinates are looked up per city, `market_index` is the mean over all prediction-set loads on the same date, and `quote_signal` is the December median for Dry Van loads.

## Project layout

```
freight_rate/
  config.py       paths and constants
  data.py         loading and cleaning
  features.py     feature construction
  models.py       baseline, log-linear and LightGBM models, label screening
  evaluation.py   metrics and rolling backtest
  backtest.py     model comparison entry point
  predict.py      final training and prediction entry point
artifacts/        backtest results and training summary
scorer_results/   December chart produced by score.py
score.py          provided scorer
```

## Running

Requires Python 3.11. Place the provided CSV files in `data/`:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

Then:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python -m freight_rate.backtest  # model comparison, writes artifacts/
python -m freight_rate.predict   # writes validation_predictions.csv and fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

All runs are seeded and deterministic.
