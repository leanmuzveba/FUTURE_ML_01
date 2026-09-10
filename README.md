# SALES AND DEMAND FORECASTING -  FUTURE_ML_01

# Future Interns — Machine Learning Task 1: Sales & Demand Forecasting

## About This Project

I'm a second-year BSc IT student majoring in Data Science and Machine Learning, and this
is my submission for Future Interns' ML Track Task 1. The brief was to build a system
that forecasts future sales from historical business data and present it in a way that
would actually be useful to a business owner — not just accurate on paper.

## Dataset

I used the **Superstore Sales Dataset** (Kaggle), which contains ~9,800 individual order
records for a US retail business between January 2015 and December 2018, including order
date and sales value per line item.

## Approach

1. **Cleaning** — parsed the date column, checked for nulls/duplicates, confirmed no
   negative or invalid sales values.

2. **Aggregation** — the raw data is order-level, so I grouped it into a daily sales
   total, filled in days with zero orders (so the timeline has no gaps), then resampled
   to **weekly totals**, since daily sales were too noisy to forecast or present clearly.

3. **Feature engineering** — built calendar features (month, week of year), a sequential
   trend index (so the model can learn growth over time without memorizing specific
   years), and lag/rolling-average features (last week's sales, 4 weeks ago, and a
   4-week rolling mean) so the model has recent momentum to learn from.

4. **Modeling** — I tested three different approaches so I could compare a simple model,
   a more flexible one, and a classical time-series method, rather than assuming any one
   technique would work best:
   - Linear Regression
   - Random Forest Regression
   - ARIMA (a classical time-series model)

   Each was trained on the first 70% of weeks and evaluated on the most recent 30%
   (a time-based split, not a random one — random shuffling would let the model
   "peek" at the future during training, which isn't valid for forecasting).

## Results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Linear Regression | 4,936.05 | 6,557.30 | 35.91% |
| Random Forest | 5,160.83 | 6,753.54 | 37.24% |
| ARIMA(1,0,1) | 7,032.15 | 9,119.67 | 45.72% |

ARIMA performed worst — the weekly sales data didn't have enough autocorrelation
structure for it to learn from, so it mostly reverted to predicting the historical
average. Linear Regression and Random Forest performed similarly, with Linear
Regression slightly ahead on every metric.

## Which Model I Used, and Why

Interestingly, when I used each model to generate a genuine forward-looking forecast
(predicting 8 weeks past the end of the dataset, feeding each week's prediction into
the next), Random Forest's forecast flattened out to a nearly constant value — a known
limitation of tree-based models when forecasting several steps ahead. Linear Regression,
on the other hand, predicted a clear drop in early January before gradually recovering,
which matches a pattern visible earlier in the same historical data: sales consistently
dip right after the November/December peak each year.

Because of that, I used **Random Forest's results to report backtest accuracy** (it's
the strongest model at explaining sales I already have the answer for), but used
**Linear Regression's forecast for the actual forward-looking business chart**, since it
captures a seasonal pattern that matters more for planning than raw backtest accuracy.

## What the Forecast Means for the Business

This model looks at past weekly sales patterns — including recent trends and the time
of year — to estimate what sales are likely to look like in the coming weeks.

For the 8 weeks immediately following the dataset (early January through late
February), the model predicts a noticeable drop from December's peak of about $36,000/
week down to roughly $7,900–$8,500/week in early January, gradually recovering to
around $10,000/week by late February.

This drop isn't a red flag — the same post-holiday slowdown shows up every year in the
historical data. It reflects a real, recurring seasonal pattern rather than a model
error.

**For a store owner or manager**, this suggests: don't over-order inventory for January
based on December's strong numbers, plan staffing levels down accordingly for the
early-year slowdown, and don't be alarmed by lower January sales — it's expected, not a
sign of a declining business.

## Limitations

- The forecast is based on 4 years of historical data (2015–2018) and doesn't account
  for external factors like new promotions, competitor activity, or broader economic
  conditions.
- With only ~205 weeks of data, the models have a limited number of seasonal cycles to
  learn from, so seasonal patterns (like the January dip) are inferred from relatively
  few repeats.
- Random Forest's forward-looking forecast has a known weakness (it flattens out over
  multi-week horizons), which is why Linear Regression's forecast was used for the
  presented chart instead, despite Random Forest's stronger backtest score.

## Tools Used

Python, pandas, NumPy, scikit-learn, statsmodels, Matplotlib, Jupyter Notebook.

## Files in This Repo

- `notebooks/01_forecasting.ipynb` — full analysis: data cleaning, feature engineering,
  all three models, evaluation, and forecast
- `data/superstore_sales.csv` — source dataset
- `sales_forecasting_chart.png` — final business-facing forecast chart