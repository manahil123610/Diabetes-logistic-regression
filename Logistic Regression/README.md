# Regression and Classification with scikit-learn

Predicting numeric targets with linear regression and a yes/no outcome with logistic regression, and judging each model on held-out test data.

## What it does

- **Simple linear regression:** predicts salary from years of experience (`Salary_dataset.csv`).
- **Multiple linear regression:** predicts house price from seven features (`house_price_regression_dataset.csv`): square footage, bedrooms, bathrooms, year built, lot size, garage size and neighbourhood quality.
- **Logistic regression:** classifies whether a patient has diabetes (`Outcome` 0 or 1) from six measurements in `diabetes.csv`.
- Splits each dataset 80/20 into training and test sets (`random_state=42`), fits `LinearRegression`, and reads the learned coefficients and intercept.
- Compares actual and predicted values on the held-out test set.
- Evaluates with MAE, MSE, RMSE and R².

## Simple linear regression results

Model: `Salary = 24,380.20 + 9,423.82 x YearsExperience`

| Metric | Value |
|---|---|
| MAE | 6,286.45 |
| MSE | 49,830,096.86 |
| RMSE | 7,059.04 |
| R² | 0.902 |

Each extra year of experience adds about 9,424 to the predicted salary, and the model explains about 90% of the variance in salary on the test set. With 30 rows in total (6 in the test set), the metrics are indicative rather than precise.

## Multiple linear regression results

Learned coefficients (price change per one-unit increase, other features held fixed):

| Feature | Coefficient |
|---|---|
| Square_Footage | 199.51 |
| Num_Bedrooms | 10,225.20 |
| Num_Bathrooms | 8,208.43 |
| Year_Built | 993.54 |
| Lot_Size | 14,885.38 |
| Garage_Size | 5,146.15 |
| Neighborhood_Quality | 115.07 |
| Intercept | -2,010,567.42 |

The large negative intercept is not a bug: `Year_Built` is around 2,000, so its term (about 993.5 x year) is roughly 2 million and the intercept offsets it. The coefficients are in different units, so their sizes cannot be compared directly as importance (a bedroom is a bigger step than a square foot). Features would need to be standardised for that.

Test-set metrics (MAE, RMSE, R²): see the evaluation cell at the end of `Multi Linear Regression.ipynb`.

## Logistic regression results (diabetes)

768 patients, no missing values, 500 without diabetes and 268 with. Features: Age, BMI, BloodPressure, Insulin, Glucose and DiabetesPedigreeFunction, with the same 80/20 split (154 test patients).

| Metric | Value |
|---|---|
| Accuracy | 0.760 |
| Precision (diabetes) | 0.661 |
| Recall (diabetes) | 0.673 |
| F1 (diabetes) | 0.667 |
| ROC AUC | 0.829 |

Confusion matrix on the test set:

| | Predicted no diabetes | Predicted diabetes |
|---|---|---|
| **Actual no diabetes** | 80 | 19 |
| **Actual diabetes** | 18 | 37 |

Always predicting "no diabetes" would score 64.3% accuracy on this test set, so 76.0% is a real but modest improvement. The model misses 18 of 55 diabetic patients, which is the error that matters most in a screening setting.

Learned coefficients (change in log-odds per one-unit increase):

| Feature | Coefficient |
|---|---|
| Age | 0.0456 |
| BMI | 0.0939 |
| BloodPressure | -0.0096 |
| Insulin | 0.0044 |
| Glucose | 0.0301 |
| DiabetesPedigreeFunction | 0.4852 |
| Intercept | -9.2359 |

Higher glucose, BMI, age and family-history score all raise the predicted probability of diabetes. Features are on different scales, so coefficient sizes are not a direct measure of importance.

## Files

| File | Contents |
|---|---|
| `Simple Linear Regression.ipynb` | Simple regression notebook with outputs |
| `Multi Linear Regression.ipynb` | Multiple regression notebook with outputs |
| `Diabetes_data_exploration_and_logistic_regression.ipynb` | Logistic regression notebook with outputs |
| `Salary_dataset.csv` | Years of experience and salary |
| `house_price_regression_dataset.csv` | House features and price |
| `diabetes.csv` | Patient measurements and diabetes outcome |
| `requirements.txt` | Python packages needed |

Keep each CSV in the same folder as the notebook that reads it, because the notebooks load them by file name.

## Run it

```
pip install -r requirements.txt
jupyter notebook
```

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.
