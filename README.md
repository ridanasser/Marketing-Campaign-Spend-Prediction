# Marketing Campaign Spend Prediction

## Project Overview
This project aims to predict the spending in marketing campaigns based on various features related to the campaigns.

## Dataset
The dataset consists of various features that influence marketing spend. It is collected from multiple sources and is available in CSV format.

## Target & Features
- **Target**: Total spend in marketing campaigns.
- **Features**: Demographics, previous spending behaviors, etc.

## Workflow
1. **Import**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
```

2. **Load/Explore**
```python
data = pd.read_csv('marketing_campaign.csv', sep='\t')
print(data.head())
```

3. **Feature Engineering TotalSpend**
```python
# Engineering the target variable

```

4. **Preprocessing/Scaling**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])
```

5. **Train/Test Split** with `shuffle=False`
```python
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, shuffle=False)
```

6. **Model Training**
   - **Linear Regression**
   - **Random Forest**
   - **SVR**

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

7. **Evaluation Metrics**
```python
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
bias = np.mean(y_pred - y_test)
```

8. **Visualizations**
```python
sns.scatterplot(x=y_test, y=y_pred)
```

9. **Scenario/ROI Section**
   - Discuss scenario analysis instead of true ROI.

## Results
| Model          | RMSE      | R²        | MAE       | MAPE      | Bias       |
|----------------|-----------|-----------|-----------|-----------|------------|
| Linear Reg.    | x.xxx     | x.xxx     | x.xxx     | x.xxx     | x.xxx      |
| Random Forest  | x.xxx     | x.xxx     | x.xxx     | x.xxx     | x.xxx      |
| SVR            | x.xxx     | x.xxx     | x.xxx     | x.xxx     | x.xxx      |

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook app.ipynb
   ```
3. Ensure `marketing_campaign.csv` is present in the working directory.

## Repository Structure
- **app.ipynb**: Main Jupyter notebook for analysis.
- **marketing_campaign.csv**: Dataset used for predictions.

## Notes & Limitations
- Engineered target variable.
- Using `shuffle=False` for splitting.
- ROI section is scenario-based and not a true metric.
- Data type warnings may occur during loading.

## Suggested Improvements
- Exploring more complex models.
- Implementing hyperparameter tuning.
- Incorporating more external datasets for a comprehensive analysis.

## License
This project is licensed under the MIT License.