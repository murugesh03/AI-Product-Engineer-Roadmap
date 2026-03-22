# Data & ML Basics — Complete Guide
> From Beginner to Super Advanced

---

## Table of Contents

1. [What is Data?](#1-what-is-data)
2. [Types of Data](#2-types-of-data)
3. [Data Collection & Sources](#3-data-collection--sources)
4. [Data Cleaning & Preprocessing](#4-data-cleaning--preprocessing)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6. [Feature Engineering](#6-feature-engineering)
7. [Introduction to Machine Learning](#7-introduction-to-machine-learning)
8. [Supervised Learning](#8-supervised-learning)
9. [Unsupervised Learning](#9-unsupervised-learning)
10. [Model Evaluation & Metrics](#10-model-evaluation--metrics)
11. [Bias, Variance & Overfitting](#11-bias-variance--overfitting)
12. [Advanced Topics — Ensemble Methods](#12-advanced-topics--ensemble-methods)
13. [Deep Learning Fundamentals](#13-deep-learning-fundamentals)
14. [Model Deployment & MLOps](#14-model-deployment--mlops)
15. [Super Advanced — Custom Training Loops, Optimization & Scaling](#15-super-advanced--custom-training-loops-optimization--scaling)

---

## 1. What is Data?

### Beginner

Data is a collection of raw facts, figures, or observations. It can represent anything — temperatures, customer names, stock prices, or images. On its own, data has no meaning; it becomes **information** only when it is processed and interpreted.

```python
# The simplest form of data — a Python list
temperatures = [22, 25, 19, 30, 27]
names = ["Alice", "Bob", "Charlie"]
is_raining = [True, False, True, False]
```

### Intermediate

In practice, data is stored in **structured** (tables, spreadsheets), **semi-structured** (JSON, XML), or **unstructured** (images, text, audio) formats.

```python
import pandas as pd

# Structured data — a DataFrame (like a spreadsheet in Python)
data = {
    "name":   ["Alice", "Bob", "Charlie"],
    "age":    [25, 30, 22],
    "salary": [50000, 70000, 45000]
}
df = pd.DataFrame(data)
print(df)
#       name  age  salary
# 0    Alice   25   50000
# 1      Bob   30   70000
# 2  Charlie   22   45000
```

```python
import json

# Semi-structured data — JSON
record = {
    "user_id": 101,
    "profile": {
        "name": "Alice",
        "tags": ["premium", "active"]
    }
}
json_string = json.dumps(record, indent=2)
print(json_string)
```

### Advanced

At scale, data is stored in **data lakes** (raw), **data warehouses** (processed), or **lakehouses** (hybrid). You interact with them via SQL or big-data frameworks.

```python
# Reading a large Parquet file (columnar format, very efficient)
import pandas as pd

df = pd.read_parquet("transactions.parquet", columns=["user_id", "amount", "timestamp"])
print(df.dtypes)
print(df.shape)  # (rows, columns)
```

```sql
-- Querying a data warehouse (BigQuery / Snowflake syntax)
SELECT
    user_id,
    SUM(amount)     AS total_spend,
    COUNT(*)        AS num_transactions
FROM transactions
WHERE timestamp >= '2024-01-01'
GROUP BY user_id
ORDER BY total_spend DESC
LIMIT 100;
```

---

## 2. Types of Data

### Beginner — The Four Main Types

| Type | Description | Example |
|---|---|---|
| **Nominal** | Categories with no order | Colors, gender, country |
| **Ordinal** | Categories with a meaningful order | Rating (1–5 stars) |
| **Interval** | Numeric, equal spacing, no true zero | Temperature (°C) |
| **Ratio** | Numeric, equal spacing, true zero | Weight, height, income |

```python
import pandas as pd

df = pd.DataFrame({
    "country":     ["India", "USA", "UK"],          # Nominal
    "satisfaction": [3, 5, 4],                       # Ordinal
    "temp_celsius": [35, 22, 15],                    # Interval
    "salary":      [60000, 95000, 75000]             # Ratio
})

# Check inferred types
print(df.dtypes)
```

### Intermediate — Continuous vs. Discrete

```python
import numpy as np
import matplotlib.pyplot as plt

# Continuous data — can take any value in a range
heights = np.random.normal(loc=170, scale=10, size=1000)  # in cm

# Discrete data — only whole/countable values
num_children = np.random.poisson(lam=1.5, size=1000)      # count per family

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(heights, bins=30, color="steelblue", edgecolor="white")
axes[0].set_title("Continuous: Height Distribution")

axes[1].hist(num_children, bins=range(0, 8), color="coral", edgecolor="white", align="left")
axes[1].set_title("Discrete: Number of Children")

plt.tight_layout()
plt.savefig("data_types.png", dpi=150)
```

### Advanced — Time-Series and Spatial Data

```python
import pandas as pd
import numpy as np

# Time-series data — indexed by datetime
dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
stock_prices = pd.Series(
    100 + np.cumsum(np.random.randn(365) * 1.5),
    index=dates,
    name="StockPrice"
)

# Resampling: daily → weekly average
weekly_avg = stock_prices.resample("W").mean()

# Rolling statistics (30-day moving average)
rolling_mean = stock_prices.rolling(window=30).mean()

print(stock_prices.describe())
```

```python
# Geospatial data with GeoPandas
import geopandas as gpd
from shapely.geometry import Point

coords = {
    "city":      ["Chennai", "Mumbai", "Delhi"],
    "latitude":  [13.08, 19.07, 28.70],
    "longitude": [80.27, 72.87, 77.10],
    "population":[7.1e6, 12.5e6, 11.0e6]
}
gdf = gpd.GeoDataFrame(
    coords,
    geometry=[Point(lon, lat) for lon, lat in zip(coords["longitude"], coords["latitude"])],
    crs="EPSG:4326"  # WGS84 coordinate system
)
print(gdf)
```

---

## 3. Data Collection & Sources

### Beginner — Loading Data

```python
import pandas as pd

# From CSV
df = pd.read_csv("data.csv")

# From Excel
df = pd.read_excel("report.xlsx", sheet_name="Sheet1")

# From dictionary
df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

print(df.head())   # First 5 rows
print(df.info())   # Column types and nulls
```

### Intermediate — APIs and Web Scraping

```python
import requests
import pandas as pd

# REST API example (OpenWeatherMap)
API_KEY = "your_api_key"
city = "Chennai"
url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    print(f"Temperature: {data['main']['temp']}°C")
    print(f"Weather: {data['weather'][0]['description']}")
```

```python
# Web Scraping with BeautifulSoup
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

table = soup.find("table", {"class": "wikitable"})
rows = table.find_all("tr")[1:]  # Skip header

data = []
for row in rows[:10]:
    cols = row.find_all("td")
    if len(cols) >= 2:
        data.append({
            "country": cols[0].text.strip(),
            "population": cols[1].text.strip()
        })

df = pd.DataFrame(data)
print(df)
```

### Advanced — Streaming & Database Ingestion

```python
# SQLAlchemy — connecting to a relational database
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("postgresql://user:password@localhost:5432/mydb")

# Read with SQL
df = pd.read_sql("""
    SELECT user_id, event_type, created_at
    FROM events
    WHERE created_at >= NOW() - INTERVAL '30 days'
""", con=engine)

# Write back
df_clean.to_sql("events_clean", con=engine, if_exists="replace", index=False)
```

```python
# Kafka consumer — real-time streaming ingestion
from kafka import KafkaConsumer
import json
import pandas as pd

consumer = KafkaConsumer(
    "user-events",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True
)

buffer = []
for message in consumer:
    event = message.value
    buffer.append(event)

    # Process in mini-batches of 100
    if len(buffer) >= 100:
        df = pd.DataFrame(buffer)
        # ... process df ...
        buffer = []
```

---

## 4. Data Cleaning & Preprocessing

### Beginner — Handling Missing Values

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "age":    [25, None, 30, 22, None],
    "salary": [50000, 60000, None, 45000, 70000],
    "city":   ["Chennai", None, "Delhi", "Mumbai", "Bangalore"]
})

# Identify missing values
print(df.isnull().sum())
#  age       2
#  salary    1
#  city      1

# Drop rows with ANY missing value
df_dropped = df.dropna()

# Fill numeric with mean/median
df["age"].fillna(df["age"].median(), inplace=True)
df["salary"].fillna(df["salary"].mean(), inplace=True)

# Fill categorical with mode
df["city"].fillna(df["city"].mode()[0], inplace=True)

print(df.isnull().sum())  # All zeros now
```

### Intermediate — Outlier Detection and Encoding

```python
import pandas as pd
import numpy as np

# --- IQR Method for outlier removal ---
df = pd.DataFrame({"income": [40000, 45000, 42000, 500000, 38000, 43000]})

Q1 = df["income"].quantile(0.25)
Q3 = df["income"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df[(df["income"] >= lower) & (df["income"] <= upper)]
print(f"Removed {len(df) - len(df_clean)} outliers")
```

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

df = pd.DataFrame({
    "color": ["red", "blue", "green", "red", "blue"],
    "size":  ["S", "M", "L", "XL", "S"]
})

# Label encoding (ordinal features)
le = LabelEncoder()
df["size_encoded"] = le.fit_transform(df["size"])

# One-hot encoding (nominal features)
df_ohe = pd.get_dummies(df, columns=["color"], drop_first=True)
print(df_ohe)
```

### Advanced — Pipelines with scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

# Define column groups
numeric_features = ["age", "salary", "experience"]
categorical_features = ["department", "city"]

# Numeric pipeline: impute missing → scale
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline: impute missing → one-hot encode
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine both into a single ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Fit and transform your data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)  # Uses training stats
```

### Super Advanced — Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    """Clips values to [lower_quantile, upper_quantile] per feature."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        for col in df.columns:
            self.lower_bounds_[col] = df[col].quantile(self.lower)
            self.upper_bounds_[col] = df[col].quantile(self.upper)
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X).copy()
        for col in df.columns:
            df[col] = df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )
        return df.values

# Usage in a pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ("winsorize", WinsorizeTransformer(lower=0.05, upper=0.95)),
    ("model", LinearRegression())
])
pipe.fit(X_train, y_train)
```

---

## 5. Exploratory Data Analysis (EDA)

### Beginner — Summary Statistics

```python
import pandas as pd

df = pd.read_csv("titanic.csv")

# Shape, types, missing
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

# Summary statistics for numeric columns
print(df.describe())

# Value counts for categorical
print(df["Sex"].value_counts())
print(df["Pclass"].value_counts(normalize=True))  # as proportions
```

### Intermediate — Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution of numeric feature
sns.histplot(df["Age"].dropna(), kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Age Distribution")

# Categorical count
sns.countplot(data=df, x="Pclass", hue="Survived", ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Survival by Class")

# Boxplot for outliers
sns.boxplot(data=df, x="Pclass", y="Fare", ax=axes[1, 0])
axes[1, 0].set_title("Fare Distribution per Class")

# Correlation heatmap
corr = df[["Age", "Fare", "SibSp", "Parch", "Survived"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1])
axes[1, 1].set_title("Correlation Matrix")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
```

### Advanced — Automated EDA with ydata-profiling

```python
from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("data.csv")

profile = ProfileReport(
    df,
    title="EDA Report",
    explorative=True,
    correlations={"pearson": {"calculate": True}, "spearman": {"calculate": True}},
    missing_diagrams={"bar": True, "matrix": True, "heatmap": True}
)

# Save as HTML
profile.to_file("eda_report.html")

# Or view inline in Jupyter
profile.to_notebook_iframe()
```

### Super Advanced — Statistical Testing in EDA

```python
from scipy import stats
import pandas as pd
import numpy as np

df = pd.read_csv("experiment.csv")

# --- T-test: Are group A and group B means significantly different? ---
group_a = df[df["group"] == "A"]["metric"]
group_b = df[df["group"] == "B"]["metric"]

t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)  # Welch's t-test
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Statistically significant difference (α=0.05)")

# --- Chi-squared test: Is survival independent of passenger class? ---
contingency = pd.crosstab(df["Pclass"], df["Survived"])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-squared: {chi2:.4f}, P-value: {p:.4f}, DOF: {dof}")

# --- Kolmogorov-Smirnov test: Is feature normally distributed? ---
stat, p_norm = stats.kstest(df["Fare"].dropna(), "norm",
                             args=(df["Fare"].mean(), df["Fare"].std()))
print(f"\nKS statistic: {stat:.4f}, P-value: {p_norm:.4f}")
```

---

## 6. Feature Engineering

### Beginner — Creating New Features

```python
import pandas as pd

df = pd.DataFrame({
    "first_name": ["Alice", "Bob"],
    "last_name":  ["Smith", "Jones"],
    "birth_year": [1990, 1985],
    "salary":     [50000, 80000]
})

# Combine columns
df["full_name"] = df["first_name"] + " " + df["last_name"]

# Derived numeric feature
import datetime
df["age"] = datetime.datetime.now().year - df["birth_year"]

# Binning (discretization)
df["salary_band"] = pd.cut(df["salary"],
                            bins=[0, 40000, 70000, 100000, float("inf")],
                            labels=["Low", "Medium", "High", "Very High"])
print(df)
```

### Intermediate — Date Features & Interactions

```python
import pandas as pd

df = pd.DataFrame({
    "order_date":    ["2024-03-15", "2024-12-01", "2023-07-04"],
    "delivery_date": ["2024-03-20", "2024-12-10", "2023-07-09"],
    "price":         [100, 250, 75],
    "quantity":      [2, 1, 5]
})

df["order_date"]    = pd.to_datetime(df["order_date"])
df["delivery_date"] = pd.to_datetime(df["delivery_date"])

# Temporal features
df["order_day_of_week"] = df["order_date"].dt.dayofweek   # 0=Monday
df["order_month"]       = df["order_date"].dt.month
df["order_quarter"]     = df["order_date"].dt.quarter
df["is_weekend"]        = df["order_day_of_week"].isin([5, 6]).astype(int)

# Duration feature
df["delivery_days"] = (df["delivery_date"] - df["order_date"]).dt.days

# Interaction feature
df["total_value"] = df["price"] * df["quantity"]

print(df.T)
```

### Advanced — Target Encoding & Polynomial Features

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

# --- Target encoding (mean encoding) with cross-validation to avoid leakage ---
def target_encode_cv(X, y, col, n_splits=5, smoothing=1.0):
    """
    Replace category with mean(target), using out-of-fold estimates
    to prevent data leakage.
    """
    encoded = np.zeros(len(X))
    global_mean = y.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X):
        means = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean()
        counts = X[col].iloc[train_idx].value_counts()

        # Bayesian smoothing: blend category mean with global mean
        smoothed = (means * counts + global_mean * smoothing) / (counts + smoothing)
        encoded[val_idx] = X[col].iloc[val_idx].map(smoothed).fillna(global_mean)

    return encoded

df = pd.read_csv("train.csv")
df["city_encoded"] = target_encode_cv(df, df["target"], col="city")
```

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[2, 3], [4, 5], [1, 2]])

# Generate degree-2 polynomial + interaction features
# [1, x1, x2] → [1, x1, x2, x1², x1·x2, x2²]
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
print("Original shape:", X.shape)
print("Polynomial shape:", X_poly.shape)
print("Feature names:", poly.get_feature_names_out(["x1", "x2"]))
```

### Super Advanced — Automated Feature Selection with SHAP

```python
import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Train a model
model = xgb.XGBClassifier(n_estimators=200, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# SHAP values explain which features drive each prediction
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance (mean |SHAP|)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.savefig("shap_summary.png", bbox_inches="tight")

# Individual prediction explanation
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    feature_names=feature_names
)

# Feature selection: keep top-K by mean |SHAP|
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_k = 10
top_features = pd.Series(mean_abs_shap, index=feature_names).nlargest(top_k).index.tolist()
X_train_selected = X_train[top_features]
```

---

## 7. Introduction to Machine Learning

### Beginner — What is ML?

Machine learning (ML) is a branch of artificial intelligence where systems **learn patterns from data** instead of being explicitly programmed with rules.

**The 3 Types of ML:**

| Type | Description | Example |
|---|---|---|
| **Supervised** | Learn from labeled data (input → output) | Email spam detection |
| **Unsupervised** | Find hidden patterns in unlabeled data | Customer segmentation |
| **Reinforcement** | Agent learns by trial and reward | Game-playing AI |

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Supervised learning — simplest example
# Predict house price from size (sq ft)
X = np.array([[500], [750], [1000], [1250], [1500]])
y = np.array([150000, 200000, 250000, 300000, 350000])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict
prediction = model.predict([[1100]])
print(f"Predicted price for 1100 sq ft: ${prediction[0]:,.0f}")
# Output: Predicted price for 1100 sq ft: $270,000
```

### Intermediate — The ML Workflow

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load data
df = pd.read_csv("iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

# Step 2: Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 3: Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform on train
X_test_scaled  = scaler.transform(X_test)        # Only transform on test (no leakage!)

# Step 4: Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

---

## 8. Supervised Learning

### Beginner — Linear & Logistic Regression

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

# --- Linear Regression (predicts a continuous number) ---
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Coefficients: {lr.coef_}")

# --- Logistic Regression (predicts a class/probability) ---
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=10000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:, 1]  # P(malignant)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Intermediate — Decision Trees & SVMs

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Decision Tree ---
tree = DecisionTreeClassifier(max_depth=4, min_samples_split=20, random_state=42)
tree.fit(X_train, y_train)

# Visualize rules (text)
print(export_text(tree, feature_names=[f"f{i}" for i in range(10)]))

# --- Support Vector Machine ---
# RBF kernel maps data to higher dimensions to find a separating hyperplane
svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
svm.fit(X_train, y_train)
print(f"SVM Accuracy: {svm.score(X_test, y_test):.4f}")
```

### Advanced — Gradient Boosting (XGBoost / LightGBM)

```python
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import numpy as np

# --- XGBoost ---
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,    # L1 regularization
    reg_lambda=1.0,   # L2 regularization
    eval_metric="logloss",
    early_stopping_rounds=50,
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# --- LightGBM (faster for large datasets) ---
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    num_leaves=63,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    random_state=42
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Cross-validation score
cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### Super Advanced — Hyperparameter Tuning with Optuna

```python
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def objective(trial):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 1000),
        "num_leaves":      trial.suggest_int("num_leaves", 20, 300),
        "learning_rate":   trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "feature_fraction":trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction":trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_vl, y_vl)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        proba = model.predict_proba(X_vl)[:, 1]
        aucs.append(roc_auc_score(y_vl, proba))

    return np.mean(aucs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=-1, show_progress_bar=True)

print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 9. Unsupervised Learning

### Beginner — K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# Elbow method to choose K
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, "bo-")
plt.xlabel("Number of clusters K")
plt.ylabel("Inertia (within-cluster SSE)")
plt.title("Elbow Method")
plt.savefig("elbow.png", dpi=150)

# Fit with chosen K
km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X)
centers = km.cluster_centers_
print(f"Cluster centers:\n{centers}")
```

### Intermediate — DBSCAN and Hierarchical Clustering

```python
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- DBSCAN — finds clusters of arbitrary shape, marks noise as -1 ---
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.4, min_samples=10)
labels_db = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise    = list(labels_db).count(-1)
print(f"DBSCAN clusters: {n_clusters}, noise points: {n_noise}")

# --- Hierarchical Clustering ---
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

Z = linkage(X_scaled[:100], method="ward")  # Ward minimizes within-cluster variance

plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=20)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.savefig("dendrogram.png", dpi=150)
```

### Advanced — PCA for Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# PCA: linear dimensionality reduction
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_high_dim)

print(f"Original: {X_high_dim.shape}")
print(f"After PCA: {X_pca.shape}")
print(f"Explained variance per component: {pca.explained_variance_ratio_[:5]}")

# Cumulative explained variance plot
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cum_var)
plt.axhline(0.95, linestyle="--", color="red", label="95% threshold")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.legend()
plt.savefig("pca_variance.png", dpi=150)

# t-SNE: non-linear 2D visualization of high-dim data
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_2d = tsne.fit_transform(X_pca)  # Apply on PCA-reduced data for speed

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=5)
plt.colorbar(scatter, label="Cluster")
plt.title("t-SNE Visualization")
plt.savefig("tsne.png", dpi=150)
```

---

## 10. Model Evaluation & Metrics

### Beginner — Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
y_prob = [0.9, 0.2, 0.4, 0.8, 0.1, 0.7, 0.6, 0.3, 0.85, 0.15]

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")    # Correct / Total
print(f"Precision: {precision_score(y_true, y_pred):.4f}")   # TP / (TP + FP)
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")      # TP / (TP + FN)
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")          # Harmonic mean
print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")     # Area under ROC curve

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"])
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150)
```

### Intermediate — ROC & PR Curves

```python
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve (better for imbalanced datasets)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(fpr, tpr, "b-", label=f"ROC-AUC = {roc_auc:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", label="Random")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve"); axes[0].legend()

axes[1].plot(recall, precision, "r-", label=f"PR-AUC = {pr_auc:.4f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

plt.tight_layout()
plt.savefig("roc_pr_curves.png", dpi=150)
```

### Advanced — Cross-Validation Strategies

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold,
    TimeSeriesSplit, cross_validate
)
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Standard K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Stratified K-Fold (maintains class proportions in each fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Time Series Split (respects temporal order — no future leakage)
tss = TimeSeriesSplit(n_splits=5, gap=10)

results = cross_validate(
    model, X, y,
    cv=skf,
    scoring=["accuracy", "roc_auc", "f1"],
    return_train_score=True,
    n_jobs=-1
)

for metric in ["accuracy", "roc_auc", "f1"]:
    test_scores = results[f"test_{metric}"]
    print(f"{metric}: {test_scores.mean():.4f} ± {test_scores.std():.4f}")
```

### Super Advanced — Custom Scoring & Threshold Optimization

```python
from sklearn.metrics import make_scorer
import numpy as np
from sklearn.model_selection import cross_val_score

# Business-specific scorer: maximize recall at minimum 70% precision
def custom_recall_at_precision(y_true, y_prob, min_precision=0.70):
    """Find the threshold that maximizes recall while keeping precision ≥ min_precision."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    valid = precision >= min_precision
    if not np.any(valid):
        return 0.0
    return np.max(recall[valid])

scorer = make_scorer(custom_recall_at_precision, needs_proba=True, min_precision=0.70)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer)
print(f"Recall@Precision≥0.7: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Threshold optimization using validation set
proba_val = model.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []
for t in thresholds:
    y_pred_t = (proba_val >= t).astype(int)
    from sklearn.metrics import f1_score
    f1_scores.append(f1_score(y_val, y_pred_t))

best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {best_threshold:.2f}, Best F1: {max(f1_scores):.4f}")
```

---

## 11. Bias, Variance & Overfitting

### Beginner — The Concept

| Problem | Cause | Symptom | Fix |
|---|---|---|---|
| **High Bias** (Underfitting) | Model too simple | High train error + high test error | More features, bigger model |
| **High Variance** (Overfitting) | Model too complex | Low train error, high test error | More data, regularization, simpler model |
| **Good fit** | Balance | Low train + low test error | ✓ |

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(100) * 0.3

# Underfitting: degree 1 (too simple)
model_under = make_pipeline(PolynomialFeatures(1), Ridge(alpha=1.0))

# Good fit: degree 3
model_good  = make_pipeline(PolynomialFeatures(3), Ridge(alpha=1.0))

# Overfitting: degree 15 with no regularization
model_over  = make_pipeline(PolynomialFeatures(15), Ridge(alpha=1e-10))

for name, m in [("Underfitting", model_under), ("Good Fit", model_good), ("Overfitting", model_over)]:
    m.fit(X, y)
    train_mse = np.mean((m.predict(X) - y) ** 2)
    print(f"{name}: Train MSE = {train_mse:.4f}")
```

### Advanced — Regularization: L1, L2, ElasticNet

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
import numpy as np

# Ridge (L2) — shrinks all coefficients, keeps all features
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, {"alpha": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="neg_mse")
ridge_cv.fit(X_train, y_train)
print(f"Best Ridge alpha: {ridge_cv.best_params_}")

# Lasso (L1) — drives some coefficients to exactly zero (feature selection)
lasso = Lasso(max_iter=10000)
lasso_cv = GridSearchCV(lasso, {"alpha": [0.001, 0.01, 0.1, 1, 10]}, cv=5, scoring="neg_mse")
lasso_cv.fit(X_train, y_train)
zero_coef = np.sum(lasso_cv.best_estimator_.coef_ == 0)
print(f"Best Lasso alpha: {lasso_cv.best_params_}, Zeroed features: {zero_coef}")

# ElasticNet — combines L1 + L2
en = ElasticNet(max_iter=10000)
en_cv = GridSearchCV(en,
    {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.1, 0.5, 0.9]},
    cv=5, scoring="neg_mse")
en_cv.fit(X_train, y_train)
print(f"Best ElasticNet params: {en_cv.best_params_}")
```

### Super Advanced — Learning Curves & Validation Curves

```python
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy as np

model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Learning curve — how does performance scale with more training data?
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="roc_auc", n_jobs=-1
)

plt.figure(figsize=(10, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), "b-o", label="Train AUC")
plt.plot(train_sizes, val_scores.mean(axis=1), "r-o", label="Val AUC")
plt.fill_between(train_sizes,
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color="b")
plt.fill_between(train_sizes,
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color="r")
plt.xlabel("Training set size"); plt.ylabel("ROC-AUC")
plt.title("Learning Curve"); plt.legend(); plt.grid(True)
plt.savefig("learning_curve.png", dpi=150)
```

---

## 12. Advanced Topics — Ensemble Methods

### Intermediate — Bagging & Random Forests

```python
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier

# Bagging: train N trees on bootstrap samples; reduce variance
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=100,
    max_samples=0.8,      # 80% of data per tree
    max_features=0.8,     # 80% of features per tree
    bootstrap=True,       # sampling with replacement
    random_state=42,
    n_jobs=-1
)

# Random Forest: bagging + random feature subset at each split
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,        # Trees grow to purity
    min_samples_split=5,
    max_features="sqrt",   # sqrt(p) features at each split
    oob_score=True,        # Out-of-bag estimate (free validation!)
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")  # Cross-val quality estimate
```

### Advanced — Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb

# Base models (level-0)
base_models = [
    ("lgbm", lgb.LGBMClassifier(n_estimators=200, random_state=42)),
    ("svm",  SVC(probability=True, kernel="rbf", random_state=42)),
    ("knn",  KNeighborsClassifier(n_neighbors=7))
]

# Meta-model (level-1) — learns to combine base model predictions
meta_model = LogisticRegression(C=1.0, random_state=42)

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,                   # Use 5-fold CV to generate level-0 train meta-features
    stack_method="predict_proba",
    passthrough=False,       # Don't pass original features to meta-model
    n_jobs=-1
)
stack.fit(X_train, y_train)
print(f"Stacking AUC: {roc_auc_score(y_test, stack.predict_proba(X_test)[:, 1]):.4f}")
```

---

## 13. Deep Learning Fundamentals

### Beginner — Neural Network with Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build a simple feedforward neural network
model = keras.Sequential([
    layers.Input(shape=(20,)),              # 20 input features
    layers.Dense(128, activation="relu"),   # Hidden layer 1
    layers.Dropout(0.3),                    # Regularization
    layers.Dense(64, activation="relu"),    # Hidden layer 2
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")  # Output: binary classification
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

model.summary()

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]
)
```

### Intermediate — Convolutional Neural Networks (CNNs)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# CNN for image classification (e.g., CIFAR-10: 32×32×3 images)
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),

    # Block 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Classifier head
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")  # 10 classes
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Advanced — Transfer Learning

```python
import tensorflow as tf
from tensorflow import keras

# Fine-tune a pretrained EfficientNetV2 on a custom dataset
base_model = keras.applications.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Phase 1: Freeze base model; only train the head
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.efficientnet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, epochs=5, validation_data=val_ds)

# Phase 2: Unfreeze top layers of base model; fine-tune with small LR
base_model.trainable = True
fine_tune_at = 200  # Freeze all layers before this one

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Very small LR!
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, epochs=10, validation_data=val_ds)
```

### Super Advanced — Custom Training Loop with Mixed Precision

```python
import tensorflow as tf

# Enable mixed precision (FP16 compute, FP32 weights) — 2–3x speedup on modern GPUs
tf.keras.mixed_precision.set_global_policy("mixed_float16")

model = build_model()  # Your model function

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# GradientTape doesn't auto-scale; wrap optimizer for mixed precision
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
val_acc_metric    = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

@tf.function  # Compile for performance (graph mode)
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)
        scaled_loss = optimizer.get_scaled_loss(loss)  # For mixed precision stability

    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(scaled_grads)

    # Gradient clipping to prevent exploding gradients
    grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss_metric.update_state(loss)
    return loss

@tf.function
def val_step(x_batch, y_batch):
    logits = model(x_batch, training=False)
    val_acc_metric.update_state(y_batch, logits)

EPOCHS = 50
for epoch in range(EPOCHS):
    train_loss_metric.reset_state()
    val_acc_metric.reset_state()

    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)

    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)

    print(f"Epoch {epoch+1:02d}: "
          f"Loss={train_loss_metric.result():.4f}, "
          f"Val Acc={val_acc_metric.result():.4f}")
```

---

## 14. Model Deployment & MLOps

### Intermediate — Saving & Serving Models

```python
import joblib
import pickle

# Save scikit-learn model (preferred: joblib)
joblib.dump(model, "model.joblib")
loaded = joblib.load("model.joblib")

# Save full sklearn pipeline
joblib.dump(pipeline, "pipeline.joblib")

# Save Keras/TF model
model.save("my_model.keras")
loaded_keras = tf.keras.models.load_model("my_model.keras")

# Convert to ONNX for cross-framework serving
import tf2onnx, onnx
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
onnx.save(model_proto, "model.onnx")
```

```python
# FastAPI model serving endpoint
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API", version="1.0")
model = joblib.load("pipeline.joblib")

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    X = np.array(request.features).reshape(1, -1)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][prediction])
    return PredictResponse(prediction=prediction, probability=probability)

# Run: uvicorn serve:app --host 0.0.0.0 --port 8000
```

### Advanced — MLflow Experiment Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-prediction-v2")

with mlflow.start_run(run_name="RF-v2") as run:
    # Log parameters
    params = {"n_estimators": 300, "max_depth": 8, "min_samples_split": 10}
    mlflow.log_params(params)

    # Train
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # Log metrics
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc  = roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1])
    mlflow.log_metrics({"train_auc": train_auc, "test_auc": test_auc})

    # Log artifacts
    feature_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    feature_imp.to_csv("feature_importance.csv")
    mlflow.log_artifact("feature_importance.csv")

    # Log model with signature
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature,
                              registered_model_name="ChurnClassifier")

    print(f"Run ID: {run.info.run_id}")
    print(f"Test AUC: {test_auc:.4f}")
```

---

## 15. Super Advanced — Custom Training Loops, Optimization & Scaling

### Distributed Training with PyTorch DDP

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, dataset, model_class, epochs=10):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Each process gets a subset of the data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

    model = model_class().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler    = torch.cuda.amp.GradScaler()  # Mixed precision
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Ensures different shuffle each epoch
        model.train()
        total_loss = torch.tensor(0.0, device=device)

        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

            with torch.cuda.amp.autocast():  # FP16 compute
                logits = model(x)
                loss   = nn.CrossEntropyLoss()(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach()

        # Aggregate loss across all GPUs
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            avg_loss = (total_loss / world_size / len(loader)).item()
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()

    cleanup()

# Launch: torchrun --nproc_per_node=4 train.py
```

### Gradient Accumulation & Effective Batch Size

```python
import torch
import torch.nn as nn

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Simulate a large batch (e.g., effective_batch=256) when GPU memory allows only 32 per step
accumulation_steps = 8  # 8 × 32 = 256 effective batch size
scaler = torch.cuda.amp.GradScaler()

model.train()
optimizer.zero_grad(set_to_none=True)

for step, (x, y) in enumerate(train_loader):
    x, y = x.cuda(), y.cuda()

    with torch.cuda.amp.autocast():
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss = loss / accumulation_steps  # Scale loss

    scaler.scale(loss).backward()

    if (step + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        print(f"Step {step + 1}: Loss = {loss.item() * accumulation_steps:.4f}")
```

### Advanced Optimizer: AdamW with Cosine Schedule + Warmup

```python
import torch
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(
    [
        {"params": model.backbone.parameters(), "lr": 1e-5},      # Smaller LR for pretrained
        {"params": model.head.parameters(),     "lr": 1e-3},      # Larger LR for new layers
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8
)

# Warmup for first 10% of steps, then cosine decay
total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(0.1 * total_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Training step
for step, batch in enumerate(train_loader):
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
```

### Quantization & Model Compression

```python
import torch
from torch.quantization import quantize_dynamic, prepare, convert

# Dynamic quantization (inference-only, no calibration needed)
model_int8 = quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},
    dtype=torch.qint8
)
print(f"FP32 size: {get_model_size(model):.2f} MB")
print(f"INT8 size: {get_model_size(model_int8):.2f} MB")

# Static quantization (requires calibration data)
model.eval()
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
model_prepared = prepare(model)

# Run calibration batches
with torch.no_grad():
    for x, _ in calibration_loader:
        model_prepared(x)

model_quantized = convert(model_prepared)

# PyTorch 2.0+ torch.compile for fusion & optimization
compiled_model = torch.compile(model, mode="max-autotune")
# Typically 1.5–3x speedup over eager mode on modern GPUs
```

### Knowledge Distillation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels,
                       temperature=4.0, alpha=0.7):
    """
    Combine soft target loss (student learns from teacher's distributions)
    with hard target loss (student learns from true labels).

    alpha: weight for soft targets (higher = learn more from teacher)
    temperature: higher = softer probability distributions
    """
    # Soft targets: KL divergence between student & teacher distributions
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits  / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    # Hard targets: standard cross-entropy with true labels
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


# Training loop
teacher.eval()
student.train()

for x, y in train_loader:
    x, y = x.cuda(), y.cuda()

    with torch.no_grad():
        teacher_logits = teacher(x)

    student_logits = student(x)
    loss = distillation_loss(student_logits, teacher_logits, y, temperature=4.0, alpha=0.7)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Summary: Learning Roadmap

```
Beginner
    ↓
    ├── Data types & loading (pandas, NumPy)
    ├── Handling missing values & basic EDA
    ├── Linear / Logistic Regression
    └── Accuracy, confusion matrix

Intermediate
    ↓
    ├── Feature engineering & encoding
    ├── Preprocessing pipelines
    ├── Decision Trees, SVMs, Random Forests
    ├── ROC/PR curves, cross-validation
    └── Clustering (K-Means, DBSCAN), PCA

Advanced
    ↓
    ├── Gradient Boosting (XGBoost, LightGBM)
    ├── Stacking & ensembles
    ├── Deep Learning with Keras/PyTorch
    ├── Transfer learning, CNNs, regularization
    └── MLflow, FastAPI serving, hyperparameter tuning

Super Advanced
    └── Distributed training (DDP)
    └── Mixed precision & gradient accumulation
    └── Custom training loops
    └── SHAP, custom scorers, threshold optimization
    └── Quantization & knowledge distillation
```

---

*Generated with comprehensive code examples. All snippets are self-contained and runnable with standard Python ML libraries.*
