
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 65)
print("       FOODWISE AI - DEMAND PREDICTION MODEL V2")
print("=" * 65)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("\n1. Loading datasets...")

train = pd.read_csv("dataset/train.csv")
meal = pd.read_csv("dataset/meal_info.csv")
center = pd.read_csv("dataset/fulfilment_center_info.csv")

data = train.merge(
    meal,
    on="meal_id",
    how="left",
    validate="many_to_one"
)

data = data.merge(
    center,
    on="center_id",
    how="left",
    validate="many_to_one"
)

# Sort chronologically within each center and meal
data = data.sort_values(
    ["center_id", "meal_id", "week"]
).reset_index(drop=True)

print("Dataset shape:", data.shape)

# --------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------

print("\n2. Creating historical demand features...")

groups = ["center_id", "meal_id"]
grouped = data.groupby(groups)["num_orders"]

# Previous observations
data["lag_1"] = grouped.shift(1)
data["lag_2"] = grouped.shift(2)
data["lag_3"] = grouped.shift(3)

# Historical rolling averages
data["rolling_mean_4"] = grouped.transform(
    lambda s: s.shift(1).rolling(
        window=4, min_periods=1
    ).mean()
)

data["rolling_mean_8"] = grouped.transform(
    lambda s: s.shift(1).rolling(
        window=8, min_periods=1
    ).mean()
)

# Price features
data["price_difference"] = (
    data["base_price"] - data["checkout_price"]
)

data["discount_percentage"] = (
    data["price_difference"] /
    data["base_price"].replace(0, np.nan)
) * 100

# Week seasonality
data["week_sin"] = np.sin(
    2 * np.pi * data["week"] / 52
)

data["week_cos"] = np.cos(
    2 * np.pi * data["week"] / 52
)

# Remove rows without required historical features
data = data.dropna(
    subset=[
        "lag_1",
        "lag_2",
        "lag_3",
        "discount_percentage",
        "category",
        "cuisine",
        "center_type"
    ]
).copy()

print("Records after feature engineering:", len(data))

# --------------------------------------------------
# 3. DEFINE FEATURES
# --------------------------------------------------

categorical_features = [
    "center_id",
    "meal_id",
    "category",
    "cuisine",
    "city_code",
    "region_code",
    "center_type"
]

numerical_features = [
    "week",
    "checkout_price",
    "base_price",
    "emailer_for_promotion",
    "homepage_featured",
    "op_area",
    "price_difference",
    "discount_percentage",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_4",
    "rolling_mean_8",
    "week_sin",
    "week_cos"
]

features = categorical_features + numerical_features

X = data[features]
y = data["num_orders"]

# --------------------------------------------------
# 4. CHRONOLOGICAL VALIDATION SPLIT
# --------------------------------------------------

print("\n3. Splitting training and validation data...")

cutoff_week = int(data["week"].max()) - 19

train_mask = data["week"] < cutoff_week
valid_mask = data["week"] >= cutoff_week

X_train = X.loc[train_mask]
X_valid = X.loc[valid_mask]

y_train = y.loc[train_mask]
y_valid = y.loc[valid_mask]

print("Training records:", len(X_train))
print("Validation records:", len(X_valid))
print("Validation starts at week:", cutoff_week)

# --------------------------------------------------
# 5. CREATE PREPROCESSOR
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# --------------------------------------------------
# 6. TRAIN MODEL V2
# --------------------------------------------------

print("\n4. Creating Random Forest V2...")

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=22,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features=0.8,
                max_samples=0.8,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Logarithmic target transformation
y_train_log = np.log1p(y_train)

print("\n5. Training V2 model...")
print("Please wait...")

model.fit(X_train, y_train_log)

print("Training completed!")

# --------------------------------------------------
# 7. PREDICT AND REVERSE TRANSFORMATION
# --------------------------------------------------

print("\n6. Generating predictions...")

predictions_log = model.predict(X_valid)

predictions = np.expm1(predictions_log)

# Demand cannot be negative
predictions = np.maximum(predictions, 0)

# --------------------------------------------------
# 8. EVALUATE MODEL
# --------------------------------------------------

mae = mean_absolute_error(y_valid, predictions)

rmse = np.sqrt(
    mean_squared_error(y_valid, predictions)
)

r2 = r2_score(y_valid, predictions)

print("\n" + "=" * 65)
print("             MODEL V2 PERFORMANCE")
print("=" * 65)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")
print(f"R2 Percentage: {r2 * 100:.2f}%")

# --------------------------------------------------
# 9. SAVE MODEL
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    model,
    "models/food_demand_model_v2.pkl"
)

# Save predictions
results = pd.DataFrame({
    "Actual_Demand": y_valid.values,
    "Predicted_Demand": predictions
})

results.to_csv(
    "reports/demand_predictions_v2.csv",
    index=False
)

# Save metrics
metrics = pd.DataFrame([{
    "Model": "Random Forest V2",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}])

metrics.to_csv(
    "reports/demand_model_metrics_v2.csv",
    index=False
)

print("\nModel saved successfully!")
print("models/food_demand_model_v2.pkl")

print("\nPredictions saved successfully!")
print("reports/demand_predictions_v2.csv")

print("\n" + "=" * 65)
print("       FOODWISE AI MODEL V2 COMPLETED")
print("=" * 65)
