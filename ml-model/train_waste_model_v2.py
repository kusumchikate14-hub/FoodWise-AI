
import os
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# FOODWISE AI - FOOD WASTE PREDICTION MODEL V2
# Chronological validation and leakage-safe features
# ============================================================

print("=" * 70)
print("       FOODWISE AI - FOOD WASTE PREDICTION MODEL V2")
print("=" * 70)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/food_waste_dataset.csv"

MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42
MAX_TREES = 1000
EARLY_STOPPING_ROUNDS = 50

VALIDATION_START = pd.Timestamp("2025-10-01")


# ============================================================
# 2. PROGRESS BAR CALLBACK
# ============================================================

class TqdmProgressBar(TrainingCallback):

    def __init__(self, total):
        self.total = total
        self.progress = None

    def before_training(self, model):
        self.progress = tqdm(
            total=self.total,
            desc="Training Waste Model V2",
            unit="tree"
        )
        return model

    def after_iteration(self, model, epoch, evals_log):

        if self.progress is not None:
            self.progress.update(1)

            try:
                metrics = evals_log.get("validation_0", {})
                values = metrics.get("mlogloss", [])

                if values:
                    self.progress.set_postfix(
                        validation_mlogloss=f"{values[-1]:.4f}"
                    )

            except (AttributeError, IndexError, TypeError):
                pass

        return False

    def after_training(self, model):

        if self.progress is not None:
            self.progress.close()

        return model


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n1. Loading food waste dataset...")

df = pd.read_csv(DATASET_PATH)

print("Original dataset shape:", df.shape)


# ============================================================
# 4. VERIFY REQUIRED COLUMNS
# ============================================================

print("\n2. Checking required columns...")

required_columns = [
    "Date",
    "Business_Type",
    "Food_Category",
    "Purchase_Quantity",
    "Shelf_Life_Days",
    "Storage_Temperature",
    "Promotion",
    "Weather",
    "Waste_Level"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns are available!")


# ============================================================
# 5. CLEAN DATA AND EXTRACT DATE FEATURES
# ============================================================

print("\n3. Cleaning dataset...")

df = df.copy()

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# Remove records without valid dates or target labels
df = df.dropna(
    subset=["Date", "Waste_Level"]
).copy()

df["Waste_Level"] = (
    df["Waste_Level"]
    .astype(str)
    .str.strip()
    .str.title()
)

# Create calendar features
df["Month"] = df["Date"].dt.month
df["Day_of_Week"] = df["Date"].dt.day_of_week
df["Day_of_Month"] = df["Date"].dt.day

# Cyclical month features
df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

# Remove the original date and identifier
df = df.drop(
    columns=[
        "Date",
        "Inventory_ID"
    ],
    errors="ignore"
)

# Explicitly remove information unavailable at advance prediction time
excluded_columns = [
    "Units_Sold",
    "Remaining_Stock",
    "Daily_Demand",
    "Waste_Quantity",
    "Waste_Reason",
    "Donation_Made"
]

df = df.drop(
    columns=[
        col for col in excluded_columns
        if col in df.columns
    ]
)

print("Records after cleaning:", len(df))


# ============================================================
# 6. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n4. Waste level distribution:")

print(df["Waste_Level"].value_counts())


# ============================================================
# 7. SELECT FEATURES
# ============================================================

print("\n5. Selecting leakage-safe features...")

target = "Waste_Level"

X = df.drop(columns=[target]).copy()
y = df[target].copy()

categorical_features = [
    "Business_Type",
    "Food_Category",
    "Promotion",
    "Weather"
]

numerical_features = [
    "Purchase_Quantity",
    "Shelf_Life_Days",
    "Storage_Temperature",
    "Month",
    "Day_of_Week",
    "Day_of_Month",
    "Month_Sin",
    "Month_Cos"
]

# Ensure only configured features are selected
features = categorical_features + numerical_features

X = X[features]

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

print("\nFeatures excluded to prevent leakage:")
print(excluded_columns)


# ============================================================
# 8. ENCODE TARGET
# ============================================================

print("\n6. Encoding target labels...")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

class_names = label_encoder.classes_

print("Classes:", class_names)


# ============================================================
# 9. CHRONOLOGICAL TRAINING SPLIT
# ============================================================

print("\n7. Creating chronological split...")

# Keep original row indices for date-based split
date_values = pd.to_datetime(
    pd.read_csv(DATASET_PATH, usecols=["Date"])["Date"],
    errors="coerce"
)

# Reconstruct the cleaned date index to match df
# The original DataFrame indices are preserved through cleaning.
clean_dates = pd.to_datetime(
    pd.read_csv(DATASET_PATH)["Date"],
    errors="coerce"
)

clean_dates.index = df.index

train_mask = clean_dates < VALIDATION_START
valid_mask = clean_dates >= VALIDATION_START

X_train = X.loc[train_mask].copy()
X_valid = X.loc[valid_mask].copy()

y_train = y_encoded[train_mask.to_numpy()]
y_valid = y_encoded[valid_mask.to_numpy()]

print("Training records:", len(X_train))
print("Validation records:", len(X_valid))

print("Training period: January - September 2025")
print("Validation period: October - December 2025")

if len(X_train) == 0 or len(X_valid) == 0:
    raise ValueError(
        "Training or validation dataset is empty. Check dataset dates."
    )

if len(np.unique(y_train)) != len(class_names):
    raise ValueError(
        "Not all target classes are present in the training data."
    )


# ============================================================
# 10. PREPROCESSING
# ============================================================

print("\n8. Preparing preprocessing pipeline...")

numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_transformer,
            numerical_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)

X_valid_encoded = preprocessor.transform(X_valid)

print("Preprocessing completed!")


# ============================================================
# 11. CLASS BALANCING
# ============================================================

print("\n9. Calculating class weights...")

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("Class balancing completed!")


# ============================================================
# 12. CREATE XGBOOST CLASSIFIER
# ============================================================

print("\n10. Creating XGBoost classifier...")

progress_callback = TqdmProgressBar(MAX_TREES)

classifier = XGBClassifier(
    objective="multi:softprob",

    num_class=len(class_names),

    n_estimators=MAX_TREES,
    learning_rate=0.04,

    max_depth=5,
    min_child_weight=4,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=0.2,
    reg_lambda=3,

    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,

    eval_metric="mlogloss",
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,

    callbacks=[progress_callback]
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n11. Training FoodWise AI V2...")

classifier.fit(
    X_train_encoded,
    y_train,
    sample_weight=sample_weights,
    eval_set=[
        (X_valid_encoded, y_valid)
    ],
    verbose=False
)

print("\nTraining completed!")


# ============================================================
# 14. GENERATE PREDICTIONS
# ============================================================

print("\n12. Generating predictions...")

predicted_encoded = classifier.predict(
    X_valid_encoded
).astype(int)

predictions = label_encoder.inverse_transform(
    predicted_encoded
)

actual_labels = label_encoder.inverse_transform(
    y_valid
)


# ============================================================
# 15. MODEL EVALUATION
# ============================================================

print("\n13. Evaluating model...")

accuracy = accuracy_score(
    actual_labels,
    predictions
)

precision = precision_score(
    actual_labels,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    actual_labels,
    predictions,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    actual_labels,
    predictions,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    actual_labels,
    predictions,
    average="macro",
    zero_division=0
)

high_recall = recall_score(
    actual_labels,
    predictions,
    labels=["High"],
    average=None,
    zero_division=0
)[0]

best_iteration = getattr(
    classifier,
    "best_iteration",
    None
)

print("\n" + "=" * 70)
print("       FOODWISE AI - WASTE MODEL V2 PERFORMANCE")
print("=" * 70)

print(f"Accuracy           : {accuracy * 100:.2f}%")
print(f"Weighted Precision : {precision * 100:.2f}%")
print(f"Weighted Recall    : {recall * 100:.2f}%")
print(f"Weighted F1        : {weighted_f1 * 100:.2f}%")
print(f"Macro F1           : {macro_f1 * 100:.2f}%")
print(f"High Waste Recall  : {high_recall * 100:.2f}%")
print(f"Best iteration     : {best_iteration}")

print("=" * 70)

print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predictions,
        labels=class_names,
        zero_division=0
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("\n14. Creating confusion matrix...")

cm = confusion_matrix(
    actual_labels,
    predictions,
    labels=class_names
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

print(cm_df)

cm_path = os.path.join(
    REPORT_DIR,
    "waste_confusion_matrix_v2.csv"
)

cm_df.to_csv(cm_path)


# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

print("\n15. Saving predictions...")

prediction_df = X_valid.copy()

prediction_df["actual_waste_level"] = actual_labels
prediction_df["predicted_waste_level"] = predictions

prediction_df["correct_prediction"] = (
    prediction_df["actual_waste_level"]
    == prediction_df["predicted_waste_level"]
)

prediction_path = os.path.join(
    REPORT_DIR,
    "waste_predictions_v2.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 18. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Weighted_Precision",
        "Weighted_Recall",
        "Weighted_F1",
        "Macro_F1",
        "High_Waste_Recall",
        "Best_Iteration",
        "Training_Records",
        "Validation_Records"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        weighted_f1,
        macro_f1,
        high_recall,
        best_iteration,
        len(X_train),
        len(X_valid)
    ]
})

metrics_path = os.path.join(
    REPORT_DIR,
    "waste_model_metrics_v2.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 19. SAVE COMPLETE MODEL
# ============================================================

print("\n16. Saving trained model...")

# Remove callback containing the live progress-bar stream
classifier.set_params(callbacks=None)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)

model_artifact = {
    "pipeline": pipeline,
    "label_encoder": label_encoder,
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "prediction_classes": class_names.tolist(),
    "excluded_columns": excluded_columns,
    "validation_start": str(VALIDATION_START.date())
}

model_path = os.path.join(
    MODEL_DIR,
    "food_waste_model_v2.pkl"
)

joblib.dump(
    model_artifact,
    model_path
)

print("Model saved:", model_path)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("       FOODWISE AI - WASTE MODEL V2 COMPLETED")
print("=" * 70)

print("Model file       :", model_path)
print("Prediction file  :", prediction_path)
print("Metrics file     :", metrics_path)
print("Confusion matrix :", cm_path)

print(f"\nAccuracy          : {accuracy * 100:.2f}%")
print(f"Macro F1          : {macro_f1 * 100:.2f}%")
print(f"High Waste Recall : {high_recall * 100:.2f}%")

print("\nAll tasks completed successfully!")
print("=" * 70)
