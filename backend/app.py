
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime, date
from decimal import Decimal
import mysql.connector
import joblib
import numpy as np
import pandas as pd
import os


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "ml-model",
    "models",
    "food_waste_model_v2.pkl"
)

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "ml-model",
    "dataset",
    "food_waste_dataset.csv"
)

# =========================================================
# DEMAND FORECASTING FILE PATHS
# =========================================================

DEMAND_MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "ml-model",
    "models",
    "food_demand_model_v2.pkl"
)

DEMAND_TRAIN_PATH = os.path.join(
    PROJECT_DIR,
    "ml-model",
    "dataset",
    "train.csv"
)

MEAL_INFO_PATH = os.path.join(
    PROJECT_DIR,
    "ml-model",
    "dataset",
    "meal_info.csv"
)

CENTER_INFO_PATH = os.path.join(
    PROJECT_DIR,
    "ml-model",
    "dataset",
    "fulfilment_center_info.csv"
)

DEMAND_MODEL_NAME = "FoodWise AI Demand Forecasting V2"

MODEL_NAME = "FoodWise AI Waste Prediction V2"

app = Flask(__name__)
CORS(app)


# =========================================================
# 2. LOAD WASTE PREDICTION MODEL
# =========================================================

waste_pipeline = None
label_encoder = None

try:
    model_artifact = joblib.load(MODEL_PATH)

    waste_pipeline = model_artifact["pipeline"]
    label_encoder = model_artifact["label_encoder"]

    print("Waste prediction model loaded successfully.")
    print("Model:", MODEL_NAME)

except Exception:
    app.logger.exception("Could not load the waste prediction model.")


# =========================================================
# LOAD DEMAND FORECASTING MODEL AND DATASETS
# =========================================================

demand_model = None
demand_history = None
meal_info = None
center_info = None

try:
    # Load trained demand forecasting model
    demand_model = joblib.load(DEMAND_MODEL_PATH)

    # Load historical demand data
    demand_history = pd.read_csv(DEMAND_TRAIN_PATH)

    # Load meal information
    meal_info = pd.read_csv(MEAL_INFO_PATH)

    # Load fulfilment centre information
    center_info = pd.read_csv(CENTER_INFO_PATH)

    print("\nDemand forecasting model loaded successfully.")
    print("Historical demand records:", len(demand_history))
    print("Meal records:", len(meal_info))
    print("Fulfilment centre records:", len(center_info))

except Exception:
    app.logger.exception(
        "Could not load the demand forecasting model or datasets."
    )


# =========================================================
# DEMAND FORECASTING API
# =========================================================

@app.route("/api/predict-demand", methods=["POST"])
def predict_demand():

    # Check whether model and datasets loaded
    if (
        demand_model is None
        or demand_history is None
        or meal_info is None
        or center_info is None
    ):
        return jsonify({
            "status": "error",
            "message": "Demand forecasting model or datasets are unavailable."
        }), 503

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Please provide valid JSON input."
            }), 400

        # Validate required inputs
        required_fields = [
            "center_id",
            "meal_id",
            "week",
            "checkout_price",
            "base_price",
            "emailer_for_promotion",
            "homepage_featured"
        ]

        missing_fields = [
            field for field in required_fields
            if field not in data
            or data[field] is None
            or str(data[field]).strip() == ""
        ]

        if missing_fields:
            return jsonify({
                "status": "error",
                "message": "Required fields are missing.",
                "missing_fields": missing_fields
            }), 400

        # Convert and validate numeric inputs
        try:
            center_id = int(data["center_id"])
            meal_id = int(data["meal_id"])
            week = int(data["week"])

            checkout_price = float(data["checkout_price"])
            base_price = float(data["base_price"])

            emailer_for_promotion = int(
                data["emailer_for_promotion"]
            )

            homepage_featured = int(
                data["homepage_featured"]
            )

        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Please provide valid numeric inputs."
            }), 400

        numeric_values = [
            checkout_price,
            base_price
        ]

        if not all(np.isfinite(value) for value in numeric_values):
            return jsonify({
                "status": "error",
                "message": "Prices must be finite numbers."
            }), 400

        if week < 1:
            return jsonify({
                "status": "error",
                "message": "Week must be at least 1."
            }), 400

        if checkout_price <= 0 or base_price <= 0:
            return jsonify({
                "status": "error",
                "message": "Prices must be greater than zero."
            }), 400

        if emailer_for_promotion not in [0, 1]:
            return jsonify({
                "status": "error",
                "message": "Email promotion must be 0 or 1."
            }), 400

        if homepage_featured not in [0, 1]:
            return jsonify({
                "status": "error",
                "message": "Homepage featured must be 0 or 1."
            }), 400

        # Look up fulfilment centre information
        selected_center = center_info[
            center_info["center_id"] == center_id
        ]

        if selected_center.empty:
            return jsonify({
                "status": "error",
                "message": "Invalid fulfilment centre ID."
            }), 400

        # Look up meal information
        selected_meal = meal_info[
            meal_info["meal_id"] == meal_id
        ]

        if selected_meal.empty:
            return jsonify({
                "status": "error",
                "message": "Invalid meal ID."
            }), 400

        # Get the historical records for this centre and meal
        history = demand_history[
            (demand_history["center_id"] == center_id)
            &
            (demand_history["meal_id"] == meal_id)
            &
            (demand_history["week"] < week)
        ].sort_values("week")

        # Eight previous observations are needed for rolling_mean_8
        if len(history) < 8:
            return jsonify({
                "status": "error",
                "message": (
                    "At least 8 previous observations are required "
                    "for this centre and meal combination."
                ),
                "available_observations": int(len(history))
            }), 400

        previous_orders = history["num_orders"].astype(float).tolist()

        # Construct lag and rolling features
        lag_1 = previous_orders[-1]
        lag_2 = previous_orders[-2]
        lag_3 = previous_orders[-3]

        rolling_mean_4 = float(
            np.mean(previous_orders[-4:])
        )

        rolling_mean_8 = float(
            np.mean(previous_orders[-8:])
        )

        # Retrieve category and centre details
        meal = selected_meal.iloc[0]
        center = selected_center.iloc[0]

        # Calculate price features
        price_difference = base_price - checkout_price

        discount_percentage = (
            price_difference / base_price
        ) * 100

        # Calculate seasonal week features
        week_sin = np.sin(2 * np.pi * week / 52)
        week_cos = np.cos(2 * np.pi * week / 52)

        # Prepare all 22 features in training order
        model_input = pd.DataFrame([{
            "center_id": center_id,
            "meal_id": meal_id,
            "category": meal["category"],
            "cuisine": meal["cuisine"],
            "city_code": center["city_code"],
            "region_code": center["region_code"],
            "center_type": center["center_type"],
            "week": week,
            "checkout_price": checkout_price,
            "base_price": base_price,
            "emailer_for_promotion": emailer_for_promotion,
            "homepage_featured": homepage_featured,
            "op_area": center["op_area"],
            "price_difference": price_difference,
            "discount_percentage": discount_percentage,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "rolling_mean_4": rolling_mean_4,
            "rolling_mean_8": rolling_mean_8,
            "week_sin": week_sin,
            "week_cos": week_cos
        }])

        # Predict log-transformed demand
        predicted_log_orders = float(
            demand_model.predict(model_input)[0]
        )

        # Reverse log1p transformation
        predicted_orders = float(
            np.expm1(predicted_log_orders)
        )

        # Demand cannot be negative
        predicted_orders = max(0, predicted_orders)

        predicted_orders = round(predicted_orders, 2)

        return jsonify({
            "status": "success",
            "message": "Demand prediction generated successfully.",

            "prediction": {
                "predicted_orders": predicted_orders,
                "unit": "orders"
            },

            "input": {
                "center_id": center_id,
                "meal_id": meal_id,
                "week": week,
                "checkout_price": checkout_price,
                "base_price": base_price,
                "emailer_for_promotion": emailer_for_promotion,
                "homepage_featured": homepage_featured
            },

            "historical_features": {
                "available_observations": len(history),
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "rolling_mean_4": round(rolling_mean_4, 2),
                "rolling_mean_8": round(rolling_mean_8, 2)
            },

            "model": DEMAND_MODEL_NAME,

            "notice": (
                "Experimental demand forecast based on the "
                "Food Demand Forecasting Kaggle dataset. "
                "Predictions are estimates, not guaranteed orders."
            )
        }), 200

    except Exception:
        app.logger.exception("Demand prediction failed.")

        return jsonify({
            "status": "error",
            "message": "Could not generate demand prediction."
        }), 500

# =========================================================
# 3. MYSQL CONNECTION
# =========================================================

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "foodwise_db")
    )


def close_database_resources(cursor=None, connection=None):
    if cursor is not None:
        try:
            cursor.close()
        except mysql.connector.Error:
            app.logger.exception("Could not close database cursor.")

    if connection is not None:
        try:
            if connection.is_connected():
                connection.close()
        except mysql.connector.Error:
            app.logger.exception("Could not close database connection.")


# =========================================================
# 4. HOME ROUTE
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "FoodWise AI",
        "message": "FoodWise AI backend is running!",
        "status": "success"
    })


@app.route("/api/routes", methods=["GET"])
def list_api_routes():
    """Show routes registered in this exact running Flask process."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "path": str(rule),
            "methods": sorted(
                method for method in rule.methods
                if method not in {"HEAD", "OPTIONS"}
            )
        })
    return jsonify({"status": "success", "routes": routes}), 200


# =========================================================
# 5. BACKEND HEALTH CHECK
# =========================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "backend": "running",
        "project": "FoodWise AI",
        "model_loaded": waste_pipeline is not None
    })


# =========================================================
# 6. DATABASE HEALTH CHECK
# =========================================================

@app.route("/api/db-health", methods=["GET"])
def database_health():

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT DATABASE()")

        database_name = cursor.fetchone()[0]

        return jsonify({
            "database": "connected",
            "database_name": database_name
        })

    except mysql.connector.Error:
        app.logger.exception("Database health check failed.")

        return jsonify({
            "database": "connection_failed",
            "error": "Could not connect to the configured database."
        }), 500

    finally:
        close_database_resources(cursor, connection)


# =========================================================
# 7. WASTE PREDICTION API
# =========================================================

@app.route("/api/predict-waste", methods=["POST"])
def predict_waste():

    if waste_pipeline is None or label_encoder is None:
        return jsonify({
            "error": "Waste prediction model is unavailable."
        }), 503

    connection = None
    cursor = None

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please provide valid JSON input."
            }), 400

        # Validate required fields
        required_fields = [
            "business_type",
            "food_category",
            "purchase_quantity",
            "shelf_life_days",
            "storage_temperature",
            "promotion",
            "weather",
            "date"
        ]

        missing_fields = [
            field for field in required_fields
            if field not in data
            or data[field] is None
            or str(data[field]).strip() == ""
        ]

        if missing_fields:
            return jsonify({
                "error": "Required fields are missing.",
                "missing_fields": missing_fields
            }), 400

        # Validate date
        try:
            prediction_date = datetime.strptime(
                str(data["date"]),
                "%Y-%m-%d"
            ).date()

        except ValueError:
            return jsonify({
                "error": "Date must use YYYY-MM-DD format."
            }), 400

        # Validate numerical values
        try:
            purchase_quantity = float(data["purchase_quantity"])
            shelf_life_days = float(data["shelf_life_days"])
            storage_temperature = float(data["storage_temperature"])

        except (ValueError, TypeError):
            return jsonify({
                "error": "Quantity, shelf life and temperature must be numeric."
            }), 400

        numerical_values = [
            purchase_quantity,
            shelf_life_days,
            storage_temperature
        ]

        if not all(np.isfinite(value) for value in numerical_values):
            return jsonify({
                "error": "Numerical values must be finite."
            }), 400

        if purchase_quantity <= 0:
            return jsonify({
                "error": "Purchase quantity must be greater than zero."
            }), 400

        if shelf_life_days < 0:
            return jsonify({
                "error": "Shelf life cannot be negative."
            }), 400

        # Categorical values
        business_type = str(data["business_type"]).strip()
        food_category = str(data["food_category"]).strip()
        promotion = str(data["promotion"]).strip()
        weather = str(data["weather"]).strip()

        # Generate date features
        month = prediction_date.month
        day_of_week = prediction_date.weekday()
        day_of_month = prediction_date.day

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # Prepare model input
        model_input = pd.DataFrame([{
            "Business_Type": business_type,
            "Food_Category": food_category,
            "Promotion": promotion,
            "Weather": weather,
            "Purchase_Quantity": purchase_quantity,
            "Shelf_Life_Days": shelf_life_days,
            "Storage_Temperature": storage_temperature,
            "Month": month,
            "Day_of_Week": day_of_week,
            "Day_of_Month": day_of_month,
            "Month_Sin": month_sin,
            "Month_Cos": month_cos
        }])

        # Predict class
        predicted_encoded = int(
            waste_pipeline.predict(model_input)[0]
        )

        predicted_label = str(
            label_encoder.inverse_transform(
                [predicted_encoded]
            )[0]
        )

        # Calculate probabilities
        probabilities = waste_pipeline.predict_proba(model_input)[0]

        class_probabilities = {
            str(class_name): round(
                float(probability) * 100, 2
            )
            for class_name, probability in zip(
                label_encoder.classes_,
                probabilities
            )
        }

        high_probability = class_probabilities["High"]
        medium_probability = class_probabilities["Medium"]
        low_probability = class_probabilities["Low"]

        # Save prediction to MySQL
        connection = get_db_connection()
        cursor = connection.cursor()

        insert_query = """
            INSERT INTO waste_prediction_history (
                business_type,
                food_category,
                purchase_quantity,
                shelf_life_days,
                storage_temperature,
                promotion,
                weather,
                prediction_date,
                predicted_waste_level,
                high_probability,
                medium_probability,
                low_probability,
                model_name
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
        """

        values = (
            business_type,
            food_category,
            purchase_quantity,
            shelf_life_days,
            storage_temperature,
            promotion,
            weather,
            prediction_date,
            predicted_label,
            high_probability,
            medium_probability,
            low_probability,
            MODEL_NAME
        )

        cursor.execute(insert_query, values)
        connection.commit()

        prediction_id = cursor.lastrowid

        return jsonify({
            "status": "success",
            "message": "Prediction generated and saved successfully.",
            "prediction_id": prediction_id,
            "prediction": predicted_label,
            "probabilities": class_probabilities,
            "input": {
                "business_type": business_type,
                "food_category": food_category,
                "purchase_quantity": purchase_quantity,
                "date": str(prediction_date)
            },
            "model": MODEL_NAME,
            "notice": (
                "Experimental prediction based on a synthetic "
                "training dataset."
            )
        }), 200

    except mysql.connector.Error:
        if connection is not None and connection.is_connected():
            connection.rollback()

        app.logger.exception("Could not save prediction to MySQL.")

        return jsonify({
            "error": "Prediction could not be saved to the database."
        }), 500

    except Exception:
        if connection is not None and connection.is_connected():
            connection.rollback()

        app.logger.exception("Waste prediction failed.")

        return jsonify({
            "error": "The prediction could not be completed."
        }), 500

    finally:
        close_database_resources(cursor, connection)


# =========================================================
# 8. PREDICTION HISTORY API
# =========================================================

@app.route("/api/prediction-history", methods=["GET"])
def prediction_history():

    connection = None
    cursor = None

    try:
        try:
            limit = int(request.args.get("limit", 50))
        except ValueError:
            return jsonify({
                "error": "Limit must be a whole number."
            }), 400

        limit = max(1, min(limit, 200))

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                prediction_id,
                business_type,
                food_category,
                purchase_quantity,
                shelf_life_days,
                storage_temperature,
                promotion,
                weather,
                prediction_date,
                predicted_waste_level,
                high_probability,
                medium_probability,
                low_probability,
                model_name,
                created_at
            FROM waste_prediction_history
            ORDER BY prediction_id DESC
            LIMIT %s
        """

        cursor.execute(query, (limit,))
        records = cursor.fetchall()

        # Convert database values to JSON-compatible values
        for record in records:
            for key, value in record.items():

                if isinstance(value, datetime):
                    record[key] = value.isoformat(sep=" ")

                elif isinstance(value, date):
                    record[key] = value.isoformat()

                elif isinstance(value, Decimal):
                    record[key] = float(value)

        return jsonify({
            "status": "success",
            "total_returned": len(records),
            "predictions": records
        })

    except mysql.connector.Error:
        app.logger.exception("Prediction history retrieval failed.")

        return jsonify({
            "error": "Could not retrieve prediction history."
        }), 500

    finally:
        close_database_resources(cursor, connection)


# =========================================================
# 9. DASHBOARD ANALYTICS API
# =========================================================

@app.route("/api/dashboard-stats", methods=["GET"])
def dashboard_stats():

    connection = None
    cursor = None

    try:
        # Check dataset availability
        if not os.path.exists(DATASET_PATH):
            return jsonify({
                "error": "Food waste dataset file was not found.",
                "expected_path": DATASET_PATH
            }), 404

        # Load synthetic dataset
        dataset = pd.read_csv(DATASET_PATH)

        required_columns = [
            "Food_Category",
            "Waste_Quantity",
            "Waste_Level"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in dataset.columns
        ]

        if missing_columns:
            return jsonify({
                "error": "Required dataset columns are missing.",
                "missing_columns": missing_columns
            }), 500

        # Calculate total waste
        waste_quantity = pd.to_numeric(
            dataset["Waste_Quantity"],
            errors="coerce"
        )

        total_food_waste = float(
            waste_quantity.fillna(0).sum()
        )

        # Calculate food categories
        category_count = int(
            dataset["Food_Category"].nunique()
        )

        # Calculate waste by category
        category_data = (
            dataset.assign(
                Waste_Quantity=waste_quantity.fillna(0)
            )
            .groupby("Food_Category")["Waste_Quantity"]
            .sum()
            .sort_values(ascending=False)
        )

        waste_by_category = [
            {
                "category": str(category),
                "waste_quantity": round(float(quantity), 2)
            }
            for category, quantity in category_data.items()
        ]

        # Retrieve live MySQL prediction statistics
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                COUNT(*) AS total_predictions,

                COALESCE(
                    SUM(predicted_waste_level = 'High'), 0
                ) AS high_predictions,

                COALESCE(
                    SUM(predicted_waste_level = 'Medium'), 0
                ) AS medium_predictions,

                COALESCE(
                    SUM(predicted_waste_level = 'Low'), 0
                ) AS low_predictions

            FROM waste_prediction_history
        """)

        stats = cursor.fetchone()

        prediction_stats = {
            "total_predictions": int(stats["total_predictions"]),
            "high_predictions": int(stats["high_predictions"]),
            "medium_predictions": int(stats["medium_predictions"]),
            "low_predictions": int(stats["low_predictions"])
        }

        return jsonify({
            "status": "success",

            "dataset_statistics": {
                "total_food_waste": round(total_food_waste, 2),
                "food_categories": category_count,
                "dataset_records": int(len(dataset)),
                "data_type": "Synthetic dataset"
            },

            "prediction_statistics": prediction_stats,

            "waste_by_category": waste_by_category
        }), 200

    except mysql.connector.Error:
        app.logger.exception("Dashboard database query failed.")

        return jsonify({
            "error": "Could not retrieve dashboard database statistics."
        }), 500

    except Exception:
        app.logger.exception("Dashboard analytics calculation failed.")

        return jsonify({
            "error": "Could not calculate dashboard analytics."
        }), 500

    finally:
        close_database_resources(cursor, connection)



# ==========================================
# INVENTORY MANAGEMENT API
# ==========================================

@app.route("/api/inventory", methods=["GET"])
def get_inventory():
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT *
            FROM inventory
            ORDER BY inventory_id DESC
        """)

        records = cursor.fetchall()

        for item in records:
            for field in ["purchase_date", "expiry_date",
                          "created_at", "updated_at"]:
                if item.get(field):
                    item[field] = item[field].isoformat()

            for field in ["quantity", "reorder_level",
                          "storage_temperature"]:
                if item.get(field) is not None:
                    item[field] = float(item[field])

        return jsonify({
            "status": "success",
            "count": len(records),
            "inventory": records
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


@app.route("/api/inventory", methods=["POST"])
def add_inventory():
    connection = None
    cursor = None

    try:
        data = request.get_json(silent=True) or {}

        required_fields = [
            "item_name",
            "food_category",
            "business_type",
            "quantity",
            "unit",
            "purchase_date",
            "expiry_date"
        ]

        for field in required_fields:
            if data.get(field) is None or str(data[field]).strip() == "":
                return jsonify({
                    "status": "error",
                    "message": f"{field} is required."
                }), 400

        try:
            quantity = float(data["quantity"])
            reorder_level = float(data.get("reorder_level", 5))
            storage_temperature = data.get("storage_temperature")

            if storage_temperature == "":
                storage_temperature = None
            elif storage_temperature is not None:
                storage_temperature = float(storage_temperature)

            purchase_date = date.fromisoformat(data["purchase_date"])
            expiry_date = date.fromisoformat(data["expiry_date"])

        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Please enter valid quantities, temperature and dates."
            }), 400

        if quantity < 0 or reorder_level < 0:
            return jsonify({
                "status": "error",
                "message": "Quantity and reorder level cannot be negative."
            }), 400

        if expiry_date < purchase_date:
            return jsonify({
                "status": "error",
                "message": "Expiry date cannot be before purchase date."
            }), 400

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO inventory (
                item_name,
                food_category,
                business_type,
                quantity,
                unit,
                purchase_date,
                expiry_date,
                reorder_level,
                storage_temperature
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            str(data["item_name"]).strip(),
            str(data["food_category"]).strip(),
            str(data["business_type"]).strip(),
            quantity,
            str(data["unit"]).strip(),
            purchase_date,
            expiry_date,
            reorder_level,
            storage_temperature
        ))

        connection.commit()
        inventory_id = cursor.lastrowid

        return jsonify({
            "status": "success",
            "message": "Inventory item added successfully.",
            "inventory_id": inventory_id
        }), 201

    except Exception as e:
        if connection:
            connection.rollback()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


@app.route("/api/inventory/<int:inventory_id>", methods=["PUT"])
def update_inventory(inventory_id):
    connection = None
    cursor = None

    try:
        data = request.get_json(silent=True) or {}

        required_fields = [
            "item_name",
            "food_category",
            "business_type",
            "quantity",
            "unit",
            "purchase_date",
            "expiry_date"
        ]

        for field in required_fields:
            if data.get(field) is None or str(data[field]).strip() == "":
                return jsonify({
                    "status": "error",
                    "message": f"{field} is required."
                }), 400

        try:
            quantity = float(data["quantity"])
            reorder_level = float(data.get("reorder_level", 5))
            storage_temperature = data.get("storage_temperature")

            if storage_temperature == "":
                storage_temperature = None
            elif storage_temperature is not None:
                storage_temperature = float(storage_temperature)

            purchase_date = date.fromisoformat(data["purchase_date"])
            expiry_date = date.fromisoformat(data["expiry_date"])

        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Please enter valid quantities, temperature and dates."
            }), 400

        if quantity < 0 or reorder_level < 0:
            return jsonify({
                "status": "error",
                "message": "Quantity and reorder level cannot be negative."
            }), 400

        if expiry_date < purchase_date:
            return jsonify({
                "status": "error",
                "message": "Expiry date cannot be before purchase date."
            }), 400

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            UPDATE inventory
            SET item_name = %s,
                food_category = %s,
                business_type = %s,
                quantity = %s,
                unit = %s,
                purchase_date = %s,
                expiry_date = %s,
                reorder_level = %s,
                storage_temperature = %s
            WHERE inventory_id = %s
        """, (
            str(data["item_name"]).strip(),
            str(data["food_category"]).strip(),
            str(data["business_type"]).strip(),
            quantity,
            str(data["unit"]).strip(),
            purchase_date,
            expiry_date,
            reorder_level,
            storage_temperature,
            inventory_id
        ))

        if cursor.rowcount == 0:
            cursor.execute(
                "SELECT inventory_id FROM inventory WHERE inventory_id = %s",
                (inventory_id,)
            )

            if not cursor.fetchone():
                return jsonify({
                    "status": "error",
                    "message": "Inventory item not found."
                }), 404

        connection.commit()

        return jsonify({
            "status": "success",
            "message": "Inventory item updated successfully."
        }), 200

    except Exception as e:
        if connection:
            connection.rollback()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


@app.route("/api/inventory/<int:inventory_id>", methods=["DELETE"])
def delete_inventory(inventory_id):
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            "DELETE FROM inventory WHERE inventory_id = %s",
            (inventory_id,)
        )

        if cursor.rowcount == 0:
            return jsonify({
                "status": "error",
                "message": "Inventory item not found."
            }), 404

        connection.commit()

        return jsonify({
            "status": "success",
            "message": "Inventory item deleted successfully."
        }), 200

    except Exception as e:
        if connection:
            connection.rollback()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()




# =========================================================
# FOOD DONATION MANAGEMENT API
# =========================================================

VALID_DONATION_STATUSES = [
    "Pending",
    "Accepted",
    "Completed",
    "Cancelled"
]


@app.route("/api/donations", methods=["GET"])
def get_donations():

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT *
            FROM food_donations
            ORDER BY donation_id DESC
        """)

        records = cursor.fetchall()

        for record in records:
            for field in [
                "donation_date",
                "expiry_date",
                "created_at",
                "updated_at"
            ]:
                if record.get(field):
                    value = record[field]

                    if isinstance(value, datetime):
                        record[field] = value.isoformat(sep=" ")
                    else:
                        record[field] = value.isoformat()

            if record.get("quantity") is not None:
                record["quantity"] = float(record["quantity"])

        return jsonify({
            "status": "success",
            "count": len(records),
            "donations": records
        }), 200

    except mysql.connector.Error:
        app.logger.exception("Could not retrieve donations.")

        return jsonify({
            "status": "error",
            "message": "Could not retrieve donation records."
        }), 500

    finally:
        close_database_resources(cursor, connection)


@app.route("/api/donations", methods=["POST"])
def add_donation():

    connection = None
    cursor = None

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Please provide valid JSON input."
            }), 400

        required_fields = [
            "food_name",
            "food_category",
            "quantity",
            "unit",
            "donor_name",
            "recipient_name",
            "donation_date",
            "expiry_date"
        ]

        missing_fields = [
            field for field in required_fields
            if data.get(field) is None
            or str(data[field]).strip() == ""
        ]

        if missing_fields:
            return jsonify({
                "status": "error",
                "message": "Required fields are missing.",
                "missing_fields": missing_fields
            }), 400

        try:
            quantity = float(data["quantity"])

            donation_date = date.fromisoformat(
                str(data["donation_date"])
            )

            expiry_date = date.fromisoformat(
                str(data["expiry_date"])
            )

        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Enter a valid quantity and dates in YYYY-MM-DD format."
            }), 400

        if not np.isfinite(quantity) or quantity <= 0:
            return jsonify({
                "status": "error",
                "message": "Donation quantity must be greater than zero."
            }), 400

        if expiry_date < donation_date:
            return jsonify({
                "status": "error",
                "message": "Expiry date cannot be before donation date."
            }), 400

        donation_status = data.get(
            "donation_status", "Pending"
        )

        if donation_status not in VALID_DONATION_STATUSES:
            return jsonify({
                "status": "error",
                "message": "Invalid donation status."
            }), 400

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO food_donations (
                food_name,
                food_category,
                quantity,
                unit,
                donor_name,
                recipient_name,
                recipient_contact,
                donation_date,
                expiry_date,
                donation_status,
                notes
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
        """, (
            str(data["food_name"]).strip(),
            str(data["food_category"]).strip(),
            quantity,
            str(data["unit"]).strip(),
            str(data["donor_name"]).strip(),
            str(data["recipient_name"]).strip(),
            data.get("recipient_contact") or None,
            donation_date,
            expiry_date,
            donation_status,
            data.get("notes") or None
        ))

        connection.commit()

        donation_id = cursor.lastrowid

        return jsonify({
            "status": "success",
            "message": "Food donation registered successfully.",
            "donation_id": donation_id
        }), 201

    except mysql.connector.Error:
        if connection and connection.is_connected():
            connection.rollback()

        app.logger.exception("Could not save donation.")

        return jsonify({
            "status": "error",
            "message": "Could not save the donation to the database."
        }), 500

    except Exception:
        if connection and connection.is_connected():
            connection.rollback()

        app.logger.exception("Donation registration failed.")

        return jsonify({
            "status": "error",
            "message": "Donation registration failed."
        }), 500

    finally:
        close_database_resources(cursor, connection)


@app.route("/api/donations/<int:donation_id>", methods=["PUT"])
def update_donation(donation_id):

    connection = None
    cursor = None

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Please provide valid JSON input."
            }), 400

        required_fields = [
            "food_name",
            "food_category",
            "quantity",
            "unit",
            "donor_name",
            "recipient_name",
            "donation_date",
            "expiry_date",
            "donation_status"
        ]

        missing_fields = [
            field for field in required_fields
            if data.get(field) is None
            or str(data[field]).strip() == ""
        ]

        if missing_fields:
            return jsonify({
                "status": "error",
                "message": "Required fields are missing.",
                "missing_fields": missing_fields
            }), 400

        try:
            quantity = float(data["quantity"])

            donation_date = date.fromisoformat(
                str(data["donation_date"])
            )

            expiry_date = date.fromisoformat(
                str(data["expiry_date"])
            )

        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Enter a valid quantity and dates."
            }), 400

        if not np.isfinite(quantity) or quantity <= 0:
            return jsonify({
                "status": "error",
                "message": "Donation quantity must be greater than zero."
            }), 400

        if expiry_date < donation_date:
            return jsonify({
                "status": "error",
                "message": "Expiry date cannot be before donation date."
            }), 400

        donation_status = data["donation_status"]

        if donation_status not in VALID_DONATION_STATUSES:
            return jsonify({
                "status": "error",
                "message": "Invalid donation status."
            }), 400

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            UPDATE food_donations
            SET food_name = %s,
                food_category = %s,
                quantity = %s,
                unit = %s,
                donor_name = %s,
                recipient_name = %s,
                recipient_contact = %s,
                donation_date = %s,
                expiry_date = %s,
                donation_status = %s,
                notes = %s
            WHERE donation_id = %s
        """, (
            str(data["food_name"]).strip(),
            str(data["food_category"]).strip(),
            quantity,
            str(data["unit"]).strip(),
            str(data["donor_name"]).strip(),
            str(data["recipient_name"]).strip(),
            data.get("recipient_contact") or None,
            donation_date,
            expiry_date,
            donation_status,
            data.get("notes") or None,
            donation_id
        ))

        if cursor.rowcount == 0:
            cursor.execute("""
                SELECT donation_id
                FROM food_donations
                WHERE donation_id = %s
            """, (donation_id,))

            if not cursor.fetchone():
                return jsonify({
                    "status": "error",
                    "message": "Donation record not found."
                }), 404

        connection.commit()

        return jsonify({
            "status": "success",
            "message": "Donation record updated successfully."
        }), 200

    except mysql.connector.Error:
        if connection and connection.is_connected():
            connection.rollback()

        app.logger.exception("Could not update donation.")

        return jsonify({
            "status": "error",
            "message": "Could not update the donation."
        }), 500

    finally:
        close_database_resources(cursor, connection)


@app.route("/api/donations/<int:donation_id>", methods=["DELETE"])
def delete_donation(donation_id):

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            DELETE FROM food_donations
            WHERE donation_id = %s
        """, (donation_id,))

        if cursor.rowcount == 0:
            return jsonify({
                "status": "error",
                "message": "Donation record not found."
            }), 404

        connection.commit()

        return jsonify({
            "status": "success",
            "message": "Donation record deleted successfully."
        }), 200

    except mysql.connector.Error:
        if connection and connection.is_connected():
            connection.rollback()

        app.logger.exception("Could not delete donation.")

        return jsonify({
            "status": "error",
            "message": "Could not delete the donation."
        }), 500

    finally:
        close_database_resources(cursor, connection)


# =========================================================
# 10. APPLICATION START
# =========================================================

if __name__ == "__main__":

    print("\n====================================")
    print("       FOODWISE AI BACKEND")
    print("====================================")

    print("\nRegistered API routes:")

    for rule in app.url_map.iter_rules():
        print(f"{','.join(sorted(rule.methods))}  {rule}")

    print("\nStarting Flask server...\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
