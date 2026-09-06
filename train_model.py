import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/Cars.csv")

print("Original shape:", df.shape)


# ============================================================
# 2. REMOVE NEW_PRICE
# ============================================================

df.drop("New_Price", axis=1, inplace=True)


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

df.drop_duplicates(inplace=True)


# ============================================================
# 4. CLEAN NUMERICAL COLUMNS
# ============================================================

# Mileage
df["Mileage"] = (
    df["Mileage"]
    .astype(str)
    .str.extract(r"([\d.]+)")[0]
    .astype(float)
)

# Engine
df["Engine"] = (
    df["Engine"]
    .astype(str)
    .str.extract(r"([\d.]+)")[0]
    .astype(float)
)

# Power
df["Power"] = (
    df["Power"]
    .astype(str)
    .str.extract(r"([\d.]+)")[0]
    .astype(float)
)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

numerical_columns = [
    "Year",
    "Kilometers_Driven",
    "Mileage",
    "Engine",
    "Power",
    "Seats",
    "No. of Doors"
]

categorical_columns = [
    "Name",
    "Location",
    "Fuel_Type",
    "Transmission",
    "Owner_Type",
    "Colour"
]

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


# ============================================================
# 6. CREATE CAR AGE
# ============================================================

current_year = 2026

df["Car_Age"] = current_year - df["Year"]


# ============================================================
# 7. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop("Price", axis=1)

y = df["Price"]


print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print("Price")


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 9. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numerical_columns + ["Car_Age"]
        ),

        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            categorical_columns
        )
    ]
)


# ============================================================
# 10. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            random_state=42
        )
}


# ============================================================
# 11. TRAIN AND COMPARE MODELS
# ============================================================

results = {}

best_model = None
best_r2 = -np.inf
best_model_name = None


for name, model in models.items():

    print("\n===================================")
    print("Training:", name)
    print("===================================")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2   :", round(r2, 3))


    # Find best model
    if r2 > best_r2:

        best_r2 = r2
        best_model = pipeline
        best_model_name = name


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n\n===================================")
print("MODEL COMPARISON")
print("===================================")

for name, metrics in results.items():

    print(
        f"{name:20s} "
        f"R2 = {metrics['R2']:.4f} | "
        f"MAE = {metrics['MAE']:.4f} | "
        f"RMSE = {metrics['RMSE']:.4f}"
    )


# ============================================================
# 13. BEST MODEL
# ============================================================

print("\n===================================")
print("BEST MODEL")
print("===================================")

print("Model:", best_model_name)
print("R2 Score:", round(best_r2, 4))


# ============================================================
# 14. SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "model/car_price_model.pkl"
)

print("\nModel saved successfully!")

print(
    "Location: model/car_price_model.pkl"
)
