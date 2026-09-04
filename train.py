import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score


# 1. Load dataset
df = pd.read_csv("Loan_default.csv")
# 2. Remove ID
df = df.drop("LoanID", axis=1)

# 3. Separate input and target
X = df.drop("Default", axis=1)
y = df["Default"]

# 4. Identify columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# 5. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# 6. Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# 7. Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. MLflow
mlflow.set_experiment("Loan_Default_Prediction")

with mlflow.start_run():

    # 10. Train
    pipeline.fit(X_train, y_train)

    # 11. Prediction
    predictions = pipeline.predict(X_test)

    # 12. Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("F1 Score:", f1)

    # 13. Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1_score", f1)

    # 14. Log model
    mlflow.sklearn.log_model(
        pipeline,
        "loan_default_model"
    )

    # 15. Save model
    joblib.dump(pipeline, "loan_default_model.pkl")

print("Model trained successfully!")