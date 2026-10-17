import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Ingest & clean
df = pd.read_csv("data/raw/listings.csv")
df = df.dropna(subset=["price", "latitude", "longitude", "neighbourhood_group", "room_type"])
# Filter extreme price outliers for realistic distribution
df = df[(df["price"] >= 20) & (df["price"] <= 800)]

# 2. Features and target
num_features = ["latitude", "longitude", "minimum_nights", "number_of_reviews"]
cat_features = ["neighbourhood_group", "room_type"]
target = "price"

X = df[num_features + cat_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build preprocessing + model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1))
])

print("Training Machine Learning model on your M3 chip...")
model_pipeline.fit(X_train, y_train)

# 4. Evaluate metrics
preds = model_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"Model Trained! Test RMSE: ${rmse:.2f} | R² Score: {r2:.3f}")

# 5. Export artifact
os.makedirs("models", exist_ok=True)
joblib.dump(model_pipeline, "models/price_model.joblib")
print("Exported trained pipeline to models/price_model.joblib")
