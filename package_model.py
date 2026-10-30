"""Package the final delay classifier for reuse by predict_delay.py."""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, clean_shipments

# read the final settings chosen in Part C
config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
threshold = float(config["threshold"])

data = pd.read_csv("data/modelling_table.csv")
X = data[FEATURES]
y = data["is_delayed"]

# rebuild Part C's stratified split with identical settings
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# fit the final pipeline on the training rows
model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
    ("model", RandomForestClassifier(
        random_state=42,
        n_estimators=int(config["n_estimators"]),
        max_depth=max_depth,
        min_samples_leaf=int(config["min_samples_leaf"]))),
])
model.fit(X_train, y_train)

# save the four files in models/
joblib.dump(model, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)
impute = pd.DataFrame({
    "column": list(medians.keys()),
    "median": [round(v, 4) for v in medians.values()],
})
impute.to_csv("models/impute_values.csv", index=False)

# reload check: saved model must give the same probabilities as the in-memory one
reloaded = joblib.load("models/delay_model.pkl")
p_memory = model.predict_proba(X_test)[:, 1]
p_reloaded = reloaded.predict_proba(X_test)[:, 1]
max_diff = float(np.abs(p_memory - p_reloaded).max())

check = pd.DataFrame(
    [["test_rows_scored", len(X_test)],
     ["reload_max_abs_probability_diff", round(max_diff, 4)],
     ["threshold", threshold]],
    columns=["metric", "value"], dtype=object)
check.to_csv("outputs/part_d_reload_check.csv", index=False)
print(check)
print(impute)