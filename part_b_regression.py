"""Part B: delivery-hours regression."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.shipment_prep import FEATURES
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score


data = pd.read_csv("data/modelling_table.csv")
X = data[FEATURES]           
y = data["delivery_hours"]    

# Task 1: split 70% train /  15% validation / 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print(len(X_train), len(X_val), len(X_test))   


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   
X_test_scaled = scaler.transform(X_test)         

# Largest absolute column mean of each scaled block
train_max = np.abs(X_train_scaled.mean(axis=0)).max()
test_max = np.abs(X_test_scaled.mean(axis=0)).max()

checks = pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows",
               "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": [len(X_train), len(X_val), len(X_test),
              round(train_max, 4), round(test_max, 4)],
}, dtype=object)
checks.to_csv("outputs/part_b_checks.csv", index=False)
print(checks)


def score(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

metric_rows = []   # collects rows for part_b_metrics.csv 

# Task 3: baseline -> predict the training mean for every validation row
baseline_pred = np.full(len(y_val), y_train.mean())
mae, mse, rmse, r2 = score(y_val, baseline_pred)
metric_rows.append(["mean_baseline", "val", mae, mse, rmse, r2])
print("baseline val R2:", round(r2, 4))

# Task 4: tree-depth 
depths = [1, 2, 3, 4, 6, 8, 12, None]    
depth_rows = []
for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, tree.predict(X_train)))
    val_rmse = np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))
    label = "unlimited" if d is None else d
    depth_rows.append([label, round(train_rmse, 4), round(val_rmse, 4)])

depth_table = pd.DataFrame(depth_rows, columns=["max_depth", "train_rmse", "val_rmse"])
depth_table.to_csv("outputs/part_b_tree_depth.csv", index=False)
print(depth_table)

# Best depth: lowest val_rmse. Going in order with "<" means ties keep the smaller depth.
best_depth = None
best_val = None
for d, (_, _, v) in zip(depths, depth_rows):
    if best_val is None or v < best_val:
        best_val = v
        best_depth = d
print("best depth:", best_depth, "val_rmse:", best_val)



# Task 5: four candidate models 
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()),
                                   ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)                
    val_pred = model.predict(X_val)            
    mae, mse, rmse, r2 = score(y_val, val_pred)
    metric_rows.append([name, "val", mae, mse, rmse, r2])

val_table = pd.DataFrame(metric_rows,
                         columns=["model", "split", "MAE", "MSE", "RMSE", "R2"])
print(val_table.round(4))



# ---------- Task 6: learning-rate experiment ----------
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_val)
    lr_rmse = np.sqrt(mean_squared_error(y_val, pred))
    lr_r2 = r2_score(y_val, pred)
    lr_rows.append([lr, round(lr_rmse, 4), round(lr_r2, 4)])

lr_table = pd.DataFrame(lr_rows, columns=["learning_rate", "val_rmse", "val_r2"])
lr_table.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
print(lr_table)


# ---------- Task 7: cross-validation and final test ----------
# Rank the four real candidates (not the baseline) by validation RMSE
candidates = val_table[val_table["model"] != "mean_baseline"]
ranked = candidates.sort_values("RMSE")["model"].tolist()
top_two = ranked[:2]
final_name = ranked[0]
print("validation ranking:", ranked)

# Cross-validate the top two on train + validation rows, same KFold for both
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rows = []
for name in top_two:
    scores = cross_val_score(models[name], X_trainval, y_trainval, cv=kfold)
    cv_rows.append([name, round(scores.mean(), 4), round(scores.std(), 4)])

cv_table = pd.DataFrame(cv_rows, columns=["model", "mean_r2", "std_r2"])
cv_table.to_csv("outputs/part_b_cv.csv", index=False)
print(cv_table)

# Score the final model on the test rows, once (it is already fitted on train)
test_pred = models[final_name].predict(X_test)
mae, mse, rmse, r2 = score(y_test, test_pred)
metric_rows.append([final_name, "test", mae, mse, rmse, r2])

all_metrics = pd.DataFrame(metric_rows,
                           columns=["model", "split", "MAE", "MSE", "RMSE", "R2"])
all_metrics[["MAE", "MSE", "RMSE", "R2"]] = all_metrics[["MAE", "MSE", "RMSE", "R2"]].round(4)
all_metrics.to_csv("outputs/part_b_metrics.csv", index=False)
print(all_metrics)