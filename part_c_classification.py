import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES

data = pd.read_csv("data/modelling_table.csv")
X = data[FEATURES]
y = data["is_delayed"]

# Task 1: stratified two-step split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(len(X_train), len(X_val), len(X_test))                   
print(y_train.sum(), y_val.sum(), y_test.sum())                


def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return [int(tn), int(fp), int(fn), int(tp),
            round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)]


# Task 2: blind baseline and three classifiers, scored on validation at the 0.5 rule
model_rows = []

blind_pred = np.zeros(len(y_val), dtype=int)       # "not delayed" for every parcel
model_rows.append(["blind_all_on_time"] + evaluate(y_val, blind_pred))

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
logreg.fit(X_train, y_train)
model_rows.append(["logistic_regression"] + evaluate(y_val, logreg.predict(X_val)))

# decision tree: pick the depth with the highest validation F1 
best_d = None
best_f1 = -1
for d in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    f1 = f1_score(y_val, tree.predict(X_val))
    if f1 > best_f1:               # keeps the smaller depth on a tie
        best_f1 = f1
        best_d = d
best_tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42)
best_tree.fit(X_train, y_train)
model_rows.append([f"decision_tree_depth_{best_d}"] + evaluate(y_val, best_tree.predict(X_val)))

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
model_rows.append(["random_forest_default"] + evaluate(y_val, forest.predict(X_val)))

models_table = pd.DataFrame(
    model_rows,
    columns=["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"],
    dtype=object)
models_table.to_csv("outputs/part_c_models.csv", index=False)
print(models_table)

# Task 3: which columns carry signal 
from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import NOISE_COLUMNS

# forest importance on all 17 columns, training rows only
rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train, y_train)

# F-score for every column, training rows only
selector_all = SelectKBest(f_classif, k="all")
selector_all.fit(X_train, y_train)

scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector_all.scores_,
    "rf_importance": rf_all.feature_importances_,
})
scores["f_rank"] = scores["f_score"].rank(ascending=False, method="first").astype(int)
scores["is_noise"] = scores["feature"].isin(NOISE_COLUMNS)
scores = scores.sort_values("f_rank")[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
scores[["f_score", "rf_importance"]] = scores[["f_score", "rf_importance"]].round(4)
scores.to_csv("outputs/part_c_feature_scores.csv", index=False)
print(scores)
print("noise share of forest importance:", round(scores.loc[scores["is_noise"], "rf_importance"].sum(), 4))

# validation F1 when keeping only the top k columns
selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
    pipe.fit(X_train, y_train)
    selectk_rows.append([k, round(f1_score(y_val, pipe.predict(X_val)), 4)])

selectk_table = pd.DataFrame(selectk_rows, columns=["k", "val_f1"])
selectk_table.to_csv("outputs/part_c_selectk.csv", index=False)
print(selectk_table)

# Task 4: tuning on the top 7 columns 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# fit the selector on TRAINING rows only, then transform train and validation
selector7 = SelectKBest(f_classif, k=7)
X_train7 = selector7.fit_transform(X_train, y_train)
X_val7 = selector7.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train7, y_train)

random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid,
                                   n_iter=9, cv=5, random_state=42)
random_search.fit(X_train7, y_train)

# the untuned forest on the same 7 columns
default_forest = RandomForestClassifier(n_estimators=100, random_state=42)
default_forest.fit(X_train7, y_train)

tuning_rows = []
for name, search in [("grid_search", grid_search), ("randomized_search", random_search)]:
    candidates = len(search.cv_results_["params"])
    folds = 5
    val_f1 = f1_score(y_val, search.best_estimator_.predict(X_val7))
    tuning_rows.append([name, candidates, folds, candidates * folds,
                        str(search.best_params_),
                        round(search.best_score_, 4), round(val_f1, 4)])

default_f1 = f1_score(y_val, default_forest.predict(X_val7))
tuning_rows.append(["default_forest_same_7_features", 1, 0, 1,
                    "n_estimators=100, max_depth=None, min_samples_leaf=1",
                    "", round(default_f1, 4)])

tuning_table = pd.DataFrame(
    tuning_rows,
    columns=["search", "candidates", "cv_folds", "total_fits",
             "best_params", "best_cv_accuracy", "val_f1"],
    dtype=object)
tuning_table.to_csv("outputs/part_c_tuning.csv", index=False)
print(tuning_table.to_string())

# Task 5: choose the final configuration 
configs = {
    "default_forest_same_7_features": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    "grid_search": grid_search.best_params_,
    "randomized_search": random_search.best_params_,
}
val_f1_by_name = {row[0]: row[6] for row in tuning_rows}

chosen_from = None
best_val_f1 = -1
# order = tie-break priority; ">" means an earlier name keeps the win on a tie
for name in ["default_forest_same_7_features", "grid_search", "randomized_search"]:
    if val_f1_by_name[name] > best_val_f1:
        best_val_f1 = val_f1_by_name[name]
        chosen_from = name
chosen_params = configs[chosen_from]
print("chosen:", chosen_from, chosen_params)

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params)),
])
final_model.fit(X_train, y_train)          # training rows only

#  Task 6: alert threshold 
val_proba = final_model.predict_proba(X_val)[:, 1]     # probability of "delayed"
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

threshold_rows = []
for t in thresholds:
    pred = (val_proba >= t).astype(int)
    threshold_rows.append([t] + evaluate(y_val, pred))

threshold_table = pd.DataFrame(
    threshold_rows,
    columns=["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"],
    dtype=object)
threshold_table.to_csv("outputs/part_c_thresholds.csv", index=False)
print(threshold_table.to_string())

# operations rule: highest threshold whose validation recall is at least 0.80
chosen_threshold = 0.1
for row in threshold_rows:
    if row[7] >= 0.80:          # row[7] is recall
        chosen_threshold = row[0]
print("chosen threshold:", chosen_threshold)

# Task 7: one test evaluation
test_proba = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= chosen_threshold).astype(int)

final_test = pd.DataFrame(
    [[chosen_threshold] + evaluate(y_test, test_pred)],
    columns=["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"],
    dtype=object)
final_test.to_csv("outputs/part_c_final_test.csv", index=False)
print(final_test.to_string())

# one-row config file, read later by package_model.py
max_depth_value = "unlimited" if chosen_params["max_depth"] is None else chosen_params["max_depth"]
final_config = pd.DataFrame(
    [[7, chosen_params["n_estimators"], max_depth_value,
      chosen_params["min_samples_leaf"], chosen_threshold, chosen_from]],
    columns=["selector_k", "n_estimators", "max_depth", "min_samples_leaf",
             "threshold", "chosen_from"],
    dtype=object)
final_config.to_csv("outputs/part_c_final_config.csv", index=False)
print(final_config.to_string())