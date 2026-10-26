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