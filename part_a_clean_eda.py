"""Part A: clean the raw export, build the modelling table, write evidence files."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# "Before" numbers are measured after removing duplicates, before any filling
dedup = raw.drop_duplicates()

clean, medians = clean_shipments(raw)
feat = build_features(clean, hubs)
feat["is_delayed"] = (feat["delivery_hours"] > feat["promised_hours"]).astype(int)

feat.to_csv("data/modelling_table.csv", index=False)

# checks file 
hours = feat["delivery_hours"]
q1, q3 = hours.quantile(0.25), hours.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr   

value = feat["declared_value_inr"]               # median-filled column
log_value = np.log(value)

rows = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", int(dedup["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(dedup["parcel_weight_kg"].isna().sum())),
    ("sentinel_weight_rows", int((dedup["parcel_weight_kg"] > 100).sum())),
    ("missing_declared_value_before", int(dedup["declared_value"].isna().sum())),
    ("missing_payment_mode_before", int(dedup["payment_mode"].isna().sum())),
    ("service_type_spellings_before", int(dedup["service_type"].nunique())),
    ("service_type_spellings_after", int(clean["service_type"].nunique())),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("declared_value_mean_inr", value.mean()),
    ("declared_value_median_inr", value.median()),
    ("log_declared_value_mean", log_value.mean()),
    ("log_declared_value_median", log_value.median()),
    ("rows_after_hub_merge", len(feat)),
    ("nulls_in_features_and_target",
     int(feat[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(feat["is_delayed"].sum())),
    ("delay_rate", feat["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower),
    ("hours_upper_bound", upper),
    ("hours_outlier_count", int(((hours < lower) | (hours > upper)).sum())),
]

# round decimals to 4 places, leave whole numbers alone
rounded_rows = []
for name, value in rows:
    if isinstance(value, float):
        value = round(value, 4)
    rounded_rows.append((name, value))

checks = pd.DataFrame(rounded_rows, columns=["metric", "value"], dtype=object)
checks.to_csv("outputs/part_a_checks.csv", index=False)

# delay evidence by group
feat["distance_band"] = pd.cut(
    feat["distance_km"], bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"], right=True)

parts = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = feat.groupby(dim, observed=True).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index().rename(columns={dim: "group"})
    g.insert(0, "dimension", dim)
    g["group"] = g["group"].astype(str)
    parts.append(g)
by_group = pd.concat(parts, ignore_index=True)
by_group[["delay_rate", "mean_hours"]] = by_group[["delay_rate", "mean_hours"]].round(4)
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# correlation matrix
corr = feat.select_dtypes(include="number").corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv")   

print("modelling table:", feat.shape)
print(by_group)
print("corr(hours, promised):", corr.loc["delivery_hours", "promised_hours"])
print("corr(hours, distance):", corr.loc["delivery_hours", "distance_km"])
print("corr(delayed, distance):", corr.loc["is_delayed", "distance_km"])