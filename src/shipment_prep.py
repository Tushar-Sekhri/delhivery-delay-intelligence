"""Shared constants and preparation functions for the Delay Intelligence project."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw, medians=None):
    """Clean a raw shipments table. Returns (clean_df, medians)."""
    df = raw.copy()

    # 1. Drop exact duplicate rows (the export repeated 24 rows)
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Fix service_type spellings: "  express " / "EXPRESS" -> "Express"
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 3. Missing payment mode becomes an explicit "Unknown" category
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. "₹1,290" (text) -> 1290.0 (number), in a new column; drop the text one
    df["declared_value_inr"] = (
        df["declared_value"]
        .str.replace("\u20b9", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df = df.drop(columns=["declared_value"])

    # 5. timestamp -> real datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # 6. Weights above 100 kg are scanner faults (max real parcel is 40 kg) -> missing
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. Fill gaps with medians. Compute them only if not supplied, so that new parcels at prediction time reuse the historical medians.
    if medians is None:
        medians = {c: float(df[c].median()) for c in MEDIAN_COLUMNS}
    for c in MEDIAN_COLUMNS:
        df[c] = df[c].fillna(medians[c])

    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and create model features. Never touches delivery_hours."""
    # Left merge keeps every parcel even if a hub were missing from hubs.csv
    df = clean.merge(hubs, on="hub_id", how="left")

    # Promise in hours, looked up from the service type
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)

    # 1 for cash-on-delivery, 0 for Prepaid and Unknown
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    # pandas: Monday = 0 ... Saturday = 5, Sunday = 6
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)

    # Hour of day (6 to 21 in this data)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)

    # Declared value is right-skewed, so use its natural log
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df