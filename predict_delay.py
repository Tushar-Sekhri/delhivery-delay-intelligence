"""Score new parcels and write an action list."""
import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

# load the inputs and the saved artefacts
new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
segments = pd.read_csv("outputs/hub_segments.csv")

model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = {row["column"]: float(row["median"]) for _, row in impute.iterrows()}

# same preparation as in training, using the stored medians
clean, _ = clean_shipments(new, medians)
features = build_features(clean, hubs)

# probabilities, flag, segment, action
features["delay_probability"] = model.predict_proba(features[feature_order])[:, 1].round(4)
features["flagged"] = (features["delay_probability"] >= threshold).astype(int)
features = features.merge(segments[["hub_id", "segment"]], on="hub_id", how="left")

features["action"] = "No action"
features.loc[features["flagged"] == 1, "action"] = "Proactive customer call"
features.loc[(features["flagged"] == 1) & (features["segment"] == "Escalate"),
             "action"] = "Escalate to hub manager"

out = features[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)

print(len(out), "parcels scored,", int(out["flagged"].sum()), "flagged,",
      int((out["action"] == "Escalate to hub manager").sum()), "escalations")