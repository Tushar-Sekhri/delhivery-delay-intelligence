"""Part D: hub profile, K-Means segments, separation evidence, tier check."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# Task 1: hub profile ,,groupby returns hubs in ascending hub_id order
profile = data.groupby("hub_id").agg(
    shipments=("hub_id", "size"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
).reset_index()
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)
print(profile.round(4).head(3))

# Task 2: scale the four profile columns, then cluster
cluster_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
X_scaled = StandardScaler().fit_transform(profile[cluster_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)


# Task 3a: across-to-within distance ratio for K = 2, 3, 4, 5
def across_over_within(X, lab):
    across, within = [], []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):          # every unordered pair once (276 pairs)
            dist = np.sqrt(((X[i] - X[j]) ** 2).sum())
            if lab[i] == lab[j]:
                within.append(dist)
            else:
                across.append(dist)
    return np.mean(across) / np.mean(within)


sep_rows = []
for k in [2, 3, 4, 5]:
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    sizes = sorted(np.bincount(lab), reverse=True)       # biggest cluster first
    sizes_text = "-".join(str(s) for s in sizes)
    sep_rows.append([k, round(across_over_within(X_scaled, lab), 4), sizes_text])

sep_table = pd.DataFrame(sep_rows, columns=["K", "across_over_within", "cluster_sizes"])
sep_table.to_csv("outputs/part_d_separation.csv", index=False)
print(sep_table)

# Task 3b: name the three clusters by their mean delay rate (lowest -> highest)
profile["cluster"] = labels
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
names = {cluster: name for cluster, name in zip(order, ["Standard", "Watch", "Escalate"])}
profile["segment"] = profile["cluster"].map(names)

# hub_segments.csv
hub_segments = profile[["hub_id", "segment"]].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
hub_segments.to_csv("outputs/hub_segments.csv", index=False)

# segment_profile.csv: means of the hub-level values
seg_profile = profile.groupby("segment").agg(
    hubs=("hub_id", "size"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean"),
).reset_index()
seg_profile["segment"] = pd.Categorical(
    seg_profile["segment"], ["Standard", "Watch", "Escalate"], ordered=True)
seg_profile = seg_profile.sort_values("segment")
seg_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)
print(seg_profile.round(4))

# tier_vs_segment_mismatch.csv: Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
tier_to_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
implied = hub_segments["hub_tier"].map(tier_to_segment)
mismatch = hub_segments[hub_segments["segment"] != implied]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print(mismatch)