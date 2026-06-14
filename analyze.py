"""
Cluster analysis: assigns all users to personas, then profiles each cluster
by behavioral stats and top games.
"""

import pandas as pd
import numpy as np
from pipeline import SteamSegmentationPipeline, PERSONAS

pipe = SteamSegmentationPipeline.load_pipeline()
df_raw = pd.read_csv("archive/steam-200k.csv", header=None)
df = pipe._parse_raw_logs(df_raw)
features, user_ids = pipe._engineer_features(df)

X = features.values.astype(float)
X_scaled = pipe.transformer_.transform(X)
X_pca = pipe.pca_.transform(X_scaled)
raw_labels = pipe.kmeans_.predict(X_pca)
persona_labels = np.array([pipe.cluster_persona_map_[c] for c in raw_labels])

features["user_id"] = user_ids.values
features["persona_id"] = persona_labels
features["persona"] = [PERSONAS[p]["name"] for p in persona_labels]

print("=" * 60)
print("CLUSTER SIZE BREAKDOWN")
print("=" * 60)
counts = features["persona"].value_counts()
for name, count in counts.items():
    pct = count / len(features) * 100
    print(f"  {name:<30} {count:>5,} users  ({pct:.1f}%)")

behavioral_cols = [
    "total_play_hours", "average_play_hours", "max_play_hours",
    "unique_games_played", "games_purchased", "play_to_purchase_ratio",
]

print("\n" + "=" * 60)
print("BEHAVIORAL MEDIANS PER PERSONA")
print("=" * 60)
medians = features.groupby("persona")[behavioral_cols].median().round(1)
print(medians.to_string())

# Game associations per persona
play_df = df[df["behavior"] == "play"][["user_id", "game_name", "value"]].copy()
play_df = play_df.merge(features[["user_id", "persona"]], on="user_id", how="left")

print("\n" + "=" * 60)
print("TOP 8 GAMES PER PERSONA (by median hours among players)")
print("=" * 60)
for pid in range(4):
    persona_name = PERSONAS[pid]["name"]
    subset = play_df[play_df["persona"] == persona_name]
    top = (
        subset.groupby("game_name")["value"]
        .agg(median_hours="median", player_count="count")
        .query("player_count >= 10")
        .sort_values("median_hours", ascending=False)
        .head(8)
    )
    print(f"\n  {persona_name}")
    for game, row in top.iterrows():
        print(f"    {game:<40} {row['median_hours']:>7.1f}h  ({int(row['player_count'])} players)")
