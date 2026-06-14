"""
Deep dive: Football Manager franchise behavior across personas.
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

play_df = df[df["behavior"] == "play"][["user_id", "game_name", "value"]].copy()
play_df = play_df.merge(features[["user_id", "persona"]], on="user_id", how="left")

fm_df = play_df[play_df["game_name"].str.contains("Football Manager", na=False)].copy()
fm_users = fm_df.merge(features[["user_id", "persona_id", "total_play_hours", "games_purchased"]], on="user_id", how="left")
fm_users["persona"] = fm_users["persona_id"].map({p: PERSONAS[p]["name"] for p in PERSONAS})

print("=" * 60)
print("FM FRANCHISE: PERSONA DISTRIBUTION")
print("=" * 60)
fm_unique = fm_users.drop_duplicates("user_id")
dist = fm_unique["persona"].value_counts()
total_fm = len(fm_unique)
for persona, count in dist.items():
    print(f"  {persona:<35} {count:>4} users ({count/total_fm*100:.1f}%)")

print(f"\n  Total FM players: {total_fm}")
print(f"  FM players as % of all users: {total_fm / len(features) * 100:.1f}%")

print("\n" + "=" * 60)
print("FM HOURS BY PERSONA (median per user)")
print("=" * 60)
fm_hours = fm_users.groupby(["user_id", "persona"])["value"].sum().reset_index()
summary = fm_hours.groupby("persona")["value"].agg(
    median_fm_hours="median",
    mean_fm_hours="mean",
    max_fm_hours="max",
    player_count="count"
).round(1)
print(summary.to_string())

print("\n" + "=" * 60)
print("DO FM PLAYERS COLLECT MULTIPLE EDITIONS?")
print("=" * 60)
editions_per_user = fm_users.groupby(["user_id", "persona"])["game_name"].nunique().reset_index()
editions_per_user.columns = ["user_id", "persona", "editions_owned"]
edition_summary = editions_per_user.groupby("persona")["editions_owned"].agg(
    median="median", mean="mean", max="max"
).round(2)
print(edition_summary.to_string())

print("\n" + "=" * 60)
print("FM vs NON-FM HOURS: SPECIALIST CLUSTER")
print("=" * 60)
specialist_users = features[features["persona"] == "The Hyper-Focused Specialist"]["user_id"]
specialist_play = play_df[play_df["user_id"].isin(specialist_users)]
is_fm = specialist_play["game_name"].str.contains("Football Manager", na=False)
print(f"  Specialist users who play FM:     {specialist_play[is_fm]['user_id'].nunique()}")
print(f"  Specialist users who don't:       {specialist_play[~is_fm]['user_id'].nunique()}")
fm_hrs = specialist_play[is_fm]["value"].sum()
non_fm_hrs = specialist_play[~is_fm]["value"].sum()
total_hrs = fm_hrs + non_fm_hrs
print(f"  % of all Specialist hours from FM: {fm_hrs/total_hrs*100:.1f}%")

print("\n" + "=" * 60)
print("TOP NON-FM GAMES FOR SPECIALISTS")
print("=" * 60)
top_non_fm = (
    specialist_play[~is_fm]
    .groupby("game_name")["value"]
    .agg(median_hours="median", player_count="count")
    .query("player_count >= 3")
    .sort_values("median_hours", ascending=False)
    .head(10)
)
for game, row in top_non_fm.iterrows():
    print(f"  {game:<45} {row['median_hours']:>7.1f}h  ({int(row['player_count'])} players)")
