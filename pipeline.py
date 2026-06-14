import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

TOP_GAMES_COUNT = 20      # sparse game-specific features to include per user
N_CLUSTERS = 4
PCA_VARIANCE = 0.85       # keep components until 85% of variance is explained
PIPELINE_PATH = "steam_pipeline.pkl"

# persona IDs are canonical (0–3), not raw KMeans labels — see _assign_personas
PERSONAS = {
    0: {
        "name": "The Whale",
        "marketing_action": "Target with high-ticket battle passes and expansion packs.",
    },
    1: {
        "name": "The Digital Hoarder",
        "marketing_action": "Trigger notifications during major platform seasonal sales.",
    },
    2: {
        "name": "The Hyper-Focused Specialist",
        "marketing_action": "Serve customized microtransactions or DLC for their anchored game.",
    },
    3: {
        "name": "The Casual Explorer",
        "marketing_action": "Target with free-to-play conversions or high-discount introductory bundles.",
    },
}


class SteamSegmentationPipeline:
    """End-to-end pipeline: raw Steam logs → PowerTransform → PCA → KMeans → persona."""

    def __init__(
        self,
        n_top_games: int = TOP_GAMES_COUNT,
        n_clusters: int = N_CLUSTERS,
        pca_variance: float = PCA_VARIANCE,
    ):
        self.n_top_games = n_top_games
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance

        # all _  attributes are unset until fit() is called
        self.top_games_: list[str] | None = None
        self.feature_columns_: list[str] | None = None
        self.transformer_: PowerTransformer | None = None
        self.pca_: PCA | None = None
        self.kmeans_: KMeans | None = None
        self.cluster_persona_map_: dict[int, int] | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_raw_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        # raw CSV has no header — assign names and drop the unused trailing column
        df = df.copy()
        df.columns = ["user_id", "game_name", "behavior", "value", "extra"]
        df = df[["user_id", "game_name", "behavior", "value"]]
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
        # each row is either a purchase event or a play session — split them first
        play = df[df["behavior"] == "play"].copy()
        purchase = df[df["behavior"] == "purchase"].copy()

        # roll up play rows into one summary row per user
        play_agg = (
            play.groupby("user_id")["value"]
            .agg(
                total_play_hours="sum",
                average_play_hours="mean",
                max_play_hours="max",
                unique_games_played="count",
            )
            .reset_index()
        )

        # count distinct titles purchased per user
        purchase_agg = (
            purchase.groupby("user_id")["game_name"]
            .nunique()
            .reset_index()
            .rename(columns={"game_name": "games_purchased"})
        )

        users = play_agg.merge(purchase_agg, on="user_id", how="left")
        users["games_purchased"] = users["games_purchased"].fillna(0)
        # replace(0,1) avoids divide-by-zero for users with no purchases logged
        users["play_to_purchase_ratio"] = users["unique_games_played"] / users[
            "games_purchased"
        ].replace(0, 1)

        # lock in top games on first fit; reuse the same list at inference time
        # so training and inference features always align
        if self.top_games_ is None:
            game_totals = play.groupby("game_name")["value"].sum().nlargest(self.n_top_games)
            self.top_games_ = game_totals.index.tolist()

        # pivot: one column per top game, value = hours that user spent in it
        game_pivot = play[play["game_name"].isin(self.top_games_)].pivot_table(
            index="user_id",
            columns="game_name",
            values="value",
            aggfunc="sum",
            fill_value=0,
        )
        # a game may have zero players in a subset — ensure every column exists
        for game in self.top_games_:
            if game not in game_pivot.columns:
                game_pivot[game] = 0.0
        game_pivot = game_pivot[self.top_games_]

        features = users.set_index("user_id").join(game_pivot, how="left")
        features[self.top_games_] = features[self.top_games_].fillna(0)

        user_ids = features.index
        return features.reset_index(drop=True), user_ids

    def _assign_personas(self, X_raw: np.ndarray, labels: np.ndarray) -> dict[int, int]:
        # KMeans cluster IDs are arbitrary — map them to named personas via centroid profiling
        # so the same persona always gets the same ID regardless of random seed
        col = {name: i for i, name in enumerate(self.feature_columns_)}
        profiles: dict[int, dict] = {}
        for c in range(self.n_clusters):
            subset = X_raw[labels == c]
            profiles[c] = {
                "total_play_hours": np.median(subset[:, col["total_play_hours"]]),
                "games_purchased": np.median(subset[:, col["games_purchased"]]),
                "max_play_hours": np.median(subset[:, col["max_play_hours"]]),
                "play_to_purchase_ratio": np.median(subset[:, col["play_to_purchase_ratio"]]),
            }

        remaining = set(range(self.n_clusters))
        mapping: dict[int, int] = {}

        # greedily assign each persona in priority order using distinguishing metrics
        whale = max(
            remaining,
            key=lambda c: profiles[c]["total_play_hours"] * profiles[c]["games_purchased"],
        )
        mapping[whale] = 0
        remaining.discard(whale)

        hoarder = max(remaining, key=lambda c: profiles[c]["games_purchased"])
        mapping[hoarder] = 1
        remaining.discard(hoarder)

        specialist = max(remaining, key=lambda c: profiles[c]["max_play_hours"])
        mapping[specialist] = 2
        remaining.discard(specialist)

        mapping[remaining.pop()] = 3  # casual explorer — whatever's left

        return mapping

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "SteamSegmentationPipeline":
        raw = self._parse_raw_logs(df)
        features, _ = self._engineer_features(raw)

        self.feature_columns_ = list(features.columns)
        X = features.values.astype(float)

        # Yeo-Johnson handles the heavy power-law skew in gaming hours
        # better than standard scaling, which would be dominated by outliers
        self.transformer_ = PowerTransformer(method="yeo-johnson", standardize=True)
        X_scaled = self.transformer_.fit_transform(X)

        # compress 26 features down to ~17 components, removing correlated noise
        self.pca_ = PCA(n_components=self.pca_variance, svd_solver="full")
        X_pca = self.pca_.fit_transform(X_scaled)

        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
        labels = self.kmeans_.fit_predict(X_pca)

        # profile clusters in original feature space (pre-transform) for interpretability
        self.cluster_persona_map_ = self._assign_personas(X, labels)

        self._is_fitted = True
        n_components = self.pca_.n_components_
        explained = self.pca_.explained_variance_ratio_.sum()
        print(
            f"Fit complete — {len(features):,} users | "
            f"PCA: {n_components} components ({explained:.1%} variance) | "
            f"Cluster map: {self.cluster_persona_map_}"
        )
        return self

    def _build_feature_vector(self, payload: dict) -> np.ndarray:
        # reconstruct a row in the exact column order used during training
        if self.feature_columns_ is None or self.top_games_ is None:
            raise RuntimeError("Pipeline not fitted.")
        game_hours: dict = payload.get("game_specific_hours") or {}
        row = {
            "total_play_hours": float(payload.get("total_play_hours", 0)),
            "average_play_hours": float(payload.get("average_play_hours", 0)),
            "max_play_hours": float(payload.get("max_play_hours", 0)),
            "unique_games_played": float(payload.get("unique_games_played", 0)),
            "games_purchased": float(payload.get("games_purchased", 0)),
            "play_to_purchase_ratio": float(payload.get("play_to_purchase_ratio", 0)),
        }
        # zero-fill any top game the user hasn't played
        for game in self.top_games_:
            row[game] = float(game_hours.get(game, 0))
        return np.array([[row.get(col, 0.0) for col in self.feature_columns_]])

    def predict(self, payload: dict) -> dict:
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit() or load_pipeline() first.")
        X = self._build_feature_vector(payload)
        # run through the same transform chain used during training
        X_scaled = self.transformer_.transform(X)
        X_pca = self.pca_.transform(X_scaled)
        raw_cluster = int(self.kmeans_.predict(X_pca)[0])
        # translate raw cluster ID → canonical persona ID → persona metadata
        persona_id = self.cluster_persona_map_[raw_cluster]
        persona = PERSONAS[persona_id]
        return {
            "assigned_cluster": persona_id,
            "assigned_persona": persona["name"],
            "marketing_action_item": persona["marketing_action"],
        }

    def save_pipeline(self, path: str = PIPELINE_PATH) -> None:
        # joblib serializes the entire fitted object including all sklearn estimators
        joblib.dump(self, path)
        print(f"Pipeline saved → {path}")

    @classmethod
    def load_pipeline(cls, path: str = PIPELINE_PATH) -> "SteamSegmentationPipeline":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj
