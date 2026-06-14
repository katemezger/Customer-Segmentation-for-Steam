# Steam Player Segmentation Engine

An end-to-end machine learning pipeline that processes raw Steam interaction logs, engineers behavioral features, and deploys a live clustering model behind a FastAPI microservice. The system automatically segments users into commercial personas and returns real-time predictions via a REST API.

---

## The Problem

Steam has millions of users who all behave differently. Some spend thousands of hours in one game. Some buy 50 titles and barely touch any of them. Some are casual browsers with minimal investment. A single marketing strategy across all of them is wasted spend.

This project lets the data find the natural groupings. No manual labeling — the algorithm discovers the structure on its own, and the personas emerge from the math.

---

## System Architecture

```
Raw Logs (CSV)
      |
      v
Feature Engineering          -- 26 behavioral metrics per user
      |
      v
Yeo-Johnson PowerTransform   -- stabilizes heavy power-law skew in gaming hours
      |
      v
PCA                          -- compresses 26 features to 17 components (87.3% variance)
      |
      v
K-Means (k=4)                -- finds natural user clusters
      |
      v
Persona Assignment           -- maps raw cluster IDs to named business personas
      |
      v
Serialized Artifact          -- steam_pipeline.pkl
      |
      v
FastAPI /predict             -- sub-15ms inference, Pydantic input validation
```

---

## Engineering Decisions

**Yeo-Johnson over standard scaling** — gaming hour data follows a power-law distribution. A handful of users have 5,000+ hours while the median is under 10. Standard scaling leaves that skew intact and K-Means clusters around outliers instead of behavioral patterns. Yeo-Johnson mathematically corrects for this.

**OOP pipeline class** — all feature engineering, scaling, and transformation logic lives inside `SteamSegmentationPipeline`. The same object that fits on training data is serialized and loaded by the API, eliminating any risk of data leakage or preprocessing mismatches between training and inference.

**Persona auto-mapping** — K-Means returns arbitrary cluster IDs (0, 1, 2, 3) that can shuffle between runs. After fitting, the pipeline profiles each cluster's median stats and assigns personas using business rules (highest hours × purchases = Whale, etc.), making labels stable and interpretable.

---

## Discovered Personas

Derived from statistical profiling of 11,350 unique users across 200,000 interaction logs.

**The Whale** — top 8% by both total playtime and games purchased. Highly active across multiple titles including CS:GO, Arma 3, Skyrim, and GTA V. Median 998 hours, 44 games purchased.
Strategy: high-ticket battle passes and expansion packs.

**The Digital Hoarder** — high purchase count, very low play-to-purchase ratio. Buys frequently during sales; many titles show 0 hours. Median 141 hours across 5 games purchased. Concentrated in long-tail simulation titles.
Strategy: seasonal sale notifications targeting unplayed backlog.

**The Hyper-Focused Specialist** — fewer than 5 games purchased but median 1,203 hours logged. The dominant game for this cluster is Football Manager across multiple annual editions — 76% of all Specialist hours come from a single franchise. Non-FM players in this cluster gravitate toward Total War, Civilization, and Crusader Kings II. These are systems-mastery players, not variety seekers.
Strategy: franchise DLC and annual edition upgrades.

**The Casual Explorer** — low playtime, low spend, scattered across indie titles. Median 2.4 total hours. Represents 51% of the user base.
Strategy: free-to-play conversions and introductory discount bundles.

---

## Project Structure

```
pipeline.py          core ML class — feature engineering, transform, PCA, KMeans, persona mapping
train.py             CLI training script — fits the pipeline and writes steam_pipeline.pkl
app.py               FastAPI microservice — loads the artifact and serves /predict
index.html           static persona card display
requirements.txt     dependencies
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download the dataset**

Get `steam-200k.csv` from [Kaggle — Steam Video Games by Tamber](https://www.kaggle.com/datasets/tamber/steam-video-games) and place it in the project root (or `archive/` subfolder).

**3. Train the pipeline**
```bash
python train.py --data archive/steam-200k.csv
```

This produces `steam_pipeline.pkl`.

**4. Start the API**
```bash
uvicorn app:app --reload --port 8000
```

---

## API Reference

**GET /**
```json
{ "status": "healthy", "model_loaded": true }
```

**POST /predict**

Request:
```json
{
  "total_play_hours": 450.5,
  "average_play_hours": 12.2,
  "max_play_hours": 120.0,
  "unique_games_played": 14,
  "games_purchased": 45,
  "play_to_purchase_ratio": 0.31,
  "game_specific_hours": {
    "Dota 2": 120.0,
    "Team Fortress 2": 45.5
  }
}
```

Response:
```json
{
  "assigned_cluster": 1,
  "assigned_persona": "The Digital Hoarder",
  "marketing_action_item": "Trigger notifications during major platform seasonal sales."
}
```

Interactive docs available at `http://127.0.0.1:8000/docs` when the server is running.
