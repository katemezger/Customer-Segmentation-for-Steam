# Steam Player Segmentation Engine: End-to-End Production ML Pipeline

A data-science-focused engineering project that moves beyond standard notebooks to build a live, production-ready inference system. This project processes highly skewed, sparse user interaction data from 200,000 Steam logs, engineers behavioral metrics, scales them via mathematical transformations, reduces dimensionality, and deploys the resulting cluster logic behind an asynchronous API microservice.

---

## Core Engineering Highlights
* **Production-Grade Architecture:** Wrapped all data wrangling, preprocessing, and scaling into an object-oriented Python pipeline class (`pipeline.py`) to eliminate data leakage between training and inference phases.
* **Mathematical Data Stabilization:** Handled heavy gaming-hour power-law skews using a `Yeo-Johnson PowerTransformer` instead of simple standard scaling, maximizing cluster density and algorithm stability.
* **Dimensionality Reduction:** Managed high-dimensional feature sparsity (tracking individual top-tier video game titles) by leveraging **PCA** to capture 85% of variance across dense components.
* **Low-Latency Deployment:** Serialized the optimized K-Means artifacts and embedded them inside a **FastAPI microservice** utilizing **Pydantic** for rigid input data validation, delivering sub-15ms cluster predictions.
* **Business Intelligence Mapping:** Translated raw mathematical cluster spaces into 4 distinct, actionable commercial personas (*The Whale, The Hoarder, The Specialist, The Casual Explorer*) paired with automated targeted marketing actions.

---

## System Architecture Diagram

┌──────────────────────────────────────────────────────────────┐
│ PRODUCTION ARCHITECTURE │
└──────────────────────────────────────────────────────────────┘
┌───────────────┐ ┌─────────────────┐ ┌───────────────┐
│ Raw Logs │ ───> │ Preprocess │ ───> │ PCA + │
│ (CSV Data) │ │ & PowerTransform│ │ K-Means │
└───────────────┘ └─────────────────┘ └───────────────┘
│
▼
┌───────────────┐ ┌─────────────────┐ ┌───────────────┐
│ FastAPI Endpt│ <─── │ Inference │ <─── │ Serialized │
│ (/predict) │ │ Engine (json) │ │ Artifacts │
└───────────────┘ └─────────────────┘ └───────────────┘

---

## Discovered Customer Personas

Based on statistical cluster profiling of the median usage hours and acquisition metrics, the engine segments users into four clear business profiles:

* **The Whales (Cluster 0):** Top 5% in both total playtime and games purchased. Highly active across multiple competitive multiplayer games. *Strategy: Target with high-ticket battle passes and expansion packs.*
* **The Digital Hoarders (Cluster 1):** High game purchase count, but an extremely low completion/play ratio. Many games show 0 hours of playtime. *Strategy: Trigger notifications during major platform seasonal sales.*
* **The Hyper-Focused Specialists (Cluster 2):** Low purchase count (under 5 games total), but hundreds of hours logged into a single title. *Strategy: Serve customized microtransactions or DLC specifically for their anchored game.*
* **The Casual Explorers (Cluster 3):** Low playtime and minimal financial investment. Playtime is scattered thinly across indie titles. *Strategy: Target with free-to-play conversions or high-discount introductory bundles.*

---

## Installation & Setup

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com
cd steam-player-segmentation
pip install -r requirements.txt
```

*Ensure your `requirements.txt` contains at least: `numpy`, `pandas`, `scikit-learn`, `joblib`, `fastapi`, `pydantic`, `uvicorn`.*

### 2. Prepare the Data & Train the Pipeline
Download the dataset from Kaggle or your source, place it in the root directory as `steam-200k.csv`, and run your training script to generate the serialized pipeline object:
```bash
python -c "from pipeline import SteamSegmentationPipeline; import pandas as pd; df=pd.read_csv('steam-200k.csv', header=None); pipe=SteamSegmentationPipeline().fit(df); pipe.save_pipeline()"
```
This generates the native production artifact file: `steam_pipeline.pkl`.

### 3. Launch the Microservice
Start the live inference engine locally using `uvicorn`:
```bash
uvicorn app:app --reload --port 8000
```

---

## API Endpoints & Usage Example

### Health Check
* **GET** `/`
* **Response:** `{"status": "healthy", "model_loaded": true}`

### Real-Time Cluster Inference
* **POST** `/predict`
* **Payload (JSON):**
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

* **Response (JSON):**
```json
{
  "assigned_cluster": 1,
  "assigned_persona": "The Digital Hoarder",
  "marketing_action_item": "Target with massive deep-cut sales on unplayed wishlisted titles."
}
```

------------------------------
