# Smadex Creative Copilot

**A Creative Intelligence prototype for mobile advertisers — built for the Smadex Creative Intelligence Challenge.**

The Creative Copilot helps advertising teams answer four hard operational questions:

- **Which creatives are working** — and why?
- **Which are wearing out** — and when exactly did fatigue kick in?
- **What should I test next** — given what I am already running?
- **What would a brand-new creative score** — before I spend a dollar producing it?

It answers all four through a single self-contained HTML dashboard backed by a scikit-learn + XGBoost ML pipeline.

---

## Features

| Tab | What it does |
|-----|-------------|
| 🔍 Performance Explorer | Browse, filter and rank all 1 080 creatives by any KPI. Paginated grid view or bar chart. |
| 📉 Fatigue Detection | Time-series CTR chart for any creative. Normalized view (% of launch baseline) with per-status benchmark lines. |
| 🎯 Recommendations | KPI-aware action cards: Scale / Monitor / Pause, ranked by performance score. |
| 💡 ML Insights | Random Forest feature importance (global + per KPI goal). Cross-validation R² and MAE. PCA variance breakdown. |
| 🔵 Clusters | K-Means cluster profiles (k=8). Cluster composition, dominant traits, average performance. |
| 🧮 Portfolio Optimizer | Gram-Schmidt orthogonal portfolio builder with a **strategy slider** (pure performance ↔ pure diversity) and KPI targeting. |
| 🔮 Predict Creative | Estimate the performance of a hypothetical new creative before producing it. |
| 📖 Help | Full metric glossary, tab guide, field to save a **Gemini** API key in-browser, and an embedded **Ask Gemini** chatbot. |

---

## Architecture

```
data/   (six CSVs + data_dictionary.csv + assets/)
      │
      ▼
src/ml_pipeline_sklearn.py       ← Python, run once from repo root
  ├─ pandas: data loading & joins
  ├─ PIL: image feature extraction from data/assets thumbnails
  ├─ sklearn: PCA · KMeans(k=8) · RandomForestRegressor
  ├─ xgboost: XGBRegressor (KPI-specific models)
  ├─ shap: TreeExplainer (per-creative feature attribution)
  └─ 5-fold cross-validation (R² ± std, MAE)
      │
      ▼
app_data.js (repo root)          ← embedded APP_DATA (creatives + timeseries)
ml_results.json (repo root)      ← feature importance · clusters · SHAP · model params
      │
      ▼
src/build_dashboard.py           ← patches both into src/dashboard_template.html
      │
      ▼
creative_copilot.html (root)     ← single self-contained file, open in any browser
  ├─ Chart.js 4: all charts
  ├─ Google Gemini API: live AI (key from Help tab → localStorage only; never in the file)
  └─ Gram-Schmidt orthogonalisation in JS (Portfolio Optimizer)
```

---

## Setup

### Prerequisites

- Python 3.9+
- pip

Run all commands below from the **repository root** (the folder that contains `data/`, `src/`, and `requirements.txt`).

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

```
pandas>=1.5
scikit-learn>=1.3
xgboost>=2.0
shap>=0.44
Pillow>=9.0
numpy>=1.24
```

> **Note:** `xgboost` and `shap` are optional. The pipeline falls back to `sklearn.ensemble.RandomForestRegressor` and simplified feature attribution if either is unavailable.

### 2. Run the ML pipeline

```bash
python src/ml_pipeline_sklearn.py
```

This takes roughly **2–5 minutes** (image feature extraction is the bottleneck). It produces:

- `app_data.js` — all creative and timeseries data (~3 MB)
- `ml_results.json` — ML outputs: feature importance, clusters, SHAP values, cross-validation metrics, model parameters (~500 KB)

### 3. Build the dashboard

```bash
python src/build_dashboard.py
```

This embeds `app_data.js` and `ml_results.json` into `src/dashboard_template.html` and writes `creative_copilot.html` at the repo root (~3 MB). **No API key is placed in the HTML.** The built file is listed in `.gitignore` because of its size; remove that line if you want to track it in Git.

### 4. Open the dashboard

```bash
# macOS
open creative_copilot.html

# Linux
xdg-open creative_copilot.html

# Windows
start creative_copilot.html
```

No server required — it runs entirely in the browser.

---

## Dashboard Tab Reference

### 🔍 Performance Explorer

Browse all 1 080 creatives with live filtering by advertiser, KPI goal, status, format, vertical, and theme. Switch between a paginated **card grid** (with thumbnail, status badge, KPI value, and an AI Explain button) and a **bar chart** of the top 20 by any metric.

**Controls:**
- Sort by: Performance Score, CTR, CVR, ROAS, IPM, Spend
- Filters: Status · KPI Goal · Format · Vertical · Theme
- View: Grid (24 per page) or Chart (top 20)

---

### 📉 Fatigue Detection

Pick any creative and see its daily CTR over time. Toggle between:

- **Absolute CTR** — raw click-through rate per day
- **Normalized (% of launch)** — CTR expressed as a fraction of the first-7-day average (= 100%). Makes fatigue curves comparable across creatives with different baseline CTRs.

Three benchmark lines show where each status group typically lands:
- Top Performer: retains ~33% of launch CTR
- Stable: retains ~23%
- Fatigued: retains ~16%

> **Important:** CTR naturally decreases for *all* digital ads (banner blindness, audience saturation). The distinction between "good" and "fatigued" is not *whether* CTR drops, but *how fast* and *how far*.

---

### 🎯 Recommendations

For each creative, the engine assigns one of three actions:

| Action | Criteria |
|--------|----------|
| **Scale** | `top_performer` status |
| **Monitor** | `stable` status |
| **Pause** | `fatigued` or `underperformer` status |

The KPI metric shown in each card matches the campaign's `kpi_goal`: CPA → CVR, ROAS → ROAS, IPM → IPM, CTR → CTR.

---

### 💡 ML Insights

Three panels:

1. **Feature Importance** — select a model (global or KPI-specific) to see which creative attributes the model found most predictive. Image features (extracted from PNG thumbnails via PIL) are prefixed with `img_`.
2. **Cross-Validation** — 5-fold CV results: R² ± std and Mean Absolute Error for an honest out-of-sample accuracy estimate.
3. **PCA Variance** — cumulative explained variance across 20 principal components.

---

### 🔵 Clusters

K-Means (k=8) groups the 1 080 creatives into 8 behavioural archetypes based on performance metrics, creative attributes, and image features. Each cluster card shows dominant format/theme/tone/KPI, average performance score, and status distribution.

---

### 🧮 Portfolio Optimizer

**Step 1 — Build your portfolio.** Search and add the creatives you are already running.

**Step 2 — Choose your strategy.** A slider controls the blend:
- `α = 0` → pure performance: recommend the best-performing remaining creatives
- `α = 1` → pure orthogonality: recommend creatives that explore the most *unexplored creative territory* relative to your current portfolio
- Any point in between blends both signals proportionally

**KPI targeting** — check one or more KPI types to give a 2× score boost to candidates whose campaign matches your goals.

**How orthogonality works:** Each creative is represented by its 10-dimensional PCA feature vector. The optimizer computes a Gram-Schmidt orthonormal basis for the portfolio's subspace, then scores each candidate by the fraction of its vector that lies *outside* that subspace. Score = 100% means completely unexplored creative territory.

---

### 🔮 Predict Creative

Enter the attributes of a hypothetical creative (format, vertical, theme, tone, OS, KPI, duration, text density, and binary flags) and get an instant performance estimate.

1. **Cluster matching** — finds the K-Means cluster whose profile best matches your inputs.
2. **Feature adjustments** — applies boosts/penalties from the RF/XGBoost feature importance for your specific KPI.
3. **Prediction band** — maps the score to Likely Top Performer / Stable / Below Average / High-Risk with a recommended action.
4. **Benchmarks** — shows the top 3 existing creatives in the matched cluster+KPI combination.

After running `python src/ml_pipeline_sklearn.py` locally (which embeds PCA components and scaler parameters), the predictor upgrades to full PCA-based OLS inference.

---

### 📖 Help

Static glossary of all metrics, a tab-by-tab usage guide, a **Gemini API key** field (saved in the browser only), and an embedded **Ask Gemini** chatbot. The chatbot can answer questions like *"What does a perf_score of 0.3 mean?"*, *"Why does CTR always drop over time?"*, or *"How do I interpret the orthogonality score?"*

---

### 🤖 AI Explain (available on every creative card)

Each creative card has an **🤖 Explain** button:

1. Add a [Gemini API key](https://aistudio.google.com/app/apikey) under **Help → Save key** (stored in `localStorage` in this browser only — it is **not** written into `creative_copilot.html` or the repository).
2. The dashboard structures a JSON payload: performance metrics, status, cluster profile, top KPI feature drivers, and auto-detected positive/negative signals.
3. A Gemini model (e.g. Flash) turns the payload into a plain-English 3–4 sentence explanation in marketing language.

The model is strictly constrained to explain using *only the provided data*, preventing hallucination.

---

## Metrics Glossary

### Performance Metrics

| Metric | Definition | Typical range |
|--------|-----------|---------------|
| **CTR** | Clicks ÷ Impressions. Fraction of users who tap the ad. | 0.8–2.5% |
| **CVR** | Conversions ÷ Clicks. Fraction of clickers who complete the goal action. | 10–40% |
| **ROAS** | Revenue ÷ Spend. Dollars of attributed revenue per dollar spent. | >1.0 = profitable |
| **IPM** | Conversions per 1 000 impressions. Direct install-campaign efficiency. | 3–15 |
| **perf_score** | Composite score (0–1) from CTR, CVR, ROAS, and IPM normalised within the dataset. | >0.65 = top performer |

### Fatigue & Decay Metrics

| Metric | Definition |
|--------|-----------|
| **first_7d_ctr** | Average CTR across the first 7 active days — the freshness baseline. |
| **last_7d_ctr** | Average CTR across the last 7 active days of data. |
| **ctr_decay_pct** | `(last_7d_ctr − first_7d_ctr) / first_7d_ctr`. Negative = CTR fell. −0.67 = 67% drop. |
| **cvr_decay_pct** | Same as above for CVR. |
| **fatigue_day** | Day number (since launch) when performance decay became material. |
| **CTR retention** | `last_7d_ctr / first_7d_ctr`. Top performers retain ~33%; fatigued creatives only ~16%. |

### Creative Attributes

| Attribute | Definition | Range |
|-----------|-----------|-------|
| **text_density** | Fraction of visual area occupied by text. | 0 (minimal) – 1 (text-heavy) |
| **readability_score** | Ease of reading the copy. | 0 – 1 (higher = easier) |
| **brand_visibility_score** | Prominence of logo/brand mark. | 0 – 1 |
| **clutter_score** | How visually busy/cluttered the layout is. | 0 (clean) – 1 (cluttered) |
| **novelty_score** | Estimated originality relative to other creatives. | 0 – 1 |
| **motion_score** | Intensity of movement/animation. | 0 (static) – 1 (high motion) |
| **duration_sec** | Video or interactive ad length in seconds; 0 for static formats. | 0 – 120 |
| **faces_count** | Number of people/faces visible. | Integer ≥ 0 |
| **has_gameplay** | 1 if the creative shows gameplay footage. | 0 / 1 |
| **has_ugc_style** | 1 if the creative mimics a UGC / creator-content layout. | 0 / 1 |
| **has_price** | 1 if a monetary price or offer is visible. | 0 / 1 |
| **has_discount_badge** | 1 if a sale or bonus badge is shown. | 0 / 1 |

### Status Labels

| Status | Meaning | Recommended action |
|--------|---------|-------------------|
| **top_performer** | High performance, CTR holds well over time. | Scale budget; create A/B variants. |
| **stable** | Solid but not exceptional; no strong fatigue. | Monitor; consider iterating. |
| **fatigued** | Clear performance decay past `fatigue_day`. | Pause or refresh; do not increase spend. |
| **underperformer** | Consistently weak primary KPI. | Reallocate budget; retire concept. |

### KPI Goals

| KPI | Campaign focus | Primary metric |
|-----|---------------|----------------|
| **CPA** | Cost-per-acquisition — minimise cost per conversion | CVR |
| **ROAS** | Return on ad spend — maximise revenue per dollar | ROAS |
| **IPM** | Installs per mille — maximise install volume | IPM |
| **CTR** | Click-through rate — maximise engagement | CTR |

### ML Concepts

| Term | Definition |
|------|-----------|
| **Feature Importance** | How much each input variable contributed to the model's predictive accuracy. Ranges 0–1; all features sum to 1. |
| **SHAP value** | Shapley Additive exPlanation. For a specific creative, the exact contribution of each feature to the difference between its predicted score and the global average. Positive = pushed score up; negative = pulled it down. |
| **PCA** | Principal Component Analysis. Reduces the 36-feature matrix to 20 orthogonal components capturing 93% of variance. |
| **K-Means cluster** | One of 8 behavioural archetypes. Creatives in the same cluster share similar format, theme, tone, and performance profile. |
| **R²** | Coefficient of determination. Fraction of variance explained by the model. 1.0 = perfect; 0 = no better than the mean. Values above 0.6 are good for this data. |
| **MAE** | Mean Absolute Error. Average absolute difference between predicted and actual values. |
| **Orthogonality score** | Fraction of a creative's 10-PC feature vector that lies outside the current portfolio's PCA subspace. 100% = completely new creative territory. |

---

## ML Methodology

### Features (36 total)

**18 image features** extracted per PNG asset using PIL:
- Per-channel (R, G, B) mean, std, skewness (9)
- Brightness and contrast (2)
- Saturation mean and std (2)
- Edge density x/y/average (3)
- Top/mid/bottom brightness zones (3, vertical composition proxy)
- Hasler–Süsstrunk colorfulness index (1)

**12 tabular features:** CTR, CVR, ROAS, IPM, decay metrics, duration, text density, readability, brand visibility, clutter, novelty, motion scores.

**5 label-encoded categoricals:** format, vertical, theme, tone, OS.

**4 binary flags:** has_gameplay, has_ugc_style, has_price, has_discount_badge.

All features are median-imputed (`SimpleImputer`) and standardised (`StandardScaler`) before PCA/clustering/modelling.

### Models

| Purpose | Model | n |
|---------|-------|---|
| Global perf_score | RandomForestRegressor (200 trees, max_features='sqrt') | 1 080 |
| Global CTR / CVR / ROAS | RandomForestRegressor | 1 080 |
| CPA-campaign CVR | XGBRegressor (200 rounds, lr=0.05, depth=5) | ~270 |
| ROAS-campaign ROAS | XGBRegressor | ~312 |
| IPM-campaign IPM | XGBRegressor | ~252 |
| CTR-campaign CTR | XGBRegressor | ~246 |

### Validation

All models evaluated with 5-fold cross-validation (`KFold(n_splits=5, shuffle=True, random_state=42)`). R² and MAE reported in the ML Insights tab.

### SHAP

`shap.TreeExplainer` computes exact Shapley values for every creative. When not installed, a simplified approximation (`feature_importance × standardised_feature_value`) is used. Both feed the same structured JSON payload to Gemini when the Explain button is clicked.

---

## File Structure

```
Smadex-Challenge/
│
├── creative_copilot.html       ← Built dashboard — open in any browser (regenerate with src/build_dashboard.py)
├── app_data.js                 ← Generated by ML pipeline (gitignored — large)
├── ml_results.json             ← Generated by ML pipeline (gitignored — large)
├── requirements.txt
├── README.md
├── challenge_instructions.md
├── .env.example                ← Template for GEMINI_API_KEY (optional, ML pipeline personas only)
│
├── data/
│   ├── data_dictionary.csv
│   ├── advertisers.csv
│   ├── campaigns.csv
│   ├── creatives.csv
│   ├── creative_summary.csv              ← Main fact table for the dashboard
│   ├── creative_daily_country_os_stats.csv
│   ├── campaign_summary.csv
│   └── assets/                         ← Synthetic PNG thumbnails (creative_<id>.png)
│
└── src/
    ├── ml_pipeline_sklearn.py          ← Regenerates app_data.js + ml_results.json
    ├── build_dashboard.py              ← Builds creative_copilot.html
    └── dashboard_template.html        ← HTML/JS shell patched by the builder
```

---

## Design Decisions

**No API key in HTML** — Gemini credentials are supplied in the Help tab and live in the browser only, so the built `creative_copilot.html` does not embed secrets.

**Self-contained HTML** — zero server setup for demo. All data embedded as JS variables; ~3 MB is comfortably within what modern browsers handle instantly.

**PIL over CLIP** — CLIP requires PyTorch (~2 GB), impractical for a hackathon. The 18 PIL features capture the most interpretable visual signals and are fully explainable to a non-technical marketer.

**RF for global + XGBoost for KPI subsets** — RF gives stable global feature importance on the full dataset; XGBoost with SHAP gives exact per-creative attribution on the smaller KPI-specific subsets where gradient boosting's bias-variance tradeoff is more favourable.

**Gram-Schmidt for portfolio diversity** — the exact linear-algebra formulation for "what fraction of this vector lies outside the current subspace". Directly maps to "creative territory not yet covered" without heuristic approximation.

**CTR normalisation in fatigue chart** — raw CTR values are not comparable across creatives with different baseline CTRs. Expressing daily CTR as a percentage of the first-7-day average puts all creatives on the same scale so decay curves are visually comparable.

---

## Limitations

- The dataset is **fully synthetic**. Patterns may not generalise to real ad inventory.
- Image features are **heuristic** (PIL-based), not semantic (no object detection or brand recognition).
- The prediction engine uses a **cluster-match heuristic** until `python src/ml_pipeline_sklearn.py` is run locally to embed full PCA/scaler parameters.
- The AI explain, chatbot, and similar features need a **Gemini API key** in the browser (Help tab). Optionally, set `GEMINI_API_KEY` in `.env` when running `python src/ml_pipeline_sklearn.py` so optional **cluster personas** can be generated server-side — that file stays out of git via `.gitignore`.
- SHAP values are computed on training data; use CV R² for honest accuracy estimates.

---

## Acknowledgements

Built for the **Smadex Creative Intelligence Challenge**.
Dataset provided by Smadex — fully synthetic, no real user data.
Dashboard: [Chart.js](https://www.chartjs.org/) · AI: [Google Gemini](https://ai.google.dev/).
