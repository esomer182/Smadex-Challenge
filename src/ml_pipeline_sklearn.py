"""
Smadex Creative Copilot — ML Pipeline (scikit-learn + xgboost edition)
=======================================================================
Run once to regenerate ml_results.json and app_data.js, then run
python src/build_dashboard.py to rebuild creative_copilot.html.

Install dependencies:
    pip install pandas scikit-learn xgboost shap

Usage (from the repository root):
    python src/ml_pipeline_sklearn.py
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
    print("✓ xgboost found — KPI-specific models will use XGBRegressor")
except ImportError:
    HAS_XGB = False
    print("⚠  xgboost not found — using RandomForestRegressor for all models")
    print("   (install with: pip install xgboost)")

try:
    import shap
    HAS_SHAP = True
    print("✓ shap found — per-creative SHAP attributions will be computed")
except ImportError:
    HAS_SHAP = False
    print("⚠  shap not found — using simplified feature attribution")
    print("   (install with: pip install shap)")

try:
    from PIL import Image
    HAS_PIL = True
    print("✓ Pillow found — image features will be extracted from thumbnails")
except ImportError:
    HAS_PIL = False
    print("⚠  Pillow not found — skipping image features (pip install Pillow)")

# ─────────────────────────────────────────────────────────────────────
# PATHS  (relative to this script)
# ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = REPO_ROOT    # app_data.js + ml_results.json at repo root

np.random.seed(42)

# ═════════════════════════════════════════════════════════════════════
# 1.  LOAD & JOIN DATA
# ═════════════════════════════════════════════════════════════════════
print("\n── 1. Loading CSV data ───────────────────────────────────────")

creatives_df = pd.read_csv(DATA_DIR / "creative_summary.csv")
campaigns_df = pd.read_csv(DATA_DIR / "campaigns.csv")
daily_df     = pd.read_csv(DATA_DIR / "creative_daily_country_os_stats.csv")

# ── Column aliases so the rest of the code uses clean names ──────────
creatives_df = creatives_df.rename(columns={
    "format": "ad_format",
    "emotional_tone": "tone",
    "creative_status": "performance_status",
})

# ── Compute CPA = spend ÷ conversions (before any merges) ────────────
creatives_df["cpa"] = (
    pd.to_numeric(creatives_df["total_spend_usd"], errors="coerce") /
    pd.to_numeric(creatives_df["total_conversions"], errors="coerce").replace(0, np.nan)
)

# ── Join campaign metadata (kpi_goal, target_os, age, countries) ─────
creatives_df = creatives_df.merge(
    campaigns_df[["campaign_id", "kpi_goal", "target_os", "target_age_segment", "countries"]],
    on="campaign_id",
    how="left",
)
# Use target_os as the OS feature (campaigns have iOS / Android / Both)
creatives_df["os"] = creatives_df["target_os"].fillna("Both")

# ── Derived features from campaign geography ──────────────────────────
# n_countries: number of target countries (from pipe-separated string)
creatives_df["n_countries"] = (
    creatives_df["countries"].fillna("").astype(str)
    .apply(lambda x: len([c for c in x.split("|") if c.strip()]))
)
# has_us: whether US is among target countries
creatives_df["has_us"] = (
    creatives_df["countries"].fillna("").astype(str)
    .apply(lambda x: 1.0 if "US" in x.upper().split("|") else 0.0)
)

# ── Numeric coercion ─────────────────────────────────────────────────
FLOAT_COLS = [
    "overall_ctr", "overall_cvr", "overall_roas", "overall_ipm", "perf_score",
    "ctr_decay_pct", "cvr_decay_pct", "text_density", "readability_score",
    "brand_visibility_score", "clutter_score", "novelty_score", "motion_score",
    "duration_sec", "faces_count", "product_count", "copy_length_chars",
    "n_countries", "has_us",
    "has_price", "has_discount_badge", "has_gameplay", "has_ugc_style",
    "total_spend_usd", "total_impressions", "total_conversions", "total_revenue_usd",
    "fatigue_day", "first_7d_ctr", "last_7d_ctr", "first_7d_cvr", "last_7d_cvr",
    "total_days_active",
]
for col in FLOAT_COLS:
    if col in creatives_df.columns:
        creatives_df[col] = pd.to_numeric(creatives_df[col], errors="coerce")

print(f"  {len(creatives_df)} creatives, {len(campaigns_df)} campaigns")
print(f"  KPI distribution:")
print("    " + creatives_df["kpi_goal"].value_counts().to_string().replace("\n", "\n    "))
print(f"  Status distribution:")
print("    " + creatives_df["performance_status"].value_counts().to_string().replace("\n", "\n    "))


# ═════════════════════════════════════════════════════════════════════
# 2.  IMAGE FEATURE EXTRACTION  (PIL-based, ~12 features per thumbnail)
# ─────────────────────────────────────────────────────────────────────
# Lightweight visual features derived from the actual creative pixels.
# These augment the tabular feature matrix with information no other
# column captures: brightness, saturation, contrast, edge density,
# colour palette diversity, warm/cool balance, left-right symmetry,
# and a center-focus score. A CLIP/vision-model path can be plugged in
# later to replace this block with semantic embeddings; the downstream
# pipeline is agnostic to which method produced the features.
# ═════════════════════════════════════════════════════════════════════
IMG_FEATURE_NAMES = [
    "img_brightness", "img_brightness_std",
    "img_saturation", "img_contrast",
    "img_edge_density", "img_color_diversity",
    "img_warmth", "img_symmetry", "img_center_focus",
]

def _extract_image_features(image_path):
    """Return dict of 9 visual features, or None on any failure."""
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((128, 128))
            arr = np.asarray(im, dtype=np.float32) / 255.0
        h, w, _ = arr.shape
        if h < 4 or w < 4:
            return None
        # Luminance (Rec. 709)
        lum = 0.2126*arr[..., 0] + 0.7152*arr[..., 1] + 0.0722*arr[..., 2]
        brightness     = float(lum.mean())
        brightness_std = float(lum.std())
        # Saturation (HSV-style: (max-min)/max)
        max_c = arr.max(axis=2); min_c = arr.min(axis=2)
        saturation = float(np.where(max_c > 1e-6, (max_c - min_c) / np.maximum(max_c, 1e-6), 0).mean())
        # Contrast: range of luminance after smoothing extremes
        contrast = float(np.percentile(lum, 95) - np.percentile(lum, 5))
        # Edge density: mean abs gradient magnitude
        gx = np.abs(np.diff(lum, axis=1)).mean()
        gy = np.abs(np.diff(lum, axis=0)).mean()
        edge_density = float((gx + gy) / 2)
        # Color diversity: distinct buckets in 8x8x8 quantization, normalised
        q = (arr * 7.999).astype(np.uint8)
        codes = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
        color_diversity = float(np.unique(codes).size / 512.0)
        # Warmth: red+green energy vs blue (warm > cool when > 0.5)
        warm = arr[..., 0].mean() + 0.5 * arr[..., 1].mean()
        cool = arr[..., 2].mean()
        warmth = float(warm / (warm + cool + 1e-6))
        # Left-right symmetry: Pearson r of left half vs mirrored right
        half = w // 2
        left  = lum[:, :half]
        right = lum[:, w - half:][:, ::-1]
        l = left.flatten(); r = right.flatten()
        if l.std() > 1e-6 and r.std() > 1e-6:
            corr = float(((l - l.mean()) * (r - r.mean())).mean() / (l.std() * r.std()))
            symmetry = max(0.0, corr)
        else:
            symmetry = 0.0
        # Center focus: how much variance is concentrated in the central 50%
        cy0, cy1 = h // 4, 3 * h // 4
        cx0, cx1 = w // 4, 3 * w // 4
        cv = lum[cy0:cy1, cx0:cx1].var()
        full = lum.var() + 1e-6
        center_focus = float(cv / full)
        return {
            "img_brightness":      brightness,
            "img_brightness_std":  brightness_std,
            "img_saturation":      saturation,
            "img_contrast":        contrast,
            "img_edge_density":    edge_density,
            "img_color_diversity": color_diversity,
            "img_warmth":          warmth,
            "img_symmetry":        symmetry,
            "img_center_focus":    center_focus,
        }
    except Exception:
        return None


print("\n── 2. Extracting image features from creative thumbnails ─────")
ASSETS_DIR = DATA_DIR / "assets"
img_rows = []
extracted = 0
if HAS_PIL and ASSETS_DIR.exists():
    for cid in creatives_df["creative_id"]:
        path = ASSETS_DIR / f"creative_{cid}.png"
        feats = _extract_image_features(path) if path.exists() else None
        if feats is None:
            img_rows.append({n: np.nan for n in IMG_FEATURE_NAMES})
        else:
            img_rows.append(feats)
            extracted += 1
    print(f"  Extracted from {extracted}/{len(creatives_df)} thumbnails")
else:
    for _ in range(len(creatives_df)):
        img_rows.append({n: np.nan for n in IMG_FEATURE_NAMES})
    print("  (skipped — Pillow missing or assets/ directory not found)")

img_df = pd.DataFrame(img_rows, index=creatives_df.index)
for col in IMG_FEATURE_NAMES:
    creatives_df[col] = img_df[col].values


# ═════════════════════════════════════════════════════════════════════
# 3.  FEATURE MATRIX  (tabular + categorical + binary + IMAGE features)
# ═════════════════════════════════════════════════════════════════════
print("\n── 3. Building feature matrix ────────────────────────────────")

# Only creative properties knowable BEFORE the campaign runs.
# Outcome metrics (overall_ctr, overall_cvr, overall_roas, overall_ipm,
# ctr_decay_pct, cvr_decay_pct) are excluded — using them as features
# when predicting perf_score (which is derived from them) is data leakage.
TABULAR = [
    "duration_sec", "copy_length_chars", "product_count", "faces_count",
    "text_density", "readability_score", "brand_visibility_score",
    "clutter_score", "novelty_score", "motion_score",
    "n_countries",
]

CAT_COLS = ["ad_format", "vertical", "theme", "tone", "os", "language", "target_age_segment", "hook_type", "dominant_color"]
le_map   = {}
for col in CAT_COLS:
    if col in creatives_df.columns:
        le = LabelEncoder()
        creatives_df[f"{col}_enc"] = le.fit_transform(
            creatives_df[col].fillna("unknown").astype(str)
        )
        le_map[col] = le

CAT_ENCODED = [f"{c}_enc" for c in CAT_COLS if c in creatives_df.columns]
BINARY      = ["has_price", "has_discount_badge", "has_gameplay", "has_ugc_style", "has_us"]
for col in BINARY:
    if col in creatives_df.columns:
        creatives_df[col] = pd.to_numeric(creatives_df[col], errors="coerce").fillna(0)

IMG_COLS = [c for c in IMG_FEATURE_NAMES if c in creatives_df.columns]
feat_cols = TABULAR + CAT_ENCODED + [c for c in BINARY if c in creatives_df.columns] + IMG_COLS
ALL_FEAT_NAMES = feat_cols

rows, valid_ids = [], []
for _, row in creatives_df.iterrows():
    cid = row["creative_id"]
    tab = [float(row[c]) if c in creatives_df.columns and pd.notna(row.get(c)) else np.nan
           for c in feat_cols]
    rows.append(tab)
    valid_ids.append(cid)

X_raw = np.array(rows, dtype=np.float64)

imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X_raw)

scaler  = StandardScaler()
X       = scaler.fit_transform(X_imp)

print(f"  Feature matrix: {X.shape[0]} × {X.shape[1]}  "
      f"({len(feat_cols)} features: {len(TABULAR)} tabular + {len(CAT_ENCODED)} categorical + "
      f"{len([c for c in BINARY if c in creatives_df.columns])} binary + {len(IMG_COLS)} image)")


# ═════════════════════════════════════════════════════════════════════
# 4.  PCA  (sklearn.decomposition.PCA)
# ═════════════════════════════════════════════════════════════════════
print("\n── 4. PCA ────────────────────────────────────────────────────")

N_PC   = min(18, X.shape[1], X.shape[0] - 1)
pca    = PCA(n_components=N_PC, random_state=42)
X_pca  = pca.fit_transform(X)

explained  = pca.explained_variance_ratio_.tolist()
cum_var    = float(np.sum(explained))
print(f"  {N_PC} PCs explain {cum_var*100:.1f}% of variance")
print(f"  Top-3 PC variance: {[round(v*100,1) for v in explained[:3]]}%")


# ═════════════════════════════════════════════════════════════════════
# 5.  K-MEANS CLUSTERING  (sklearn, k=8, k-means++)
# ═════════════════════════════════════════════════════════════════════
print("\n── 5. K-Means clustering (k=8) ───────────────────────────────")

km     = KMeans(n_clusters=8, init="k-means++", n_init=10, random_state=42)
labels = km.fit_predict(X_pca)       # 0-indexed internally
labels_1 = labels + 1                # 1-indexed for all output (clusters 1–8)

creatives_df["cluster"] = labels_1
sizes = pd.Series(labels_1).value_counts().sort_index()
print(f"  Cluster sizes: {sizes.tolist()}")


# ═════════════════════════════════════════════════════════════════════
# 6.  FEATURE IMPORTANCE  (sklearn RF + optional xgboost)
# ═════════════════════════════════════════════════════════════════════
print("\n── 6. Feature importance models ──────────────────────────────")

def train_importance(X_tr, y_tr, feat_names, use_xgb=False):
    """Return top-20 [{name, importance}] sorted desc."""
    mask = np.isfinite(y_tr)
    if mask.sum() < 10:
        return []
    Xm, ym = X_tr[mask], y_tr[mask]

    if use_xgb and HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )

    model.fit(Xm, ym)
    pairs = sorted(zip(feat_names, model.feature_importances_),
                   key=lambda x: x[1], reverse=True)
    return [{"name": n, "importance": round(float(v), 5)} for n, v in pairs[:20]]


feature_importance: dict = {}

# Global models — all creatives, RF
GLOBAL_TARGETS = {
    "perf_score": "perf_score",
    "ctr":        "overall_ctr",
    "cvr":        "overall_cvr",
    "roas":       "overall_roas",
}
for key, col in GLOBAL_TARGETS.items():
    y  = creatives_df[col].values if col in creatives_df.columns else np.full(len(creatives_df), np.nan)
    fi = train_importance(X, y, ALL_FEAT_NAMES, use_xgb=False)
    feature_importance[key] = fi
    top = fi[0] if fi else {}
    print(f"  {key:12s}  top: {top.get('name','?')} ({top.get('importance',0):.3f})")

# KPI-specific models — subset by kpi_goal, use xgboost if available
KPI_TARGETS = {
    "kpi_CPA":  ("CPA",  "overall_cvr"),
    "kpi_ROAS": ("ROAS", "overall_roas"),
    "kpi_IPM":  ("IPM",  "overall_ipm"),
    "kpi_CTR":  ("CTR",  "overall_ctr"),
}
for key, (kpi_val, col) in KPI_TARGETS.items():
    mask = (creatives_df["kpi_goal"] == kpi_val).values
    Xk   = X[mask]
    yk   = (creatives_df.loc[mask, col].values
            if col in creatives_df.columns else np.full(mask.sum(), np.nan))
    fi   = train_importance(Xk, yk, ALL_FEAT_NAMES, use_xgb=True)
    feature_importance[key] = fi
    top  = fi[0] if fi else {}
    lib  = "XGB" if HAS_XGB else "RF"
    print(f"  {key:12s}  n={mask.sum():3d}  [{lib}] top: {top.get('name','?')} ({top.get('importance',0):.3f})")


# ═════════════════════════════════════════════════════════════════════
# 7.  CLUSTER PROFILES
# ═════════════════════════════════════════════════════════════════════
print("\n── 7. Cluster profiles ───────────────────────────────────────")

def top_val(series):
    vc = series.dropna().astype(str).value_counts()
    return vc.index[0] if len(vc) else "?"

cluster_profiles: dict = {}
for k in range(8):
    k_key = str(k + 1)  # 1-indexed keys
    sub = creatives_df[creatives_df["cluster"] == k + 1]
    if len(sub) == 0:
        cluster_profiles[k_key] = {}
        continue

    status_dist = {}
    if "performance_status" in sub.columns:
        status_dist = {
            str(s): round(float(v), 3)
            for s, v in sub["performance_status"].value_counts(normalize=True).items()
        }

    # Bootstrap 80 % CI for the four KPI metrics inside this cluster.
    # Used by the dashboard predictor to display ± uncertainty bands.
    def _ci(series):
        v = series.dropna().astype(float).values
        if len(v) < 3:
            return None
        rng = np.random.default_rng(42 + k)
        boot = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(200)])
        lo, hi = np.percentile(boot, [10, 90])
        return [round(float(lo), 6), round(float(np.mean(v)), 6), round(float(hi), 6)]

    cluster_profiles[k_key] = {
        "size":       int(len(sub)),
        "avg_perf":   round(float(sub["perf_score"].mean()), 4) if "perf_score"    in sub.columns else 0,
        "avg_ctr":    round(float(sub["overall_ctr"].mean()), 6) if "overall_ctr"  in sub.columns else 0,
        "avg_roas":   round(float(sub["overall_roas"].mean()), 4) if "overall_roas" in sub.columns else 0,
        "top_format": top_val(sub["ad_format"]),
        "top_theme":  top_val(sub["theme"]),
        "top_tone":   top_val(sub["tone"]),
        "top_kpi":    top_val(sub["kpi_goal"]),
        "status_dist": status_dist,
        "ci_perf": _ci(sub["perf_score"])    if "perf_score"    in sub.columns else None,
        "ci_ctr":  _ci(sub["overall_ctr"])   if "overall_ctr"   in sub.columns else None,
        "ci_roas": _ci(sub["overall_roas"])  if "overall_roas"  in sub.columns else None,
        "ci_ipm":  _ci(sub["overall_ipm"])   if "overall_ipm"   in sub.columns else None,
        "ci_cpa":  _ci(sub["cpa"])           if "cpa"           in sub.columns else None,
    }
    cp = cluster_profiles[k_key]
    print(f"  Cluster {k+1}: n={cp['size']:3d}  perf={cp['avg_perf']:.3f}"
          f"  {cp['top_format']}/{cp['top_theme']}/{cp['top_tone']}")


# ═════════════════════════════════════════════════════════════════════
# 8.  PER-CREATIVE ML METADATA  (cluster id + first 10 PC scores)
# ═════════════════════════════════════════════════════════════════════
_img_col_idx = [feat_cols.index(c) for c in IMG_COLS] if IMG_COLS else []

creative_ml: dict = {}
for i, cid in enumerate(valid_ids):
    entry = {
        "cluster": int(labels[i]) + 1,
        "pc":      [round(float(v), 4) for v in X_pca[i, :10]],
    }
    if _img_col_idx:
        entry["img_emb"] = [round(float(X[i, j]), 4) for j in _img_col_idx]
    creative_ml[cid] = entry


# ═════════════════════════════════════════════════════════════════════
# 8b. CROSS-VALIDATION  (5-fold, R² + MAE)
# ═════════════════════════════════════════════════════════════════════
print("\n── 8b. Cross-validation (5-fold) ─────────────────────────────")

from sklearn.metrics import mean_absolute_error

cv_results: dict = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

CV_TARGETS = {**GLOBAL_TARGETS, **{k: v[1] for k, v in KPI_TARGETS.items()}}

for key, col in CV_TARGETS.items():
    if col not in creatives_df.columns:
        continue
    y_cv = creatives_df[col].values.astype(float)
    mask  = np.isfinite(y_cv)
    if mask.sum() < 20:
        continue

    # Use KPI subset for KPI-specific models
    if key.startswith("kpi_"):
        kpi_val = key.split("_", 1)[1]
        kpi_mask = (creatives_df["kpi_goal"] == kpi_val).values & mask
        Xcv, ycv = X[kpi_mask], y_cv[kpi_mask]
    else:
        Xcv, ycv = X[mask], y_cv[mask]

    if len(Xcv) < 20:
        continue

    if HAS_XGB and key.startswith("kpi_"):
        cv_model = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=4,
            random_state=42, verbosity=0,
        )
    else:
        cv_model = RandomForestRegressor(
            n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1
        )

    r2_scores  = cross_val_score(cv_model, Xcv, ycv, cv=kf, scoring="r2")
    mae_scores = cross_val_score(cv_model, Xcv, ycv, cv=kf,
                                 scoring="neg_mean_absolute_error")

    cv_results[key] = {
        "r2_mean":  round(float(r2_scores.mean()), 4),
        "r2_std":   round(float(r2_scores.std()),  4),
        "mae_mean": round(float(-mae_scores.mean()), 6),
        "mae_std":  round(float(mae_scores.std()),  6),
        "n_samples": int(len(Xcv)),
    }
    print(f"  {key:12s}  R²={cv_results[key]['r2_mean']:+.3f} ± {cv_results[key]['r2_std']:.3f}"
          f"  MAE={cv_results[key]['mae_mean']:.4f}  n={cv_results[key]['n_samples']}")


# ═════════════════════════════════════════════════════════════════════
# 8c. SHAP / FEATURE ATTRIBUTION  (per-creative top drivers)
# ═════════════════════════════════════════════════════════════════════
print("\n── 8c. Per-creative feature attribution ──────────────────────")

# Train the main perf_score model on full data for SHAP.
# Always use RandomForestRegressor here — shap.TreeExplainer is fully
# compatible with sklearn RF across all versions, whereas xgboost 2.x
# changed how base_score is serialised, causing a parse error in older
# shap builds (ValueError: could not convert string '[5.00E-1]' to float).
# XGBoost is still used for the KPI-specific feature importance models above.
y_main    = creatives_df["perf_score"].values.astype(float)
mask_main = np.isfinite(y_main)

shap_model = RandomForestRegressor(
    n_estimators=200, max_features="sqrt", random_state=42, n_jobs=-1
)
shap_model.fit(X[mask_main], y_main[mask_main])

creative_shap: dict = {}

if HAS_SHAP:
    print("  Computing SHAP values with shap.TreeExplainer (RandomForest) …")
    explainer   = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X)       # (n_samples, n_features)
    # Newer shap (>=0.44) wraps single-output regression results in an extra
    # dimension and returns expected_value as a 1-element array, while older
    # versions return scalars / 2-D arrays. Normalise both to plain shapes.
    sv_arr = np.asarray(shap_values)
    if sv_arr.ndim == 3 and sv_arr.shape[-1] == 1:
        sv_arr = sv_arr[..., 0]
    shap_values = sv_arr
    base_value  = float(np.asarray(explainer.expected_value).flatten()[0])

    for i, cid in enumerate(valid_ids):
        sv = shap_values[i]
        pairs = sorted(
            zip(ALL_FEAT_NAMES, sv), key=lambda x: abs(x[1]), reverse=True
        )[:10]
        creative_shap[cid] = {
            "base": round(base_value, 4),
            "top":  [{"name": n, "shap": round(float(v), 5)} for n, v in pairs],
        }
    print(f"  SHAP done for {len(creative_shap)} creatives (base={base_value:.3f})")

else:
    # Simplified attribution: importance × (standardised feature value)
    # Positive = feature value pulls score up, negative = pulls down
    print("  Using simplified attribution (importance × feature value) …")
    fi_arr = shap_model.feature_importances_          # (n_features,)
    # Global mean perf score as pseudo-base
    base_value = float(np.nanmean(y_main))

    for i, cid in enumerate(valid_ids):
        x_std = X[i]                                  # already standardised
        contrib = fi_arr * x_std                      # crude attribution
        pairs = sorted(
            zip(ALL_FEAT_NAMES, contrib), key=lambda x: abs(x[1]), reverse=True
        )[:10]
        creative_shap[cid] = {
            "base": round(base_value, 4),
            "top":  [{"name": n, "shap": round(float(v), 5)} for n, v in pairs],
        }
    print(f"  Simplified attribution done for {len(creative_shap)} creatives")


# ═════════════════════════════════════════════════════════════════════
# 8d. CLUSTER SHAP PROFILES  (what defines each cluster?)
# ═════════════════════════════════════════════════════════════════════
print("\n── 8d. Cluster SHAP profiles ─────────────────────────────────")

# Train a multiclass RF classifier: predict cluster membership from features.
# SHAP on this classifier tells us which features *define* each cluster —
# i.e. which features most strongly push a creative into cluster k vs others.
cluster_clf = RandomForestClassifier(
    n_estimators=200, max_features="sqrt", random_state=42, n_jobs=-1
)
cluster_clf.fit(X, labels)   # 0-indexed labels internally

cluster_shap_profiles: dict = {}

if HAS_SHAP:
    print("  Computing cluster SHAP values (RandomForestClassifier) …")
    exp_clf = shap.TreeExplainer(cluster_clf)
    shap_values_clf = exp_clf.shap_values(X)

    # shap returns different shapes depending on version:
    #   Old shap (<0.45 approx): list of n_classes arrays, each (n_samples, n_features)
    #   New shap (>=0.45 approx): single ndarray of shape (n_samples, n_features, n_classes)
    sv_raw = np.array(shap_values_clf)
    if sv_raw.ndim == 3 and sv_raw.shape[0] == X.shape[0]:
        # New format: (n_samples, n_features, n_classes)
        get_class_sv = lambda k: sv_raw[:, :, k]
    elif sv_raw.ndim == 3 and sv_raw.shape[2] == X.shape[0]:
        # Transposed new format: (n_classes, n_features, n_samples)
        get_class_sv = lambda k: sv_raw[k].T
    else:
        # Old format: (n_classes, n_samples, n_features) after np.array on list
        get_class_sv = lambda k: sv_raw[k]

    for k in range(8):
        mask_k = labels == k
        if mask_k.sum() == 0:
            continue
        sv_k = get_class_sv(k)            # (n_samples, n_features)
        mean_sv = sv_k[mask_k].mean(axis=0)
        pairs = sorted(zip(ALL_FEAT_NAMES, mean_sv),
                       key=lambda x: abs(x[1]), reverse=True)[:10]
        cluster_shap_profiles[str(k + 1)] = {
            "top": [{"name": n, "shap": round(float(v), 5)} for n, v in pairs]
        }
    print(f"  Cluster SHAP done for {len(cluster_shap_profiles)} clusters")

else:
    # Fallback: use per-cluster mean feature values (standardised) as a proxy
    print("  Using mean standardised feature values as cluster signature …")
    for k in range(8):
        mask_k = labels == k
        if mask_k.sum() == 0:
            continue
        mean_x = X[mask_k].mean(axis=0)
        pairs = sorted(zip(ALL_FEAT_NAMES, mean_x),
                       key=lambda x: abs(x[1]), reverse=True)[:10]
        cluster_shap_profiles[str(k + 1)] = {
            "top": [{"name": n, "shap": round(float(v), 5)} for n, v in pairs]
        }
    print(f"  Fallback signatures done for {len(cluster_shap_profiles)} clusters")


# ═════════════════════════════════════════════════════════════════════
# 8d-bis. CLUSTER PERSONAS  (one-shot Gemini description per cluster)
# ─────────────────────────────────────────────────────────────────────
# Generate a marketer-friendly archetype name + 1-line description +
# "when to use" + "watch out for" — once at build time, cached into the
# JSON so the dashboard ships with rich personas instead of cold IDs.
# Skipped silently if no GEMINI_API_KEY is configured.
# ═════════════════════════════════════════════════════════════════════
print("\n── 8d-bis. Cluster personas (Gemini) ─────────────────────────")

def _load_gemini_key():
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == "GEMINI_API_KEY":
            v = value.strip().strip('"').strip("'")
            return v if v and v != "your-api-key-here" else ""
    return ""

cluster_personas: dict = {}
_gem_key = _load_gemini_key()
if _gem_key:
    import urllib.request
    import urllib.error
    print("  Gemini key found — generating personas for 8 clusters…")

    def _gemini_persona(cluster_id, profile, shap_top):
        prompt_sys = (
            "You are a senior mobile-advertising creative strategist. "
            "Given a cluster of similar ad creatives and the attributes that "
            "set them apart, produce a JSON object with exactly these keys:\n"
            "  name        : a 2-4 word marketer-friendly archetype name\n"
            "  tagline     : a single 8-15 word description\n"
            "  when_to_use : a single 12-25 word recommendation for when to use this style\n"
            "  watch_out   : a single 12-25 word warning about typical risks\n"
            "Return ONLY raw JSON, no markdown fences, no commentary."
        )
        shap_block = ""
        if shap_top:
            shap_lines = [
                f"  - {x['name']}: {'+' if x['shap']>=0 else '-'}{abs(x['shap']):.3f}"
                for x in shap_top[:6]
            ]
            shap_block = "Distinguishing signals (+ = more than dataset average, - = less):\n" + "\n".join(shap_lines)
        msg = (
            f"Cluster C{cluster_id} — n={profile.get('size','?')} creatives\n"
            f"Most common format: {profile.get('top_format','?')}\n"
            f"Most common theme:  {profile.get('top_theme','?')}\n"
            f"Most common tone:   {profile.get('top_tone','?')}\n"
            f"Most common KPI:    {profile.get('top_kpi','?')}\n"
            f"Average perf score: {(profile.get('avg_perf') or 0)*100:.0f}/100\n"
            f"Average CTR:        {(profile.get('avg_ctr') or 0)*100:.2f}%\n"
            f"Average ROAS:       {profile.get('avg_roas') or 0:.2f}x\n\n"
            f"{shap_block}"
        )
        body = {
            "system_instruction": {"parts": [{"text": prompt_sys}]},
            "contents": [{"role": "user", "parts": [{"text": msg}]}],
            "generationConfig": {"maxOutputTokens": 400, "temperature": 0.4,
                                 "thinkingConfig": {"thinkingBudget": 0}},
        }
        # Try a couple of reliable models in order.
        for model in ("gemini-2.5-flash", "gemini-2.0-flash-001", "gemini-1.5-flash-latest"):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={_gem_key}"
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                txt = (data.get("candidates", [{}])[0]
                           .get("content", {}).get("parts", [{}])[0]
                           .get("text", "") or "").strip()
                if not txt:
                    continue
                # Strip ```json fences if present
                if txt.startswith("```"):
                    txt = txt.strip("`")
                    if txt.lower().startswith("json"):
                        txt = txt[4:].strip()
                obj = json.loads(txt)
                return {k: str(obj.get(k, "")).strip() for k in ("name", "tagline", "when_to_use", "watch_out")}
            except (urllib.error.HTTPError, urllib.error.URLError,
                    json.JSONDecodeError, KeyError, ValueError):
                continue
        return None

    for k_key, prof in cluster_profiles.items():
        shap_top = (cluster_shap_profiles.get(k_key) or {}).get("top", [])
        persona = _gemini_persona(k_key, prof, shap_top)
        if persona:
            cluster_personas[k_key] = persona
            print(f"  C{k_key}: {persona.get('name','?')} — {persona.get('tagline','')[:80]}")
        else:
            print(f"  C{k_key}: skipped (Gemini call failed)")
else:
    print("  No Gemini key in .env — skipping persona generation")


# ═════════════════════════════════════════════════════════════════════
# 8e. EMBED MODEL PARAMETERS  (for JS new-creative predictor)
# ═════════════════════════════════════════════════════════════════════
print("\n── 8d. Embedding model parameters for JS predictor ──────────")

# Label-encoder category → index mappings
le_mappings: dict = {}
for col, le in le_map.items():
    le_mappings[col] = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}

# Scaler params
scaler_params = {
    "mean": [round(float(v), 6) for v in scaler.mean_],
    "std":  [round(float(v), 6) for v in scaler.scale_],
}

# Imputer medians
imputer_medians = [round(float(v), 6) for v in imputer.statistics_]


# PCA: components (n_components x n_features), mean (n_features,)
pca_params = {
    "mean":       [round(float(v), 6) for v in pca.mean_],
    "components": [[round(float(v), 6) for v in row] for row in pca.components_],
    "explained_variance_ratio": [round(float(v), 6) for v in pca.explained_variance_ratio_],
}

# Cluster centers in KMeans PCA space
km_centers = [[round(float(v), 4) for v in c] for c in km.cluster_centers_]

# Simple linear regression: perf_score ~ PC scores (OLS)
y_lr = creatives_df["perf_score"].values.astype(float)
mask_lr = np.isfinite(y_lr)
X_lr  = np.column_stack([X_pca[mask_lr], np.ones(mask_lr.sum())])
y_lr2 = y_lr[mask_lr]
try:
    beta = np.linalg.lstsq(X_lr, y_lr2, rcond=None)[0]
    lr_weights   = [round(float(v), 6) for v in beta[:-1]]
    lr_intercept = round(float(beta[-1]), 6)
except Exception:
    lr_weights   = [0.0] * N_PC
    lr_intercept = 0.35

y_pred_lr = X_pca[mask_lr] @ np.array(lr_weights) + lr_intercept
ss_res = np.sum((y_lr2 - y_pred_lr) ** 2)
ss_tot = np.sum((y_lr2 - y_lr2.mean()) ** 2)
lr_r2  = 1 - ss_res / ss_tot if ss_tot > 0 else 0
print(f"  Linear model R2 on train: {lr_r2:.3f}")
print(f"  Scaler: {len(scaler_params['mean'])} features")
print(f"  PCA: {len(pca_params['components'])} components x {len(pca_params['components'][0])} features")
print(f"  Label encoders: {list(le_mappings.keys())}")

model_params = {
    "feature_names":   ALL_FEAT_NAMES,
    "tabular_names":   TABULAR,
    "categorical_names": CAT_ENCODED,
    "binary_names":    [c for c in BINARY if c in creatives_df.columns],
    "image_names":     IMG_COLS,
    "le_mappings":     le_mappings,
    "scaler":          scaler_params,
    "imputer_medians": imputer_medians,
    "pca":             pca_params,
    "km_centers":      km_centers,
    "lr_weights":      lr_weights,
    "lr_intercept":    lr_intercept,
    "lr_r2_train":     round(lr_r2, 4),
}


# =================================================================
# 9.  CTR RETENTION BENCHMARKS  (for fatigue chart)
# =================================================================
print("\n-- 9. CTR retention benchmarks ---------------------------------")

status_ctr_retention: dict = {}
if "performance_status" in creatives_df.columns:
    for status, grp in creatives_df.groupby("performance_status"):
        valid = grp.dropna(subset=["first_7d_ctr", "last_7d_ctr"])
        valid = valid[valid["first_7d_ctr"] > 0]
        if len(valid):
            ret = (valid["last_7d_ctr"] / valid["first_7d_ctr"]).clip(0, 2).median()
            status_ctr_retention[str(status)] = round(float(ret), 3)
print(f"  {status_ctr_retention}")


# =================================================================
# 10. WRITE  ml_results.json
# =================================================================
ml_results = {
    "feature_importance":     feature_importance,
    "cluster_profiles":       cluster_profiles,
    "cluster_shap_profiles":  cluster_shap_profiles,
    "cluster_personas":       cluster_personas,
    "creative_ml":            creative_ml,
    "creative_shap":          creative_shap,
    "pca_explained_variance": [round(v, 4) for v in explained],
    "feature_names":          ALL_FEAT_NAMES,
    "image_feature_names":    IMG_COLS,
    "status_ctr_retention":   status_ctr_retention,
    "cv_results":             cv_results,
    "model_params":           model_params,
    "note": (
        "scikit-learn: PCA, KMeans, RandomForestRegressor"
        + (" | xgboost: XGBRegressor (KPI models)" if HAS_XGB else "")
        + (" | shap: TreeExplainer" if HAS_SHAP else " | simplified attribution")
        + (" | PIL image features" if HAS_PIL else "")
        + (f" | personas={len(cluster_personas)}" if cluster_personas else "")
    ),
}

out_ml = OUT_DIR / "ml_results.json"
with open(out_ml, "w", encoding="utf-8") as f:
    json.dump(ml_results, f, separators=(",", ":"))
print(f"\n+++ Saved {out_ml}  ({out_ml.stat().st_size // 1024} KB)")


# =================================================================
# 11. BUILD  app_data.js
# =================================================================
print("\n-- 11. Building app_data.js ------------------------------------")

daily_df["impressions"] = pd.to_numeric(daily_df["impressions"], errors="coerce")
daily_df["clicks"]      = pd.to_numeric(daily_df["clicks"],      errors="coerce")
daily_df["date"]        = pd.to_datetime(daily_df["date"],        errors="coerce")

daily_agg = (
    daily_df.groupby(["creative_id", "date"], as_index=False)
    .agg({"clicks": "sum", "impressions": "sum"})
)
daily_agg["ctr"] = daily_agg["clicks"] / daily_agg["impressions"].replace(0, np.nan)

ts: dict = defaultdict(list)
for cid, grp in daily_agg.groupby("creative_id"):
    grp = grp.sort_values("date")
    first7_mean = grp.head(7)["ctr"].mean()
    baseline    = float(first7_mean) if pd.notna(first7_mean) and first7_mean > 0 else None
    for _, row in grp.iterrows():
        if pd.isna(row["date"]) or pd.isna(row["ctr"]):
            continue
        entry = {
            "d":   row["date"].strftime("%m-%d"),
            "ctr": round(float(row["ctr"]), 6),
        }
        if baseline:
            entry["ctr_norm"] = round(float(row["ctr"]) / baseline * 100, 2)
        ts[cid].append(entry)

KEEP = [
    "creative_id", "campaign_id", "advertiser_name", "app_name",
    "ad_format", "vertical", "theme", "tone", "os", "language",
    "target_age_segment", "countries", "n_countries",
    "performance_status", "kpi_goal",
    "perf_score", "overall_ctr", "overall_cvr", "overall_roas", "overall_ipm",
    "ctr_decay_pct", "cvr_decay_pct", "fatigue_day", "total_days_active",
    "total_impressions", "total_conversions", "total_spend_usd", "total_revenue_usd",
    "first_7d_ctr", "last_7d_ctr",
    "text_density", "readability_score", "brand_visibility_score",
    "clutter_score", "novelty_score", "motion_score", "duration_sec",
    "copy_length_chars", "product_count", "faces_count",
    "has_gameplay", "has_ugc_style", "has_price", "has_discount_badge", "has_us",
    "dominant_color", "hook_type", "cta_text", "headline",
    "width", "height",
    "cpa",
]
KEEP = [c for c in KEEP if c in creatives_df.columns]

def safe(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):   return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 6)
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v

creatives_out = []
for _, row in creatives_df.iterrows():
    obj = {c: safe(row[c]) for c in KEEP}
    creatives_out.append(obj)

campaigns_out = {}
for _, row in campaigns_df.iterrows():
    campaigns_out[str(row["campaign_id"])] = {
        "id":     str(row.get("campaign_id", "")),
        "kpi_goal": str(row.get("kpi_goal", "CPA")),
    }

app_data = {
    "creatives": creatives_out,
    "ts":        dict(ts),
    "campaigns": campaigns_out,
}

out_app = OUT_DIR / "app_data.js"
with open(out_app, "w", encoding="utf-8") as f:
    f.write("const APP_DATA=" + json.dumps(app_data, separators=(",", ":"), default=str) + ";")

mb = out_app.stat().st_size / 1024 / 1024
print(f"+++ Saved {out_app}  ({mb:.2f} MB)")

print("\n" + "=" * 60)
print("  ML pipeline complete!")
print("  Next step:  python build_dashboard.py")
print("=" * 60)
