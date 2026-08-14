"""
02 - Modélisation : rétention des professionnels de santé (churn).

Lit `hcp_features` (construite par 01_prepare_data.py), entraîne un modèle
prédictif de rétention, l'évalue face à une baseline, et en explique les
décisions (importances + SHAP si disponible). Ajoute une segmentation
non supervisée des profils d'engagement.

Usage :
    python src/02_modeling.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "openpayments.sqlite"

NUM_FEATURES = [
    "n_payments", "total_amount", "mean_amount", "n_manufacturers", "n_natures",
    "share_food", "share_travel", "share_consulting", "share_speaker", "share_education",
]
CAT_FEATURES = ["specialty"]
TARGET = "retenu"


def load() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql("SELECT * FROM hcp_features", con)


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_FEATURES),
    ])
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1
    )
    return Pipeline([("pre", pre), ("clf", model)])


def evaluate(df: pd.DataFrame) -> Pipeline:
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

    # Baseline (classe majoritaire)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    base_auc = roc_auc_score(y_te, dummy.predict_proba(X_te)[:, 1])

    # Modèle
    pipe = build_pipeline().fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n=== Rétention HCP - évaluation ===")
    print(f"  Taux de rétention (base)   : {y.mean():.1%}")
    print(f"  ROC-AUC baseline (naïve)   : {base_auc:.3f}")
    print(f"  ROC-AUC RandomForest       : {roc_auc_score(y_te, proba):.3f}")
    print(f"  PR-AUC  RandomForest       : {average_precision_score(y_te, proba):.3f}")
    print("\n" + classification_report(y_te, pred, digits=3))
    return pipe


def feature_importance(pipe: Pipeline) -> None:
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pipe.named_steps["clf"].feature_importances_
    top = pd.Series(imp, index=names).sort_values(ascending=False).head(12)
    print("=== Importances (RandomForest) ===")
    for n, v in top.items():
        print(f"  {v:6.3f}  {n}")


def shap_explanation(pipe: Pipeline, df: pd.DataFrame) -> None:
    try:
        import shap
    except ImportError:
        print("\n(SHAP non installé - `pip install shap` pour l'explicabilité détaillée)")
        return
    X = df[NUM_FEATURES + CAT_FEATURES].sample(min(500, len(df)), random_state=0)
    Xt = pipe.named_steps["pre"].transform(X)
    names = pipe.named_steps["pre"].get_feature_names_out()
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    vals = explainer.shap_values(Xt)
    arr = vals[1] if isinstance(vals, list) else np.asarray(vals)
    if arr.ndim == 3:              # (n_samples, n_features, n_classes) -> classe positive
        arr = arr[:, :, 1]
    mean_abs = np.abs(arr).mean(axis=0)
    top = pd.Series(mean_abs, index=names).sort_values(ascending=False).head(10)
    print("\n=== SHAP - contributions moyennes (|valeur|) ===")
    for n, v in top.items():
        print(f"  {v:8.4f}  {n}")


def segmentation(df: pd.DataFrame, k: int = 4) -> None:
    X = StandardScaler().fit_transform(df[NUM_FEATURES].fillna(0))
    df = df.copy()
    df["segment"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    print("\n=== Segmentation des profils d'engagement (k-means) ===")
    summary = df.groupby("segment").agg(
        n_hcp=("segment", "size"),
        total_amount_moyen=("total_amount", "mean"),
        n_payments_moyen=("n_payments", "mean"),
        n_labos_moyen=("n_manufacturers", "mean"),
        taux_retention=(TARGET, "mean"),
    )
    print(summary.round(2).to_string())


def fairness_check(df: pd.DataFrame) -> None:
    """Performance et taux de rétention par spécialité (équité entre groupes)."""
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET].astype(int)
    # Probabilités hors-échantillon (5 folds) pour évaluer chaque groupe équitablement
    proba = cross_val_predict(build_pipeline(), X, y, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    d = df.copy()
    d["proba"] = proba
    print("\n=== Fairness - par spécialité (top 6 effectifs) ===")
    print(f"  {'spécialité':45s}  {'n':>5s}  {'rétention':>9s}  {'AUC':>6s}")
    for sp in d["specialty"].value_counts().head(6).index:
        m = d["specialty"] == sp
        if m.sum() >= 50 and d.loc[m, TARGET].nunique() > 1:
            auc = roc_auc_score(d.loc[m, TARGET], d.loc[m, "proba"])
            print(f"  {str(sp)[:45]:45s}  {m.sum():5d}  {d.loc[m, TARGET].mean():9.2f}  {auc:6.3f}")


def main() -> None:
    df = load()
    print(f"Profils chargés : {len(df)}")
    pipe = evaluate(df)
    feature_importance(pipe)
    shap_explanation(pipe, df)
    segmentation(df)
    fairness_check(df)


if __name__ == "__main__":
    main()
