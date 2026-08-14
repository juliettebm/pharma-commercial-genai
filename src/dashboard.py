"""
04 - Dashboard Streamlit : KPIs + segmentation + modele de retention + assistant GenAI.

Lancement :
    streamlit run src/dashboard.py

Necessite la base data/openpayments.sqlite (01_prepare_data.py) et, pour
l'assistant, l'application Ollama en marche (modele llama3.2).
"""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
DB = ROOT / "data" / "openpayments.sqlite"


def _load(name: str, path: Path):
    """Charge un module dont le nom de fichier commence par un chiffre."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


modeling = _load("modeling", SRC / "02_modeling.py")
assistant = _load("assistant", SRC / "03_genai_assistant.py")

st.set_page_config(page_title="Pharma Commercial Analytics", layout="wide")


@st.cache_data
def load_features() -> pd.DataFrame:
    return modeling.load()


@st.cache_resource
def train():
    df = load_features()
    X = df[modeling.NUM_FEATURES + modeling.CAT_FEATURES]
    y = df[modeling.TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    pipe = modeling.build_pipeline().fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = (pd.Series(pipe.named_steps["clf"].feature_importances_, index=names)
           .sort_values(ascending=False).head(10))
    return auc, imp


df = load_features()

st.title("💊 Pharma Commercial Analytics + GenAI")
st.caption("Engagement des professionnels de santé (CMS Open Payments, WV 2022-2023) : "
           "segmentation, rétention, assistant conversationnel.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Professionnels de santé", f"{len(df):,}")
c2.metric("Taux de rétention", f"{df['retenu'].mean():.0%}")
c3.metric("Dépense totale (USD)", f"${df['total_amount'].sum():,.0f}")
c4.metric("Paiements / HCP (moy.)", f"{df['n_payments'].mean():.1f}")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Segments d'engagement (k-means)")
    Xs = StandardScaler().fit_transform(df[modeling.NUM_FEATURES].fillna(0))
    d = df.copy()
    d["segment"] = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(Xs)
    seg = d.groupby("segment").agg(
        n_hcp=("segment", "size"),
        montant_moyen=("total_amount", "mean"),
        paiements_moyens=("n_payments", "mean"),
        retention=("retenu", "mean"),
    ).round(2)
    st.dataframe(seg, use_container_width=True)
    st.bar_chart(seg["retention"])

with right:
    st.subheader("Modèle de rétention")
    auc, imp = train()
    st.metric("ROC-AUC (test)", f"{auc:.3f}", help="Baseline naïve = 0.500")
    st.caption("Importances des variables (RandomForest)")
    st.bar_chart(imp)

st.divider()
st.subheader("Assistant conversationnel (text-to-SQL)")
st.caption("Pose une question sur les paiements. Nécessite l'application Ollama (llama3.2) en marche.")
q = st.text_input("Question", "Quels sont les 5 laboratoires qui dépensent le plus au total ?")
if st.button("Interroger") and q:
    with st.spinner("Génération SQL + exécution..."):
        try:
            answer, sql, res = assistant.ask(q, verbose=False)
            st.code(sql, language="sql")
            if res is not None:
                st.dataframe(res.head(20), use_container_width=True)
            st.success(answer)
        except Exception as exc:
            st.error(f"Erreur : {exc}")

st.divider()
st.caption("IA responsable : voir responsible_ai/model_card.md (fairness, gouvernance, RGPD).")
