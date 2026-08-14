# 💊 Pharma Commercial Analytics + GenAI

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-modeling-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-GenAI-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End-to-end commercial data science on real pharmaceutical marketing data (CMS Open Payments): predictive modeling of HCP engagement, a natural-language analytics assistant (text-to-SQL + insight narration), an interactive dashboard, and a responsible-AI layer.

> Status: complete (data pipeline, retention model, GenAI assistant, dashboard, model card).

---

## Objective

Pharmaceutical companies spend heavily engaging healthcare professionals (HCPs): consulting fees, speaking, meals, travel, education. Commercial teams need to:

- **understand and segment** that engagement,
- **predict** which HCPs stay engaged from one year to the next (retention),
- let **non-technical stakeholders** explore the data in plain language,
- all under **responsible-AI** constraints (transparency, fairness, governance).

This project builds a compact end-to-end system on public data that mirrors the missions of a pharma Commercial Data Science role.

---

## Data

**CMS Open Payments** (US, public): industry payments and transfers of value to physicians and teaching hospitals. One row per payment (recipient, specialty, state, manufacturer, amount, nature, date). See [`data/README.md`](data/README.md) for how to download a manageable slice.

The raw data is not versioned (see `.gitignore`).

---

## Components (mapped to the role's missions)

| Mission | Deliverable |
| --- | --- |
| IA / GenAI, conversational AI, analysis tools | `src/03_genai_assistant.py` : text-to-SQL assistant + LLM insight narrator |
| Preparation, modeling, evaluation of commercial data | `src/02_modeling.ipynb` : HCP retention model, baseline, metrics |
| Collection, cleaning, structuring, quality | `src/01_prepare_data.py` : cleaning + relational SQLite + quality audit |
| Predictive / ML models | HCP retention (XGBoost / Random Forest) + segmentation (k-means) |
| Data visualization + storytelling | `src/dashboard.py` : Streamlit dashboard + auto-generated narrative |
| Responsible AI | `responsible_ai/model_card.md` : SHAP transparency, bias check, governance, GDPR note |

---

## Predictive task

**HCP retention (churn).** From a physician's year-N engagement profile (specialty, state, number of distinct manufacturers, payment-type mix, number of interactions), predict whether they will still receive any industry payment in year N+1. A clean, leakage-aware commercial target, complemented by an unsupervised **segmentation** of engagement profiles.

---

## Results

Scope: West Virginia, 2022 to 2023, 5,707 HCPs, 73.3% retention.

- **Retention model**: ROC-AUC **0.807** (majority baseline 0.500), PR-AUC 0.922. Main drivers (feature importance and SHAP agree): number of payments, number of manufacturers, total amount. Retention is driven by engagement intensity, not specialty.
- **Segmentation** (k-means, 4 profiles): a loyal core (76 payments per HCP, 98% retention), a high-value group, a large low-touch group, and a minimal-contact group that churns most (56% retention).
- **Fairness**: strong and consistent for the large groups (physicians and PA/APN, AUC around 0.80) but markedly weaker for Dental Providers (AUC 0.633), so predictions are not applied uniformly. See the model card.
- **GenAI assistant**: natural-language questions are translated to SQL over a semantic view and answered (for example "how many HCPs in 2023?" returns 6,230). A small local model (llama3.2) keeps it free but imperfect, an explicit cost / reliability trade-off.

---

## Structure

```
pharma-commercial-genai/
├── data/                       # Open Payments slice (not versioned) + download guide
├── src/
│   ├── 01_prepare_data.py      # clean + structure into SQLite, quality audit
│   ├── 02_modeling.ipynb       # retention model + segmentation + evaluation + SHAP
│   ├── 03_genai_assistant.py   # text-to-SQL + insight narrator (local LLM)
│   └── dashboard.py            # Streamlit: KPIs + model + SHAP + assistant
├── responsible_ai/
│   └── model_card.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Responsible AI

Profiling HCPs by commercial value raises real questions, so the project treats them explicitly: model transparency via **SHAP**, a **fairness check** across specialties and regions, documented **limitations and leakage controls**, and a **data-governance / GDPR** note. Summarised in a **model card**.

---

## Stack

Python 3.11 · pandas · scikit-learn · XGBoost · SHAP · SQLite · LangChain · Streamlit · Ollama / OpenAI

---

## Author

**Juliette Bouli-Mengue**
Clinical Research to Data Science
