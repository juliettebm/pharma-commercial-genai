# 💊 Pharma Commercial Analytics + GenAI

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-text--to--SQL-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End-to-end commercial data science on real pharmaceutical marketing data (CMS Open Payments): HCP retention modeling with SHAP, engagement segmentation, a GenAI text-to-SQL assistant, an interactive dashboard, and a responsible-AI model card.

---

## Objective

Pharmaceutical companies spend heavily engaging healthcare professionals (HCPs). Commercial teams need to understand and segment that engagement, predict which HCPs stay engaged from one year to the next, let non-technical stakeholders explore the data in plain language, and do all of it under responsible-AI constraints.

This project builds a compact end-to-end system on public data that mirrors the missions of a pharma Commercial Data Science role: cleaning and structuring, predictive modeling and evaluation, GenAI and conversational analytics, data storytelling, and AI governance.

---

## Dataset

- **Source**: CMS Open Payments (US), public. General Payments.
- **Scope**: West Virginia, program year 2022 (features) to 2023 (retention target).
- **Size**: 5,707 HCPs; 186,223 raw payments across the two years.
- **Base rate**: 73.3% retention.
- **Access**: pulled automatically from the CMS data API by `src/01_prepare_data.py` (no manual download). The raw data and the SQLite database are not versioned (see `.gitignore`).
- **Known data-quality gap**: manufacturer names are not normalised ("ABBVIE INC." and "AbbVie Inc." coexist), which biases company-level aggregation. Flagged for governance.

---

## Project Structure

```
pharma-commercial-genai/
│
├── data/
│   └── README.md                   # data source and download notes (raw data not versioned)
├── notebooks/                      # interactive analysis (open, run and modify in Jupyter/VS Code)
│   ├── 01_exploration.ipynb        # EDA: load raw data, observe, and preparation decisions
│   ├── 02_preparation.ipynb        # apply the decisions: clean, aggregate, retention target
│   ├── 03_modeling.ipynb           # retention model, SHAP, segmentation, fairness
│   └── 04_genai_assistant.ipynb    # text-to-SQL assistant
├── src/
│   ├── 01_prepare_data.py          # pull from CMS API, clean, quality audit, build SQLite
│   ├── 02_modeling.py              # retention model, SHAP, segmentation, fairness check
│   ├── 03_genai_assistant.py       # text-to-SQL assistant over a semantic view
│   └── dashboard.py                # Streamlit: KPIs, segments, model, assistant
├── responsible_ai/
│   └── model_card.md               # metrics, fairness, ethics, governance, GDPR
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Reproduce

### 1. Clone

```bash
git clone https://github.com/juliettebm/pharma-commercial-genai.git
cd pharma-commercial-genai
```

### 2. Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Build the dataset (CMS API)

```bash
python src/01_prepare_data.py       # downloads a state slice and builds data/openpayments.sqlite
```

### 4. Model, assistant, dashboard

```bash
python src/02_modeling.py                       # retention model + SHAP + segmentation + fairness
python src/03_genai_assistant.py "Combien de professionnels de sante en 2023 ?"
streamlit run src/dashboard.py                  # interactive dashboard
```

The GenAI assistant needs a local LLM. Install Ollama, run `ollama pull llama3.2`, and keep the app running (or set `LLM_PROVIDER=openai` with an API key).

The same steps are available as interactive notebooks in [`notebooks/`](notebooks/) (01 to 03), to open, run and modify in Jupyter or VS Code.

---

## Methodology

1. **Data preparation**: paginated pull from the CMS data API for one state and two years, cleaning, quality audit, aggregation to an HCP engagement profile, and a relational SQLite database.
2. **Retention model**: Random Forest predicting whether an HCP engaged in year N is still engaged in year N+1. Temporal separation of features and target avoids leakage.
3. **Explainability**: SHAP values on top of feature importances.
4. **Segmentation**: k-means on engagement profiles.
5. **GenAI assistant**: a natural-language question is translated to SQL over a semantic view with short column names, executed, and narrated. Guardrails allow SELECT only.
6. **Responsible AI**: fairness sliced by specialty, plus a model card covering governance and GDPR.

---

## Key Results

Scope: West Virginia, 2022 to 2023, 5,707 HCPs, 73.3% retention.

| Metric | Value |
| --- | --- |
| ROC-AUC (Random Forest) | **0.807** |
| ROC-AUC (majority baseline) | 0.500 |
| PR-AUC | 0.922 |

Main drivers (feature importance and SHAP agree): number of payments, number of manufacturers, total amount. Retention is driven by engagement intensity, not specialty.

**Engagement segments (k-means):**

| Segment | HCPs | Mean amount | Mean payments | Retention |
| --- | ---: | ---: | ---: | ---: |
| Loyal core | 679 | $2,437 | 76 | **98%** |
| High value | 267 | $19,149 | 32 | 81% |
| Low touch | 4,592 | $209 | 6 | 70% |
| Minimal contact | 169 | $172 | 1.6 | **56%** |

**Fairness (AUC by specialty):** strong for the large groups (physicians 0.808, PA/APN 0.802) but markedly weaker for Dental Providers (**0.633**), so predictions are not applied uniformly.

---

## Responsible AI

Profiling HCPs by commercial value is sensitive, so the project treats it explicitly: SHAP transparency, a fairness slice that surfaced the Dental Providers gap, documented limitations and leakage controls, and a data-governance / GDPR note. Summarised in [`responsible_ai/model_card.md`](responsible_ai/model_card.md).

---

## Disclaimer

Educational proof-of-concept on public data concerning real individuals (physicians). Used for analysis and demonstration only, not for individual targeting decisions.

---

## Stack

Python 3.11 · pandas · scikit-learn · SHAP · SQLite · LangChain · Streamlit · Ollama / OpenAI

---

## License

Released under the [MIT License](LICENSE).

---

## Author

**Juliette Bouli-Mengue**
Clinical Research to Data Science
