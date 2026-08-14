# Model Card - HCP Retention Model

A model card documenting the retention model of this project, following the spirit of Mitchell et al. (2019). Written for transparency, fairness review and governance.

---

## Model details

- **Task**: binary classification. Predict whether a healthcare professional (HCP) who received at least one industry payment in year N will still receive any payment in year N+1 (retention / churn).
- **Algorithm**: Random Forest (300 trees, balanced class weights), with standardised numeric features and one-hot encoded specialty.
- **Features** (year N engagement profile): number of payments, total, mean and median amount, number of distinct manufacturers, number of payment natures, share of payment value by type (food, travel, consulting, speaker, education), specialty.
- **Target**: presence of any payment in year N+1.

## Intended use

- **In scope**: aggregate commercial analytics, HCP engagement segmentation, and decision support for planning outreach. A tool to understand engagement dynamics, not to rank individuals for coercive targeting.
- **Out of scope**: automated individual decisions without human oversight; any clinical or prescribing inference; use on populations or regions outside the training scope without re-validation.

## Data

- **Source**: CMS Open Payments (US), public. General Payments.
- **Scope**: state of West Virginia, program year 2022 (features) to 2023 (target).
- **Size**: 5,707 HCPs; 185,027 cleaned payments across the two years (186,426 raw; 1,204 strict duplicates, 0.65%, removed).
- **Base rate**: 73.3% retention.
- **Data-quality gap identified and corrected**: manufacturer names had inconsistent casing/spacing (for example "ABBVIE INC." and "AbbVie Inc."), which would have biased company-level aggregation and the `n_manufacturers` feature. Normalised (`.str.upper().str.strip()`) before aggregation.

## Metrics

| Metric | Value |
| --- | --- |
| ROC-AUC (Random Forest) | **0.805** |
| ROC-AUC (majority baseline) | 0.500 |
| PR-AUC | 0.920 |

Main drivers (feature importance and SHAP agree): **number of payments, number of manufacturers, total amount**. Retention is driven by engagement intensity, not by specialty.

## Fairness

Out-of-fold performance by specialty (5-fold), the core fairness check:

| Specialty | n | Retention | AUC |
| --- | ---: | ---: | ---: |
| Allopathic & Osteopathic Physicians | 2852 | 0.72 | 0.807 |
| Physician Assistants & Advanced Practice Nursing | 2272 | 0.77 | 0.804 |
| Dental Providers | 358 | 0.61 | **0.639** |
| Eye and Vision Services | 158 | 0.85 | 0.808 |
| Podiatric Medicine & Surgery | 57 | 0.82 | 0.898 |

**Finding**: performance is strong and consistent for the two large groups (AUC around 0.80) but **markedly weaker for Dental Providers (AUC 0.639)**. The model must therefore not be applied uniformly across specialties; predictions for under-served groups need a lower-confidence flag or a group-specific model.

## Ethical considerations

- Profiling HCPs by their commercial value is sensitive. The model is a **planning aid under human oversight**, not an autonomous targeting engine.
- **Transparency**: every prediction is explainable via SHAP, and the drivers are behavioural (engagement frequency), not demographic.
- **Proxy risk**: specialty is a feature; it must be monitored so it does not act as a proxy for protected or access-related attributes.

## Data governance and GDPR

- The data is public and concerns **real individuals** (physicians). It is used here for analysis and demonstration only.
- In an EU / GDPR context this processing would require a **lawful basis**, **data minimisation**, a defined **purpose limitation**, and transparency toward data subjects. HCP identifiers would be pseudonymised, and retention would be measured at an aggregate level wherever possible.

## Limitations

- **Coverage**: one state, one year-pair. Not representative of the whole market; re-validate before any transfer.
- **Target definition**: "any payment next year" is a coarse proxy for engagement retention.
- **Leakage control**: features (year N) and target (year N+1) are temporally separated, so there is no target leakage; engagement-intensity features legitimately carry retention signal.
- **GenAI assistant**: the text-to-SQL and narration use a small local model (llama3.2, 3B) that is convenient and free but imperfect; it is a demonstrator, not a production analytics layer.
