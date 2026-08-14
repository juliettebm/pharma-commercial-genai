"""
01 - Préparation des données CMS Open Payments (General Payments).

Récupère, via l'API CMS, une tranche maîtrisée (un État, deux années),
la nettoie, la structure en base SQLite, et construit la table de
modélisation (profil d'engagement du professionnel de santé + cible de
rétention).

- Année N   (features) : profil d'engagement du HCP.
- Année N+1 (cible)    : le HCP reçoit-il encore un paiement l'année suivante ?

Sortie : data/openpayments.sqlite
  - table `payments`      : paiements bruts des deux années (pour l'assistant text-to-SQL)
  - table `hcp_features`  : un profil par professionnel + colonne `retenu` (0/1)

Usage :
    python src/01_prepare_data.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
API = "https://openpaymentsdata.cms.gov/api/1"
STATE = "WV"                     # périmètre : un État de volume raisonnable
YEAR_FEATURES = 2022             # année du profil d'engagement
YEAR_TARGET = 2023               # année de la cible de rétention

# Identifiants des datasets "General Payment" par année (API CMS)
DATASET_IDS = {
    2022: "df01c2f8-dc1f-4e79-96cb-8208beaf143c",
    2023: "fb3a65aa-c901-4a38-a813-b04b00dfa2a9",
    2024: "e6b17c6a-2534-4207-a4a1-6746a14911ff",
}

# Colonnes utiles (noms snake_case renvoyés par le datastore)
KEEP = [
    "covered_recipient_profile_id",
    "covered_recipient_type",
    "recipient_state",
    "covered_recipient_specialty_1",
    "applicable_manufacturer_or_applicable_gpo_making_payment_name",
    "total_amount_of_payment_usdollars",
    "nature_of_payment_or_transfer_of_value",
    "program_year",
]

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "openpayments.sqlite"
PAGE = 500                       # l'API CMS plafonne limit à 500 par requête


# --------------------------------------------------------------------------
# Récupération API
# --------------------------------------------------------------------------
def fetch_state_year(year: int, state: str) -> pd.DataFrame:
    """Télécharge par pagination tous les paiements d'un État pour une année."""
    # Forme d'URL stable sur toutes les années : {datasetId}/0 (index de distribution)
    url = f"{API}/datastore/query/{DATASET_IDS[year]}/0"
    rows, offset, total = [], 0, None
    while True:
        params = {
            "limit": PAGE,
            "offset": offset,
            "conditions[0][property]": "recipient_state",
            "conditions[0][value]": state,
            "conditions[0][operator]": "=",
        }
        payload = requests.get(url, params=params, timeout=120).json()
        batch = payload.get("results", [])
        if total is None:
            total = payload.get("count", 0)
            print(f"  {year} / {state} : {total} lignes à récupérer")
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        print(f"    ... {offset}/{total}", end="\r")
        if offset >= total:
            break
    print()
    df = pd.DataFrame(rows)
    return df[[c for c in KEEP if c in df.columns]].copy()


# --------------------------------------------------------------------------
# Nettoyage et contrôle qualité
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Dedoublonnage strict (artefacts de reinjection identifies en EDA)
    df = df.drop_duplicates().copy()
    df["total_amount_of_payment_usdollars"] = pd.to_numeric(
        df["total_amount_of_payment_usdollars"], errors="coerce"
    )
    df["program_year"] = pd.to_numeric(df["program_year"], errors="coerce").astype("Int64")
    # Spécialité : on garde le niveau haut (avant le premier séparateur "|")
    df["specialty"] = (
        df["covered_recipient_specialty_1"].fillna("Unknown").str.split("|").str[0].str.strip()
    )
    # Normalisation des noms de laboratoires (casse/espaces incoherents en EDA)
    df["applicable_manufacturer_or_applicable_gpo_making_payment_name"] = (
        df["applicable_manufacturer_or_applicable_gpo_making_payment_name"].str.upper().str.strip()
    )
    # On ne garde que les professionnels identifiés (exclut les hôpitaux universitaires)
    df = df[df["covered_recipient_profile_id"].notna()]
    df = df[df["covered_recipient_profile_id"].astype(str).str.len() > 0]
    return df


def quality_audit(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== Audit qualité - {label} ===")
    print(f"  lignes                 : {len(df)}")
    print(f"  professionnels uniques : {df['covered_recipient_profile_id'].nunique()}")
    print(f"  montant manquant       : {df['total_amount_of_payment_usdollars'].isna().sum()}")
    print(f"  montant total (USD)     : {df['total_amount_of_payment_usdollars'].sum():,.0f}")
    print(f"  top natures de paiement : "
          f"{df['nature_of_payment_or_transfer_of_value'].value_counts().head(3).to_dict()}")


# --------------------------------------------------------------------------
# Agrégation : profil d'engagement + cible de rétention
# --------------------------------------------------------------------------
def build_features(df_feat: pd.DataFrame, ids_target: set) -> pd.DataFrame:
    g = df_feat.groupby("covered_recipient_profile_id")
    feats = pd.DataFrame({
        "n_payments": g.size(),
        "total_amount": g["total_amount_of_payment_usdollars"].sum(),
        "mean_amount": g["total_amount_of_payment_usdollars"].mean(),
        "median_amount": g["total_amount_of_payment_usdollars"].median(),
        "n_manufacturers": g["applicable_manufacturer_or_applicable_gpo_making_payment_name"].nunique(),
        "n_natures": g["nature_of_payment_or_transfer_of_value"].nunique(),
        "specialty": g["specialty"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
        "state": g["recipient_state"].first(),
    })
    # Part de la VALEUR (montant) de chaque nature de paiement, pas du volume
    for nature, col in [("Food and Beverage", "share_food"),
                        ("Travel and Lodging", "share_travel"),
                        ("Consulting Fee", "share_consulting"),
                        ("Compensation for services other than consulting, including serving as faculty or as a speaker at a venue other than a continuing education program", "share_speaker"),
                        ("Education", "share_education")]:
        part = df_feat[df_feat["nature_of_payment_or_transfer_of_value"] == nature] \
            .groupby("covered_recipient_profile_id")["total_amount_of_payment_usdollars"].sum()
        feats[col] = (part / feats["total_amount"]).reindex(feats.index).fillna(0.0)
    # Cible : présent (au moins un paiement) l'année suivante
    feats["retenu"] = feats.index.to_series().isin(ids_target).astype(int)
    return feats.reset_index()


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement Open Payments - État {STATE}, années {YEAR_FEATURES} et {YEAR_TARGET}")

    df_n = clean(fetch_state_year(YEAR_FEATURES, STATE))
    df_n1 = clean(fetch_state_year(YEAR_TARGET, STATE))
    quality_audit(df_n, f"{STATE} {YEAR_FEATURES}")
    quality_audit(df_n1, f"{STATE} {YEAR_TARGET}")

    ids_target = set(df_n1["covered_recipient_profile_id"].unique())
    features = build_features(df_n, ids_target)
    taux = features["retenu"].mean()
    print(f"\nProfils {YEAR_FEATURES} : {len(features)} | taux de rétention en {YEAR_TARGET} : {taux:.1%}")

    # Base relationnelle : paiements bruts (deux années) + table de modélisation
    payments = pd.concat([df_n, df_n1], ignore_index=True)
    with sqlite3.connect(DB_PATH) as con:
        payments.to_sql("payments", con, if_exists="replace", index=False)
        features.to_sql("hcp_features", con, if_exists="replace", index=False)
    print(f"\nBase écrite : {DB_PATH}")
    print("  table `payments`     :", len(payments), "lignes")
    print("  table `hcp_features` :", len(features), "profils")


if __name__ == "__main__":
    main()
