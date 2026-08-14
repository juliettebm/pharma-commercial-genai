# Données : CMS Open Payments

Ce projet utilise **CMS Open Payments** (paiements des industriels de santé aux professionnels de santé, données publiques US). Les fichiers bruts ne sont **pas versionnés**.

## Ce qu'il faut récupérer

Deux **années consécutives** de « General Payments » (ex. 2022 et 2023) : la première sert à construire le profil d'engagement, la seconde à définir la cible de rétention (le HCP reçoit-il encore un paiement l'année suivante).

Pour garder un volume raisonnable, on **filtre sur un seul État** (par ex. un État de taille moyenne) et/ou on **échantillonne**.

## Où télécharger

- Portail : **https://openpaymentsdata.cms.gov/** → section *Datasets* → *General Payments* → choisir l'année → *Download*.
- Ou via l'API de données CMS (data.cms.gov) en filtrant par État et par année (utile pour éviter les fichiers de plusieurs Go).

Le script `src/01_prepare_data.py` s'occupera du filtrage (État), de l'échantillonnage, du nettoyage et de la structuration en base SQLite.

## Colonnes clés utilisées

- `Covered_Recipient_Profile_ID` (identifiant du professionnel)
- `Covered_Recipient_Specialty_1` (spécialité)
- `Recipient_State` (État)
- `Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name` (labo payeur)
- `Total_Amount_of_Payment_USDollars` (montant)
- `Nature_of_Payment_or_Transfer_of_Value` (nature : repas, conseil, orateur, formation…)
- `Date_of_Payment`, `Program_Year`

Les noms exacts peuvent varier légèrement selon l'année : le script prévoit une correspondance souple.

## Rappel

Données publiques concernant des personnes réelles : à n'utiliser qu'à des fins d'analyse et de démonstration, dans le respect de la finalité (voir le volet IA responsable du projet).
