"""
03 - Assistant GenAI : text-to-SQL + narrateur d'insights.

Pose une question en langage naturel : l'assistant genere une requete SQL
(SQLite), l'execute sur la base `payments`, puis redige une reponse claire
pour un interlocuteur metier.

Garde-fous : uniquement des SELECT, une seule instruction, LIMIT recommande,
une nouvelle tentative en cas d'erreur SQL. LLM configurable (Ollama / OpenAI).

Usage :
    python src/03_genai_assistant.py "Quel laboratoire depense le plus en repas ?"
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "openpayments.sqlite"

SCHEMA = """
Table `paiements` (un paiement d'un laboratoire a un professionnel de sante) :
- professionnel_id (TEXT) : identifiant du professionnel de sante
- etat (TEXT)             : etat (ex. 'WV')
- specialite (TEXT)       : specialite du professionnel
- laboratoire (TEXT)      : nom du laboratoire payeur
- montant (REAL)          : montant du paiement en USD
- nature (TEXT)           : nature du paiement ('Food and Beverage', 'Travel and Lodging', 'Consulting Fee', ...)
- annee (INTEGER)         : 2022 ou 2023
"""

SQL_PROMPT = (
    "Tu traduis une question en une requete SQL SQLite.\n"
    "{schema}\n"
    "Regles STRICTES :\n"
    "- produis UNIQUEMENT une requete SELECT (jamais INSERT/UPDATE/DELETE/DROP) ;\n"
    "- une seule instruction, sans point-virgule ;\n"
    "- utilise uniquement la table `paiements` et ses colonnes ;\n"
    "- ajoute LIMIT 20 au maximum si la question renvoie une liste ;\n"
    "- reponds avec la requete SQL SEULE, sans texte ni balise Markdown.\n\n"
    "Question : {question}\n"
    "SQL :"
)

NARRATE_PROMPT = (
    "Tu es analyste de donnees commerciales pharma. En 1 a 3 phrases claires,\n"
    "reponds a la question a partir du resultat SQL, pour un interlocuteur metier.\n\n"
    "Question : {question}\n"
    "Resultat :\n{result}\n\n"
    "Reponse :"
)


def get_llm(temperature: float = 0.0):
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)
    from langchain_ollama import ChatOllama

    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=temperature)


def extract_sql(text: str) -> str:
    text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).strip("` \n")
    m = re.search(r"(?is)\bselect\b.*", text)
    sql = m.group(0) if m else text
    # coupe au premier marqueur de fin de requete (le LLM ajoute parfois une
    # explication apres le SQL, dont un ';' qui ferait echouer le garde-fou is_safe)
    sql = re.split(r";|```|\n\s*\n", sql)[0]
    return sql.strip()


def is_safe(sql: str) -> bool:
    low = " " + sql.lower().strip() + " "
    if not low.strip().startswith("select"):
        return False
    if ";" in sql:
        return False
    return not any(k in low for k in ("drop ", "delete ", "update ", "insert ", "alter ", "pragma"))


VIEW_SQL = """
CREATE VIEW IF NOT EXISTS paiements AS
SELECT
    covered_recipient_profile_id AS professionnel_id,
    recipient_state AS etat,
    specialty AS specialite,
    applicable_manufacturer_or_applicable_gpo_making_payment_name AS laboratoire,
    total_amount_of_payment_usdollars AS montant,
    nature_of_payment_or_transfer_of_value AS nature,
    program_year AS annee
FROM payments
"""


def ensure_view() -> None:
    """Cree une vue `paiements` aux noms de colonnes courts et intuitifs (pour le LLM)."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute(VIEW_SQL)


def run_sql(sql: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(sql, con)


def ask(question: str, max_retries: int = 1, verbose: bool = True):
    ensure_view()
    llm = get_llm()
    sql_chain = ChatPromptTemplate.from_template(SQL_PROMPT) | llm | StrOutputParser()

    error, sql, df = None, "", None
    for _ in range(max_retries + 1):
        q = question if error is None else f"{question}\n(La requete precedente a echoue : {error}. Corrige-la.)"
        sql = extract_sql(sql_chain.invoke({"schema": SCHEMA, "question": q}))
        if not is_safe(sql):
            return "Requete refusee (seuls les SELECT sont autorises).", sql, None
        try:
            df = run_sql(sql)
            error = None
            break
        except Exception as exc:  # SQL invalide : on redonne l'erreur au modele
            error = str(exc)
    if error is not None:
        return f"Echec de la generation SQL : {error}", sql, None

    narrate = ChatPromptTemplate.from_template(NARRATE_PROMPT) | llm | StrOutputParser()
    answer = narrate.invoke({"question": question, "result": df.head(20).to_string(index=False)})
    if verbose:
        print("\nSQL genere :\n ", sql)
        print("\nResultat (extrait) :\n", df.head(10).to_string(index=False))
    return answer, sql, df


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Quels sont les 5 laboratoires qui depensent le plus au total, et combien ?"
    answer, sql, df = ask(q)
    print("\nREPONSE :\n", answer)
