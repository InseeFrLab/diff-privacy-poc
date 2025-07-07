from src.process_tools import (
    process_request
)
from src.fonctions import (
    calcul_MCG
)
from shiny import ui
import numpy as np
import pandas as pd

# Fonctions pour du layout ----------------------------------


def afficher_resultats(results_store, requetes, data_query, modalite):
    current_results = results_store()
    panels = []
    final_results = {}

    query_comptage = {k: v for k, v in data_query.items() if v["type"].lower() in ["count", "comptage"]}
    results_comptage = {k: v for k, v in current_results.items() if k in query_comptage.keys()}
    results_comptage = calcul_MCG(results_comptage, modalite, query_comptage, "count")

    query_total = {k: v for k, v in data_query.items() if v["type"].lower() in ["sum", "total"]}
    results_total_par_variable = {}
    variables_uniques = set(v["variable"] for v in query_total.values())
    for variable in variables_uniques:
        query_total_variable = {
                k: v for k, v in query_total.items()
                if v["variable"] == variable
            }

        results_total_variable = {k: v for k, v in current_results.items() if k in query_total_variable.keys()}
        results_total_variable = calcul_MCG(results_total_variable, modalite, query_total_variable, "sum", pos=False)
        results_total_par_variable[variable] = results_total_variable

    for key, req in requetes.items():

        if req["type"] == "Total":
            key_query_comptage = next(
                (k for k, v in data_query.items() if key in v["req"] and v["type"] == "Comptage"),
                None  # valeur par défaut si rien n'est trouvé
            )
            key_query_total = next(
                (k for k, v in data_query.items() if key in v["req"] and v["type"] == "Total"),
                None
            )
            variable = req["variable"]
            L, U = req["bounds"]
            m = (U + L) / 2

            df_result_comptage = results_comptage[key_query_comptage]
            df_result_total = results_total_par_variable[variable][key_query_total]

            # On concatène horizontalement sur l’index (corrigé)
            df_result = pd.concat(
                [df_result_total.reset_index(drop=True),
                df_result_comptage.reset_index(drop=True)],
                axis=1
            )

            # Supprimer les colonnes en doublon éventuelles
            df_result = df_result.loc[:, ~df_result.columns.duplicated()]

            df_result["sum"] = df_result["sum"] + df_result["count"] * m

        elif req["type"] == "Moyenne":
            key_query_comptage = next(
                (k for k, v in data_query.items() if key in v["req"] and v["type"] == "Comptage"),
                None  # valeur par défaut si rien n'est trouvé
            )
            key_query_total = next(
                (k for k, v in data_query.items() if key in v["req"] and v["type"] == "Total"),
                None
            )
            variable = req["variable"]
            L, U = req["bounds"]
            m = (U + L) / 2

            df_result_comptage = results_comptage[key_query_comptage]
            df_result_total = results_total_par_variable[variable][key_query_total]

            # On concatène horizontalement sur l’index (corrigé)
            df_result = pd.concat(
                [df_result_total.reset_index(drop=True),
                df_result_comptage.reset_index(drop=True)],
                axis=1
            )

            # Supprimer les colonnes en doublon éventuelles
            df_result = df_result.loc[:, ~df_result.columns.duplicated()]

            df_result["sum"] = df_result["sum"] + df_result["count"] * m

            # Calcul de la moyenne
            df_result["mean"] = df_result.apply(
                lambda row: np.inf if row["count"] == 0 else row["sum"] / row["count"],
                axis=1
            )

        elif req["type"] == "Ratio":
            key_query_comptage = next(
                (k for k, v in data_query.items() if key in v["req"] and v["type"] == "Comptage"),
                None  # valeur par défaut si rien n'est trouvé
            )
            variable_num = req["variable"]
            variable_denom = req["variable_denominateur"]

            L, U = req["bounds"]
            m_num = (U + L) / 2

            L, U = req["bounds_denominateur"]
            m_denom = (U + L) / 2

            key_query_total_num = next(
                (k for k, v in data_query.items() if key in v["req"] and v["type"] == "Total" and v["variable"] == variable_num),
                None
            )
            key_query_total_denom = next(
                (k for k, v in data_query.items() if key in v["req"] and v["type"] == "Total" and v["variable"] == variable_denom),
                None
            )

            df_result_comptage = results_comptage[key_query_comptage]
            df_result_total_num = results_total_par_variable[variable_num][key_query_total_num].copy()
            df_result_total_num.rename(columns={"sum": "sum_num"}, inplace=True)
            df_result_total_denom = results_total_par_variable[variable_denom][key_query_total_denom].copy()
            df_result_total_denom.rename(columns={"sum": "sum_denom"}, inplace=True)

            # On concatène horizontalement sur l’index (corrigé)
            df_result = pd.concat(
                [df_result_total_num.reset_index(drop=True),
                df_result_total_denom.reset_index(drop=True),
                df_result_comptage.reset_index(drop=True)],
                axis=1
            )

            # Supprimer les colonnes en doublon éventuelles
            df_result = df_result.loc[:, ~df_result.columns.duplicated()]

            df_result["sum_num"] = df_result["sum_num"] + df_result["count"] * m_num
            df_result["sum_denom"] = df_result["sum_denom"] + df_result["count"] * m_denom

            # Calcul de la moyenne
            df_result["ratio"] = df_result.apply(
                lambda row: np.inf if row["sum_denom"] == 0 else row["sum_num"] / row["sum_denom"],
                axis=1
            )

        else:
            key_query = next((k for k, v in data_query.items() if key in v["req"]), None)

            if req["type"] == "Comptage":
                df_result = results_comptage[key_query]

            if req["type"] == "Quantile":
                df_result = current_results[key_query]

        df_result = df_result.round(1)

        # Remplace -0.0 par 0.0 dans toutes les colonnes numériques à virgule
        for col in df_result.select_dtypes(include=["float"]).columns:
            df_result[col] = df_result[col].apply(lambda x: 0.0 if x == -0.0 else x)

        final_results[key] = df_result

        param_card = ui.card(
            ui.card_header("Paramètres"),
            make_card_body(req)
        )

        result_card = ui.card(
            ui.card_header("Résultats après application de la DP"),
            ui.HTML(df_result.to_html(
                classes="table table-striped table-hover table-sm text-center align-middle",
                border=0,
                index=False
            )),
            height="300px",
            fillable=False,
            full_screen=True
        )

        content_row = ui.row(
            ui.column(4, param_card),
            ui.column(8, result_card)
        )

        panels.append(
            ui.accordion_panel(f"{key} — {req.get('type', '—')}", content_row, open=True)
        )

    results_store.set(final_results)

    return ui.accordion(*panels, open=True)


def make_radio_buttons(request, filter_type: list[str]):
    radio_buttons = []
    priorite = {"Comptage": "2", "Total": "2", "Moyenne": "1", "Ratio": "1","Quantile": "3"}
    for key, req in request.items():
        if req["type"] in filter_type:
            radio_buttons_id = key
            radio_buttons.append(
                ui.input_radio_buttons(radio_buttons_id, key,
                    {"1": 1, "2": 2, "3": 3}, selected=priorite[req["type"]]
                )
            )
    return radio_buttons


def make_card_body(req):
    parts = []

    fields = [
        ("variable", "📌 Variable", lambda v: f"`{v}`"),
        ("bounds", "🎯 Bornes", lambda v: f"`[{v[0]}, {v[1]}]`" if isinstance(v, list) and len(v) == 2 else "—"),
        ("by", "🧷 Group by", lambda v: f"`{', '.join(v)}`"),
        ("filtre", "🧮 Filtre", lambda v: f"`{v}`"),
        ("alpha", "📈 Alpha", lambda v: f"`{v}`"),
    ]

    for key, label, formatter in fields:
        val = req.get(key)
        if val is not None and val != "" and val != []:
            parts.append(ui.p(f"{label} : {formatter(val)}"))

    return ui.card_body(*parts)


def affichage_requete(requetes, dataset):

    panels = []
    df = dataset.lazy()

    for key, req in requetes.items():
        # Colonne de gauche : paramètres
        resultat = process_request(df, req, use_bounds=False)

        if req.get("by") is not None:
            resultat = resultat.sort(by=req.get("by"))

        param_card = ui.card(
            ui.card_header("Paramètres"),
            make_card_body(req)
        )

        # Colonne de droite : table / placeholder
        result_card = ui.card(
            ui.card_header("Résultats sans application de la DP (à titre indicatif)"),
            ui.tags.style("""
                .table {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size: 0.9rem;
                    box-shadow: 0 0 10px rgba(0,0,0,0.05);
                    border-radius: 0.25rem;
                    border-collapse: collapse;
                    width: 100%;
                    border: 1px solid #dee2e6;
                }
                .table-hover tbody tr:hover {
                    background-color: #f1f1f1;
                }
                .table-striped tbody tr:nth-of-type(odd) {
                    background-color: #fafafa;
                }
                table.table thead th {
                    background-color: #f8f9fa !important;
                    font-weight: 700 !important;
                    border-left: 1px solid #dee2e6;
                    border-right: 1px solid #dee2e6;
                    border-bottom: 2px solid #dee2e6;
                    padding: 0.3rem 0.6rem;
                    vertical-align: middle !important;
                    text-align: center;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }
                tbody td {
                    padding: 0.3rem 0.6rem;
                    vertical-align: middle !important;
                    text-align: center;
                    border-left: 1px solid #dee2e6;
                    border-right: 1px solid #dee2e6;
                }
                thead th:first-child, tbody td:first-child {
                    border-left: none;
                }
                thead th:last-child, tbody td:last-child {
                    border-right: none;
                }
            """),
            ui.HTML(resultat.to_pandas().to_html(
                classes="table table-striped table-hover table-sm text-center align-middle",
                border=0,
                index=False
            )),
            height="300px",
            fillable=False,
            full_screen=True
        ),

        # Ligne avec deux colonnes côte à côte
        content_row = ui.row(
            ui.column(4, param_card),
            ui.column(8, result_card)
        )

        # Panneau d'accordéon contenant la ligne
        panels.append(
            ui.accordion_panel(f"{key} — {req.get('type', '—')}", content_row)
        )

    return ui.accordion(*panels, open=True)
