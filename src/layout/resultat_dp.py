from shiny import ui
from src.fonctions import (
    calcul_MCG
)
from src.layout.preparer_requetes import make_card_body
import numpy as np
import pandas as pd


def page_resultat_dp():
    return ui.nav_panel(
        "Résultat DP",
        ui.panel_well(
            ui.h4("Résultat des requêtes DP"),
            ui.br(),
            ui.output_ui("req_dp_display")
        )
    )


def afficher_resultats(results_store, requetes, data_query, modalite):
    current_results = results_store()
    panels = []
    final_results = {}
    intermed_results = {}

    query_comptage = {k: v for k, v in data_query.items() if v["type"].lower() in ["count", "comptage"]}
    filtres_uniques = set(v.get("filtre") for v in query_comptage.values())

    for filtre in filtres_uniques:
        query_filtre = {
            k: v for k, v in query_comptage.items()
            if v.get("filtre") == filtre
        }

        results_filtre = {k: v for k, v in current_results.items() if k in query_filtre.keys()}

        results_filtre = calcul_MCG(results_filtre, modalite, query_comptage, "count")

        intermed_results.update(results_filtre)

    query_total = {k: v for k, v in data_query.items() if v["type"].lower() in ["sum", "total"]}
    filtres_uniques = set(v.get("filtre") for v in query_total.values())
    variables_uniques = set(v["variable"] for v in query_total.values())

    for filtre in filtres_uniques:
        for variable in variables_uniques:
            query_filtre = {
                k: v for k, v in query_total.items()
                if v.get("filtre") == filtre and v.get("variable") == variable
            }

            results_filtre = {k: v for k, v in current_results.items() if k in query_filtre.keys()}
            results_filtre = calcul_MCG(results_filtre, modalite, query_filtre, "sum", pos=False)
            intermed_results.update(results_filtre)

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
            L, U = req["bounds"]
            m = (U + L) / 2

            df_result_comptage = intermed_results[key_query_comptage]
            df_result_total = intermed_results[key_query_total]

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
            L, U = req["bounds"]
            m = (U + L) / 2

            df_result_comptage = intermed_results[key_query_comptage]
            df_result_total = intermed_results[key_query_total]

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

            df_result_comptage = intermed_results[key_query_comptage]
            df_result_total_num = intermed_results[key_query_total_num].copy()
            df_result_total_num.rename(columns={"sum": "sum_num"}, inplace=True)
            df_result_total_denom = intermed_results[key_query_total_denom].copy()
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
                df_result = intermed_results[key_query]

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

    return ui.TagList(
        ui.div("📤 Exporter vos résultats respectant la confidentialité différentielle :", class_="mb-2"),
        ui.download_button("download_xlsx", "💾 Télécharger les résultats (XLSX)", class_="btn-outline-primary mb-4"),
        ui.accordion(*panels, open=True)
    )
