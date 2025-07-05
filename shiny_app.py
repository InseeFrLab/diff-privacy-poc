# Imports
from src.plots import (
    create_histo_plot, create_fc_emp_plot,
    create_score_plot, create_proba_plot,
    create_barplot,
    plot_subset_tree
)
from src.layout.fonction_layout import (
    make_radio_buttons,
    affichage_requete,
    afficher_resultats
)
from src.layout.introduction_dp import page_introduction_dp
from src.layout.donnees import page_donnees
from src.layout.preparer_requetes import page_preparer_requetes
from src.layout.conception_budget import page_conception_budget
from src.layout.resultat_dp import page_resultat_dp
from src.layout.etat_budget_dataset import page_etat_budget_dataset
from src.process_tools import (
    process_request, calculer_toutes_les_requetes
)
from src.fonctions import (
    eps_from_rho_delta, optimization_boosted,
    update_context,
    get_weights, intervalle_confiance_quantile,
    load_data,
    generate_yaml_metadata_from_lazyframe_as_string
)
from src.constant import (
    storage_options,
    contrib_individu,
    borne_max_taille_dataset
)

from shiny import App, ui, render, reactive
from shinywidgets import render_widget
from pathlib import Path
from datetime import datetime
from scipy.stats import norm

import seaborn as sns
import opendp.prelude as dp
import numpy as np
import pandas as pd
import polars as pl
import io
import json
import yaml
from typing import Any

dp.enable_features("contrib")

www_dir = Path(__file__).parent / "www"

data_example = sns.load_dataset("penguins").dropna()


# 1. UI --------------------------------------
app_ui = ui.page_navbar(
    ui.nav_spacer(),
    page_introduction_dp(),
    page_donnees(),
    page_preparer_requetes(),
    page_conception_budget(),
    page_resultat_dp(),
    page_etat_budget_dataset(),
    title=ui.div(
        ui.img(src="insee-logo.png", height="80px", style="margin-right:10px"),
        ui.img(src="Logo_poc.png", height="60px", style="margin-right:10px"),
        style="display: flex; align-items: center; gap: 10px;"
    ),
    id="page",
)

# 2. Server ----------------------------------


def server(input, output, session):

    requetes = reactive.Value({})
    page_autorisee = reactive.Value(False)
    resultats_df = reactive.Value({})
    onglet_actuel = reactive.Value("Conception du budget")  # Onglet par défaut
    trigger_update_budget = reactive.Value(0)

    @reactive.Calc
    def get_poids_req() -> dict[str, float]:
        poids = get_weights(requetes(), input)
        return poids

    @reactive.Calc
    def dict_query() -> dict[str, dict[str, Any]]:
        data_requetes = requetes()
        req_non_moyenne = {k: v for k, v in data_requetes.items() if v["type"].lower() not in ["moyenne", "mean"]}
        req_moyenne = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["moyenne", "mean"]}
        query = {}
        i = 1
        for (key, request) in req_non_moyenne.items():
            query_request = request.copy()
            query_request["req"] = [key]
            cle = f"query_{i}"
            query[cle] = query_request
            i += 1

        for (key, request) in req_moyenne.items():
            query_request = request.copy()
            # Cas 1 : Total
            query_request["type"] = "Total"

            # Chercher s'il existe une requête identique dans query, en ignorant "req"
            found = False
            for k, v in query.items():
                v_without_req = {kk: vv for kk, vv in v.items() if kk != "req"}
                if v_without_req == query_request:
                    query[k]["req"].append(key)
                    found = True
                    break

            # Si aucune requête équivalente n'a été trouvée, on ajoute une nouvelle entrée
            if not found:
                query_request["req"] = [key]
                cle = f"query_{i}"
                query[cle] = query_request
                i += 1

            # Cas 2 : Comptage
            query_request = request.copy()
            query_request["type"] = "Comptage"
            query_request.pop("variable", None)
            query_request.pop("bounds", None)

            # Chercher s'il existe une requête identique dans query, en ignorant "req"
            found = False
            for k, v in query.items():
                v_without_req = {kk: vv for kk, vv in v.items() if kk != "req"}
                if v_without_req == query_request:
                    query[k]["req"].append(key)
                    found = True
                    break

            # Si aucune requête équivalente n'a été trouvée, on ajoute une nouvelle entrée
            if not found:
                query_request["req"] = [key]
                cle = f"query_{i}"
                query[cle] = query_request
                i += 1

        # Première passe : création des entrées avec poids brut
        for params in query.values():
            if 'by' not in params:
                groupement = frozenset()
                groupement_style = 'Aucun'
            else:
                by = params['by']
                if isinstance(by, str):
                    groupement = frozenset([by])
                    groupement_style = by
                elif isinstance(by, list):
                    groupement = frozenset(by)
                    groupement_style = by[0] if len(by) == 1 else tuple(by)

            params["groupement"] = groupement
            params["groupement_style"] = groupement_style

            # Récupération des noms de requêtes associés
            reqs = params.get('req', [])

            # Calcul du poids total
            poids_total = sum(get_poids_req().get(r, 0) for r in reqs)

            params["poids"] = poids_total

            if params["type"] == "Comptage":
                params["sigma2"] = 1/(2 * input.budget_total() * params["poids"])

            if params["type"] == "Total":
                L, U = params["bounds"]
                params["sigma2"] = max(U**2, L**2)/(2 * input.budget_total() * params["poids"])
        return query

    @reactive.Calc
    def conception_query_count() -> dict[str, dict[str, Any]]:
        data_query = dict_query()
        query_comptage = {k: v for k, v in data_query.items() if v["type"].lower() in ["count", "comptage"]}
        query_comptage = optimization_boosted(dict_query=query_comptage, modalite=key_values())
        return query_comptage

    @reactive.Calc
    def df_comptage() -> pd.DataFrame:
        query_comptage = conception_query_count()
        data_requetes = requetes()
        req_comptage = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["count", "comptage"]}
        results = []
        for query in query_comptage.values():

            for req in query["req"]:

                if req in req_comptage:

                    results.append({
                        "requête": req,
                        "groupement": query["groupement_style"],
                        "écart type estimation": query["scale"],
                        "écart type bruit": np.sqrt(query["sigma2"])
                    })

        return pd.DataFrame(results).round(1)

    @output
    @render.data_frame
    def table_comptage():
        return df_comptage()

    @render_widget
    def plot_comptage():
        return create_barplot(df_comptage(), x_col="requête", y_col="écart type estimation", hoover="groupement", color="groupement")

    @render.plot
    def graphe_plot():
        req_comptage_croisement = conception_query_count()
        liste_request = [request['groupement'] for request in req_comptage_croisement.values()]
        return plot_subset_tree(liste_request, taille_noeuds=2000, keep_gris=False)

    @reactive.Calc
    def conception_query_sum() -> dict[str, dict[str, Any]]:
        data_query = dict_query()

        query_total = {k: v for k, v in data_query.items() if v["type"].lower() in ["total", "sum", "somme"]}
        query_total_par_variable = {}

        # Récupération des variables distinctes
        variables_uniques = set(v["variable"] for v in query_total.values())

        # Boucle sur chaque variable unique
        for variable in variables_uniques:
            # Sous-ensemble des requêtes ayant cette variable
            query_total_variable = {
                k: v for k, v in query_total.items()
                if v["variable"] == variable
            }

            query_total_variable = optimization_boosted(dict_query=query_total_variable, modalite=key_values())
            query_total_par_variable[variable] = query_total_variable

        return query_total_par_variable

    @reactive.Calc
    def df_total() -> pd.DataFrame:
        query_total_par_variable = conception_query_sum()
        data_requetes = requetes()
        req_total = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["total", "sum", "somme"]}

        results = []
        for variable, query_total_variable in query_total_par_variable.items():
            for query in query_total_variable.values():
                for req in query["req"]:
                    if req in req_total:
                        scale = query["scale"]

                        label = f"{variable}<br>groupement: {query["groupement_style"]}"

                        resultat = process_request(dataset().lazy(), query)
                        resultat_non_biaise = process_request(dataset().lazy(), query, use_bounds=False)

                        list_cv = []
                        list_biais_relatif = []

                        for row_biaise, row_non_biaise in zip(resultat.iter_rows(named=True), resultat_non_biaise.iter_rows(named=True)):

                            # Calcul du CV
                            cv = 100 * scale / row_biaise["sum"] if row_biaise["sum"] != 0 else float("inf")
                            biais = row_biaise["sum"] - row_non_biaise["sum"]
                            biais_relatif = 100 * biais / row_non_biaise["sum"]

                            list_cv.append(cv)
                            list_biais_relatif.append(biais_relatif)

                        cv_moyen = np.mean(list_cv)
                        biais_relatif_moyen = np.mean(list_biais_relatif)

                        results.append({
                            "requête": req,
                            "label": label,
                            "variable": variable,
                            "groupement": query["groupement_style"],
                            "cv moyen (%)": cv_moyen,
                            "biais relatif moyen (%)": biais_relatif_moyen,
                            "écart type estimation": scale,
                            "écart type bruit": np.sqrt(query["sigma2"])

                        })
        return pd.DataFrame(results).round(1)

    @output
    @render.data_frame
    def table_total():
        df = df_total().drop(columns="label").round(0)
        return df

    @render_widget
    def plot_total():
        return create_barplot(df_total(), x_col="requête", y_col="cv moyen (%)", hoover="label", color="groupement")

    @reactive.Calc
    def df_moyenne() -> dict[str, dict[str, Any]]:
        query_comptage = conception_query_count()
        query_total_par_variable = conception_query_sum()
        data_requetes = requetes()
        req_moyenne = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["moyenne", "mean"]}

        results = []

        for key, req in req_moyenne.items():

            for query in query_comptage.values():

                if key in query["req"]:

                    scale_comptage = query["scale"]
                    break

            variable = req["variable"]

            for query in query_total_par_variable[variable].values():

                if key in query["req"]:

                    scale_total = query["scale"]

                    label = f"{variable}<br>groupement: {query["groupement_style"]}"

                    resultat = process_request(dataset().lazy(), req)
                    resultat_non_biaise = process_request(dataset().lazy(), req, use_bounds=False)

                    list_cv = []
                    list_biais_relatif = []

                    for row_biaise, row_non_biaise in zip(resultat.iter_rows(named=True), resultat_non_biaise.iter_rows(named=True)):
                        total = row_biaise.get("sum", 0)
                        total_non_biaise = row_non_biaise.get("sum", 0)
                        count = row_biaise.get("count", 1)

                        cv_total = scale_total / total if total != 0 else float("inf")
                        cv_count = scale_comptage / count if count != 0 else float("inf")
                        cv = 100 * np.sqrt(cv_total**2 + cv_count**2)

                        biais = (total - total_non_biaise) / count
                        biais_relatif = 100 * biais / (total_non_biaise/count)

                        list_cv.append(cv)
                        list_biais_relatif.append(biais_relatif)

                    cv_moyen = np.mean(list_cv)
                    biais_relatif_moyen = np.mean(list_biais_relatif)

                    results.append({
                        "requête": key,
                        "label": label,
                        "variable": variable,
                        "groupement": query["groupement_style"],
                        "cv moyen (%)": cv_moyen,
                        "biais relatif moyen (%)": biais_relatif_moyen,
                        "écart type total": scale_total,
                        "écart type comptage": scale_comptage
                    })

                    break

        return pd.DataFrame(results).round(1)

    @output
    @render.data_frame
    def table_moyenne():
        df = df_moyenne().drop(columns="label")
        # Définir l'arrondi spécifique
        arrondi = {col: 1 if col == "écart type comptage" else 0 for col in df.columns}
        df = df.round(arrondi)
        return df

    @render_widget
    def plot_moyenne():
        return create_barplot(df_moyenne(), x_col="requête", y_col="cv moyen (%)", hoover="label", color="groupement")

    @reactive.Calc
    def conception_query_quantile() -> dict[str, dict[str, Any]]:

        data_query = dict_query()
        query_quantile = {k: v for k, v in data_query.items() if v["type"].lower() in ["quantile"]}
        query_quantile_par_variable = {}

        variables_uniques = set(v["variable"] for v in query_quantile.values())

        # Boucle sur chaque variable unique
        for variable in variables_uniques:
            # Sous-ensemble des requêtes ayant cette variable
            query_quantile_variable = {
                k: v for k, v in query_quantile.items()
                if v["variable"] == variable
            }

            for key_query, query in query_quantile_variable.items():

                budget_req_quantile = input.budget_total() * query["poids"]
                epsilon = np.sqrt(8 * budget_req_quantile)

                ic = intervalle_confiance_quantile(dataset(), query, epsilon)

                query_quantile_variable[key_query]["scale"] = ic

            query_quantile_par_variable[variable] = query_quantile_variable

        return query_quantile_par_variable

    @reactive.Calc
    def df_quantile() -> pd.DataFrame:
        query_quantile_par_variable = conception_query_quantile()

        results = []
        for variable, query_quantile_variable in query_quantile_par_variable.items():
            for query in query_quantile_variable.values():

                label = f"{variable}<br>groupement: {query["groupement_style"]}"

                results.append({
                    "requête": query["req"][0],
                    "label": label,
                    "variable": variable,
                    "groupement": query["groupement_style"],
                    "taille moyenne IC 95%": query["scale"]
                })
        return pd.DataFrame(results).round(1)

    @output
    @render.data_frame
    def table_quantile():
        df = df_quantile().drop(columns="label").round(0)
        return df

    @render_widget
    def plot_quantile():
        return create_barplot(df_quantile(), x_col="requête", y_col="taille moyenne IC 95%", hoover="label", color="groupement")

    @output
    @render.ui
    @reactive.event(input.confirm_validation)
    async def req_dp_display():
        # --- Barre de progression ---
        data_query = dict_query()

        with ui.Progress(min=0, max=len(data_query)) as p:
            p.set(0, message="Traitement en cours...", detail="Analyse requête par requête...")

            context_param = {
                "data": dataset().lazy(),
                "privacy_unit": dp.unit_of(contributions=contrib_individu),
                "margins": [dp.polars.Margin(max_partition_length=borne_max_taille_dataset)],
            }

            context_rho, context_eps = update_context(context_param, input.budget_total(), data_query)
            await calculer_toutes_les_requetes(context_rho, context_eps, key_values(), data_query, p, resultats_df)

        return afficher_resultats(resultats_df, requetes(), data_query, key_values())


    # Page 1 ----------------------------------

    # Lire le dataset si importé sinon dataset déjà en mémoire
    @reactive.Calc
    def dataset() -> pl.DataFrame:
        file = input.dataset_input()
        if file is not None:
            ext = Path(file["name"]).suffix
            if ext == ".csv":
                return pl.read_csv(file["datapath"])
            elif ext == ".parquet":
                return load_data(file["datapath"], storage_options)
            else:
                raise ValueError("Format non supporté : utiliser CSV ou Parquet")
        else:
            if input.default_dataset() == "penguins":
                return pl.DataFrame(sns.load_dataset(input.default_dataset()).dropna())
            else:
                return load_data(input.default_dataset(), storage_options)


    @reactive.Calc
    def yaml_metadata_str():
        return generate_yaml_metadata_from_lazyframe_as_string(dataset(), dataset_name="my_dataset")

    # Afficher le dataset
    @output
    @render.data_frame
    def data_view():
        return dataset().head(1000)

    # Afficher les métadata
    @output
    @render.text
    def meta_data():
        return ui.tags.pre(yaml_metadata_str())

    # Page 2 ----------------------------------

    # Liste les variables qualitatives et quatitatives du jeu de données actuel
    @reactive.Calc
    def variable_choices():
        df = dataset()
        if df is None:
            return {}

        qualitative = [
            col for col, dtype in zip(df.columns, df.dtypes)
            if dtype in (pl.Utf8, pl.Categorical, pl.Boolean)
        ]

        quantitative = [
            col for col, dtype in zip(df.columns, df.dtypes)
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64)
        ]
        return {
            "": "",
            "🔤 Qualitatives": {col: col for col in qualitative},
            "🧮 Quantitatives": {col: col for col in quantitative}
        }

    # Extrait les modalités uniques des variavles qualitatives
    @reactive.Calc
    def key_values():
        df = dataset()

        # Détecter les colonnes qualitatives (str ou catégorie)
        qualitatif_cols = [
            col for col, dtype in zip(df.columns, df.dtypes)
            if dtype in [pl.Utf8, pl.Categorical, pl.Boolean]
        ]

        # Extraire les modalités uniques, triées, sans NaN
        return {
            col: sorted(df[col].drop_nulls().unique().to_list())
            for col in qualitatif_cols
        }

    # Extrait les nombres de modalités des variables qualitatives
    @reactive.Calc
    def nb_modalite_var():
        metadata_dict = yaml.safe_load(yaml_metadata_str()) if yaml_metadata_str() else {}
        if 'columns' not in metadata_dict:
            return {}

        qualitative_cols = [
            col for col, meta in metadata_dict['columns'].items()
            if meta.get('type') in ('Utf8', 'Categorical', 'Boolean', 'str', 'String')
        ]

        nb_modalites = {}
        for col in qualitative_cols:
            nb = metadata_dict['columns'][col].get('unique_values', None)
            if nb is not None:
                nb_modalites[col] = nb
        return nb_modalites

    @reactive.Effect
    def update_variable_choices():
        ui.update_selectize("group_by", choices=variable_choices())
        if input.type_req() == "Comptage":
            ui.update_selectize("variable", choices={})  # pas de choix possible
        else:
            # Met à jour dynamiquement les choix de la selectize input
            ui.update_selectize("variable", choices=variable_choices())

    # Lecture du json contenant les requêtes
    @reactive.effect
    @reactive.event(input.request_input)
    def _():
        fileinfo = input.request_input()
        if fileinfo is not None:
            filepath = Path(fileinfo[0]["datapath"])
            try:
                with filepath.open(encoding="utf-8") as f:
                    data = json.load(f)
                requetes.set(data)
                ui.update_selectize("delete_req", choices=list(data.keys()))
            except json.JSONDecodeError:
                ui.notification_show("❌ Fichier JSON invalide", type="error")

    # Téléchargement du json contenant les requêtes
    @output
    @render.download(filename=lambda: "requetes_exportees.json")
    def download_json():
        buffer = io.StringIO()
        json.dump(requetes(), buffer, indent=2, ensure_ascii=False)
        buffer.seek(0)
        return buffer

    @reactive.effect
    @reactive.event(input.add_req)
    def _():
        current = requetes().copy()

        # Charger les métadonnées YAML dans un dict (tu peux le faire une fois dans un Calc sinon)
        metadata_dict = yaml.safe_load(yaml_metadata_str()) if yaml_metadata_str() else {}

        variable = input.variable()
        bounds = None
        # Essayer de récupérer min/max dans YAML pour la variable choisie
        if 'columns' in metadata_dict and variable in metadata_dict['columns']:
            col_meta = metadata_dict['columns'][variable]
            min_val = col_meta.get('min')
            max_val = col_meta.get('max')
            if min_val is not None and max_val is not None:
                bounds = [float(min_val), float(max_val)]

        base_dict = {
            "type": input.type_req(),
            "variable": variable,
            "bounds": bounds,
            "by": sorted(input.group_by()),  # tri pour éviter doublons
            "filtre": input.filtre(),
        }

        if input.type_req() == 'Quantile':
            base_dict.update({
                "alpha": float(input.alpha()),
                "nb_candidats": input.nb_candidats(),
            })

        clean_dict = {
            k: v for k, v in base_dict.items()
            if v not in [None, "", (), ["", ""], []]
        }

        if any(
            existing_req.get("type") == clean_dict.get("type") and
            existing_req.get("variable") == clean_dict.get("variable") and
            existing_req.get("bounds") == clean_dict.get("bounds") and
            existing_req.get("by", []) == clean_dict.get("by", []) and
            existing_req.get("filtre") == clean_dict.get("filtre") and
            (
                input.type_req() != "Quantile" or (
                    existing_req.get("alpha") == clean_dict.get("alpha") and
                    existing_req.get("nb_candidats") == clean_dict.get("nb_candidats")
                )
            )
            for existing_req in current.values()
        ):
            ui.notification_show("❌ Requête déjà existante (mêmes paramètres)", type="error")
            return

        i = 1
        while f"req_{i}" in current:
            i += 1
        new_id = f"req_{i}"

        current[new_id] = clean_dict
        requetes.set(current)
        ui.notification_show(f"✅ Requête `{new_id}` ajoutée", type="message")
        ui.update_selectize("delete_req", choices=list(requetes().keys()))

    @reactive.effect
    @reactive.event(input.delete_btn)
    def _():
        current = requetes().copy()
        targets = input.delete_req()  # ceci est une liste ou un tuple de valeurs

        if not targets:
            ui.notification_show("❌ Aucune requête sélectionnée", type="error")
            return

        removed = []
        not_found = []
        for target in targets:
            if target in current:
                del current[target]
                removed.append(target)
            else:
                not_found.append(target)

        requetes.set(current)

        if removed:
            ui.notification_show(f"🗑️ Requête(s) supprimée(s) : {', '.join(removed)}", type="warning")
            ui.update_selectize("delete_req", choices=list(requetes().keys()))
        if not_found:
            ui.notification_show(f"❌ Requête(s) introuvable(s) : {', '.join(not_found)}", type="error")

    @reactive.effect
    @reactive.event(input.delete_all_btn)
    def _():
        current = requetes().copy()
        if current:
            current.clear()  # Vide toutes les requêtes
            requetes.set(current)  # Met à jour le reactive.Value
            ui.notification_show(f"🗑️ TOUTES les requêtes ont été supprimé", type="warning")
            ui.update_selectize("delete_req", choices=list(requetes().keys()))

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def req_display():
        data_requetes = requetes()
        if not data_requetes:
            return ui.p("Aucune requête chargée.")
        return affichage_requete(data_requetes, dataset())


    # Page 3 ----------------------------------

    @render.ui
    def interval_summary():
        sigma = input.scale_gauss()
        quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
        result_lines = []
        for q in quantiles:
            z = norm.ppf(0.5 + q / 2)
            bound = round(z * sigma, 3)
            result_lines.append(f"<li><strong>{int(q * 100)}%</strong> de chances que le bruit soit entre +/- <code>{round(bound,1)}</code></li>")

        return ui.HTML("""
            <div style='margin-top:20px; padding:10px; background-color:#f9f9f9; border-radius:12px; font-family: "Raleway", "Garamond", sans-serif; font-size:16px; color:#333'>
                <p style="margin-bottom:10px"><strong>Résumé des intervalles de confiance :</strong></p>
                <ul style="padding-left: 20px; margin: 0;">
                    {}
                </ul>
            </div>
        """.format("".join(result_lines)))

    @output
    @render.data_frame
    def cross_table():
        table = data_example.groupby(["species", "island"]).size().unstack(fill_value=0)
        flat_table = table.reset_index().melt(id_vars="species", var_name="island", value_name="count")
        return flat_table.sort_values(by=["species", "island"])

    @output
    @render.data_frame
    @reactive.event(input.scale_gauss)
    def cross_table_dp():
        # Table originale sans bruit
        table = data_example.groupby(["species", "island"]).size().unstack(fill_value=0)
        flat_table = table.reset_index().melt(id_vars="species", var_name="island", value_name="count")
        flat_table = flat_table.sort_values(by=["species", "island"]).reset_index(drop=True)

        # Ajout de bruit gaussien à la colonne 'count'
        sigma = input.scale_gauss()
        flat_table["count"] = flat_table["count"] + np.random.normal(loc=0, scale=sigma, size=len(flat_table))

        # Optionnel : arrondir ou tronquer selon les besoins
        flat_table["count"] = flat_table["count"].round(0).clip(lower=0).astype(int)

        return flat_table

    @output
    @render.ui
    def dp_budget_summary():
        rho = 1 / (2 * input.scale_gauss() ** 2)
        delta_exp = input.delta_slider()
        delta = f"1e{delta_exp}"
        eps = eps_from_rho_delta(rho, 10**delta_exp)

        return ui.HTML(f"""
            <div style='margin-top:20px; padding:10px; background-color:#f9f9f9; border-radius:12px;
                        font-family: "Raleway", "Garamond", sans-serif; font-size:16px; color:#333'>
                <p style="margin-bottom:10px"><strong>Budget de confidentialité différentielle :</strong></p>
                <ul style="padding-left: 20px; margin: 0;">
                    <li>En zCDP, <strong>ρ</strong> = <code>{rho:.4f}</code></li>
                    <li>En Approximate DP, (<strong>ε</strong> = <code>{eps:.3f}</code>, <strong>δ</strong> = <code>{delta}</code>)</li>
                </ul>
            </div>
        """)

    @render.plot
    def histo_plot():
        return create_histo_plot(data_example, input.alpha_slider())

    @render.plot
    def fc_emp_plot():
        return create_fc_emp_plot(data_example, input.alpha_slider())

    @render_widget
    def score_plot():
        candidat_min, candidat_max = input.candidat_slider()
        return create_score_plot(
            data_example, input.alpha_slider(), input.epsilon_slider(),
            candidat_min, candidat_max, input.candidat_step()
        )

    @render_widget
    def proba_plot():
        candidat_min, candidat_max = input.candidat_slider()
        return create_proba_plot(
            data_example, input.alpha_slider(), input.epsilon_slider(),
            candidat_min, candidat_max, input.candidat_step()
        )

    @reactive.calc
    def budgets_par_dataset():
        _ = trigger_update_budget()
        df = pd.read_csv("data/budget_dp.csv")

        # somme budget pour France entière par dataset
        df_france = df[df["echelle_geographique"] == "France entière"].groupby("nom_dataset", as_index=False)["budget_dp_rho"].sum()
        df_france = df_france.rename(columns={"budget_dp_rho": "budget_france"})

        # somme budget pour chaque autre échelle par dataset
        df_autres = df[df["echelle_geographique"] != "France entière"].groupby(["nom_dataset", "echelle_geographique"], as_index=False)["budget_dp_rho"].sum()

        # pour chaque dataset, on prend la valeur max des sommes sur les autres échelles
        df_max_autres = df_autres.groupby("nom_dataset", as_index=False)["budget_dp_rho"].max()
        df_max_autres = df_max_autres.rename(columns={"budget_dp_rho": "budget_max_autres"})

        # merge budgets France entière et max autres échelles (outer pour ne rien perdre)
        df_merge = pd.merge(df_france, df_max_autres, on="nom_dataset", how="outer").fillna(0)

        # somme finale
        df_merge["budget_dp_rho"] = df_merge["budget_france"] + df_merge["budget_max_autres"]

        # on trie et on ne garde que ce qui nous intéresse
        df_result = df_merge[["nom_dataset", "budget_dp_rho"]].sort_values("budget_dp_rho", ascending=False)

        return df_result

    @output
    @render.ui
    def budget_display():
        df_grouped = budgets_par_dataset()

        boxes = []
        for _, row in df_grouped.iterrows():
            boxes.append(
                ui.value_box(
                    title=row["nom_dataset"],
                    value=f"{row['budget_dp_rho']:.3f}"
                )
            )

        # Regrouper les value boxes en lignes de 4 colonnes max
        rows = []
        for i in range(0, len(boxes), 4):
            row = ui.row(*[ui.column(3, box) for box in boxes[i:i+4]])
            rows.append(row)

        return ui.div(*rows)

    @output
    @render.data_frame
    def data_budget_view():
        _ = trigger_update_budget()
        return pd.read_csv("data/budget_dp.csv")

    @reactive.Effect
    @reactive.event(input.confirm_validation)
    def _():

        data_requetes = requetes()

        if len(data_requetes) == 0:
            ui.notification_show(f"❌ Vous devez rentrer au moins une requête avant d'accéder aux résultats.", type="error")

        elif input.budget_total() == 0:
            ui.notification_show(f"❌ Vous devez valider un budget non nul avant d'accéder aux résultats.", type="error")

        else:
            page_autorisee.set(True)
            ui.modal_remove()
            ui.update_navs("page", selected="Résultat DP")

            nouvelle_ligne = pd.DataFrame([{
                "nom_dataset": input.dataset_name(),
                "echelle_geographique": input.echelle_geo(),
                "date_ajout": datetime.now().strftime("%d/%m/%Y"),
                "budget_dp_rho": input.budget_total()
            }])

            fichier = Path("data/budget_dp.csv")
            if fichier.exists():
                nouvelle_ligne.to_csv(fichier, mode="a", header=False, index=False, encoding="utf-8")
            else:
                nouvelle_ligne.to_csv(fichier, mode="w", header=True, index=False, encoding="utf-8")

            ui.notification_show("✅ Ligne ajoutée à `budget_dp.csv`", type="message")
            trigger_update_budget.set(trigger_update_budget() + 1)  # 🔄 Déclenche la mise à jour

    # Stocker l'onglet actuel en réactif
    @reactive.Effect
    @reactive.event(input.page)
    def on_tab_change():
        requested_tab = input.page()
        if requested_tab == "Résultat DP" and not page_autorisee():
            # Remettre l'onglet actif sur l'onglet précédent (empêche le changement)
            ui.update_navs("page", selected=onglet_actuel())
            # Afficher modal pour prévenir
            ui.modal_show(
                ui.modal(
                    "Vous devez valider le budget avant d'accéder aux résultats.",
                    title="Accès refusé",
                    easy_close=True,
                    footer=None
                )
            )
        else:
            # Autoriser le changement d'onglet
            onglet_actuel.set(requested_tab)

    @reactive.Effect
    @reactive.event(input.valider_budget)
    def _():
        ui.modal_show(
            ui.modal(
                "Êtes-vous sûr de vouloir valider le budget ? Cette action est irréversible.",
                title="Confirmation",
                easy_close=False,
                footer=ui.TagList(
                    ui.input_action_button("confirm_validation", "Valider", class_="btn-danger"),
                    ui.input_action_button("cancel_validation", "Annuler", class_="btn-secondary")
                )
            )
        )

    @reactive.Effect
    @reactive.event(input.cancel_validation)
    def _():
        ui.modal_remove()

    @output
    @render.download(filename=lambda: "resultats_dp.xlsx")
    def download_xlsx():

        resultats = resultats_df()
        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for key, df in resultats.items():
                df.to_excel(writer, sheet_name=str(key)[:31], index=False)

        buffer.seek(0)
        return buffer

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def radio_buttons_comptage():
        return ui.layout_columns(
            *make_radio_buttons(requetes(), ["Comptage"]),
            col_widths=3
        )

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def radio_buttons_total():
        return ui.layout_columns(
            *make_radio_buttons(requetes(), ["Total"]),
            col_widths=3
        )

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def radio_buttons_moyenne():
        return ui.layout_columns(
            *make_radio_buttons(requetes(), ["Moyenne"]),
            col_widths=3
        )

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def radio_buttons_quantile():
        return ui.layout_columns(
            *make_radio_buttons(requetes(), ["Quantile"]),
            col_widths=3
        )


app = App(app_ui, server, static_assets=www_dir)
# shiny run --reload shiny_app.py
# shiny run --autoreload-port 8000 shiny_app.py

# shiny run --port 5000 --host 0.0.0.0 shiny_app.py
