# Imports
from src.plots import (
    create_histo_plot, create_fc_emp_plot,
    create_score_plot, create_proba_plot,
    create_barplot
)
from src.layout.introduction_dp import page_introduction_dp
from src.layout.donnees import page_donnees
from src.layout.preparer_requetes import (
    page_preparer_requetes, affichage_requete
)
from src.layout.conception_budget import page_conception_budget, make_radio_buttons
from src.layout.resultat_dp import page_resultat_dp, afficher_resultats
from src.layout.etat_budget_dataset import page_etat_budget_dataset
from src.process_tools import (
    process_request, calculer_toutes_les_requetes, calcul_requete,
    df_comptage, df_total, df_moyenne, df_ratio, df_quantile
)
from src.fonctions import (
    eps_from_rho_delta, optimization_boosted,
    update_context, parse_filter_string,
    get_weights, intervalle_confiance_quantile,
    load_data, manual_quantile_score, extract_column_names_from_choices,
    load_yaml_metadata
)
from src.constant import (
    storage_options,
    contrib_individu,
    chemin_dataset,
    choix_quantile,
    borne_max_taille_dataset
)

from shiny import App, ui, render, reactive, module
from shinywidgets import render_plotly
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
from shinywidgets import output_widget

dp.enable_features("contrib")

www_dir = Path(__file__).parent / "www"

data_example = sns.load_dataset("penguins").dropna()

# 1. UI --------------------------------------
app_ui = ui.page_navbar(
    ui.head_content(ui.include_css(f"{www_dir}/my_style.css")),
    ui.nav_spacer(),
    page_introduction_dp(),
    page_donnees(),
    page_preparer_requetes(),
    page_conception_budget(),
    page_resultat_dp(),
    page_etat_budget_dataset(),
    title=ui.div(
        ui.img(src="logo_insee.png", height="80px", style="margin-right:10px"),
        ui.img(src="logo_poc.png", height="60px", style="margin-right:10px"),
        style="display: flex; align-items: center; gap: 10px;"
    ),
    id="page"
)

# 2. Server ----------------------------------

@module.server
def radio_buttons_server(input, output, session, type_req, requetes, req_calcul):
    @output
    @render.ui
    def radio_buttons():
        return ui.layout_columns(
            *make_radio_buttons(requetes(), type_req, req_calcul()),
            col_widths=3
        )

    def selected_values():
        data_requetes = requetes()
        req_type = {k: v for k, v in data_requetes.items() if v["type"] in type_req}
        if req_type == {}:
            return {}
        return {key: input[key]() for key in req_type.keys()}

    return selected_values


@module.server
def budget_req_server(input, output, session, dataset, requetes, conception_count, conception_sum, conception_quantile, type_req):
    @reactive.Calc
    def dataframe():
        if type_req == "Comptage":
            return df_comptage(requetes, conception_count)
        elif type_req == "Total":
            return df_total(dataset, requetes, conception_count, conception_sum)
        elif type_req == "Moyenne":
            return df_moyenne(dataset, requetes, conception_count, conception_sum)
        elif type_req == "Ratio":
            return df_ratio(dataset, requetes, conception_count, conception_sum)
        elif type_req == "Quantile":
            return df_quantile(conception_quantile)

    @output
    @render.data_frame
    def table_req():
        df = dataframe()
        if df is not None and not df.empty:
            if type_req != "Comptage":
                df = df.drop(columns="label")
            # Définir l'arrondi spécifique
            arrondi = {col: 1 if col in ["cv moyen (%)", "biais relatif moyen (%)", "écart type comptage", "écart type bruit comptage"] else 0 for col in df.columns}
            df = df.round(arrondi)
        return df

    @render_plotly
    def plot_req():
        df = dataframe()
        hover_col = "label" if "label" in df.columns else "groupement"
        if df is not None and not df.empty:
            if type_req == "Comptage":
                ycol = "écart type estimation"
            elif type_req == "Quantile":
                ycol = "taille moyenne IC 95%"
                # Supposons qu'on veuille exclure "variable" ou une autre colonne du group_by
                cols_to_group = ["requête", "label", "groupement", "filtre"]
                existing_cols = [col for col in cols_to_group if col in df.columns]

                # Moyenne des tailles d'IC par groupe
                df = df.groupby(existing_cols, dropna=False)["taille moyenne IC 95%"].mean().reset_index().dropna(axis=1, how="all")
            else:
                ycol = "cv moyen (%)"
            return create_barplot(df, x_col="requête", y_col=ycol, hoover=hover_col, color="groupement")


@module.server
def bloc_budget_server(input, output, session, requetes, header, type_req):

    def bloc_visible():
        return any(req["type"] == type_req for req in requetes().values())

    @output
    @render.ui
    def bloc_budget():
        if bloc_visible():
            return ui.panel_well(
                ui.card(
                    ui.card_header(header),
                    ui.output_ui("radio_buttons"),
                    ui.layout_columns(
                        ui.card(
                            ui.output_data_frame("table_req"),
                            full_screen=True
                        ),
                        ui.card(
                            output_widget("plot_req"),
                            full_screen=True
                        ),
                        col_widths=[6, 6]
                    )
                )
            )


def server(input, output, session):

    requetes = reactive.Value({})
    page_autorisee = reactive.Value(False)
    resultats_df = reactive.Value({})
    onglet_actuel = reactive.Value("Conception du budget")  # Onglet par défaut
    trigger_update_budget = reactive.Value(0)

    bloc_budget_server("Comptage", requetes, header="Répartition du budget pour les comptages", type_req="Comptage")
    bloc_budget_server("Total", requetes, header="Répartition du budget pour les totaux", type_req="Total")
    bloc_budget_server("Moyenne", requetes, header="Répartition du budget pour les moyennes", type_req="Moyenne")
    bloc_budget_server("Ratio", requetes, header="Répartition du budget pour les ratios", type_req="Ratio")
    bloc_budget_server("Quantile", requetes, header="Répartition du budget pour les quantiles", type_req="Quantile")

    @reactive.Calc
    def dict_query() -> dict[str, dict[str, Any]]:
        data_requetes = requetes()
        req_comptage_quantile = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["comptage", "quantile"]}
        req_total_moyenne = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["moyenne", "total"]}
        req_ratio = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["ratio"]}
        query = {}
        i = 1
        for (key, request) in req_comptage_quantile.items():
            query_request = request.copy()
            query_request["req"] = [key]
            cle = f"query_{i}"
            query[cle] = query_request
            i += 1

        for (key, request) in req_total_moyenne.items():
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

        for (key, request) in req_ratio.items():
            query_request = request.copy()
            # Cas 1 : Total variable 1
            query_request["type"] = "Total"
            query_request.pop("variable_denominateur", None)
            query_request.pop("bounds_denominateur", None)

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

            # Cas 2 : Total variable 2
            query_request = request.copy()
            query_request["type"] = "Total"
            query_request["variable"] = query_request.pop("variable_denominateur", None)
            query_request["bounds"] = query_request.pop("bounds_denominateur", None)

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

            # Cas 3 : Comptage
            query_request = request.copy()
            query_request["type"] = "Comptage"
            query_request.pop("variable", None)
            query_request.pop("bounds", None)
            query_request.pop("variable_denominateur", None)
            query_request.pop("bounds_denominateur", None)

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
                params["sigma2"] = (U - L)**2/(4 * 2 * input.budget_total() * params["poids"])
        return query

    @reactive.Calc
    def conception_query_count() -> dict[str, dict[str, Any]]:
        data_query = dict_query()

        # Sélection des requêtes de type "comptage"
        query_comptage = {
            k: v for k, v in data_query.items()
            if v["type"].lower() in ["count", "comptage"]
        }

        # Récupération des filtres distincts
        filtres_uniques = set(v.get("filtre") for v in query_comptage.values())

        # Dictionnaire final fusionné
        requetes_finales: dict[str, dict[str, Any]] = {}

        for filtre in filtres_uniques:
            # Sous-ensemble des requêtes avec ce filtre
            query_filtre = {
                k: v for k, v in query_comptage.items()
                if v.get("filtre") == filtre
            }

            # Optimisation spécifique à ce groupe
            query_filtre_opt = optimization_boosted(dict_query=query_filtre, modalite=key_values())

            # Ajout explicite du filtre dans chaque requête
            query_filtre_opt = {
                k: {**v, "filtre": filtre}
                for k, v in query_filtre_opt.items()
            }

            # Fusion dans le dictionnaire final
            requetes_finales.update(query_filtre_opt)

        return requetes_finales

    @reactive.Calc
    def conception_query_sum() -> dict[str, dict[str, Any]]:
        data_query = dict_query()

        query_total = {k: v for k, v in data_query.items() if v["type"].lower() in ["total", "sum", "somme"]}

        # Récupération des filtres distincts
        filtres_uniques = set(v.get("filtre") for v in query_total.values())

        # Récupération des variables distinctes
        variables_uniques = set(v["variable"] for v in query_total.values())

        # Dictionnaire final fusionné
        requetes_finales: dict[str, dict[str, Any]] = {}

        for filtre in filtres_uniques:
            for variable in variables_uniques:

                query_filtre_variable = {
                    k: v for k, v in query_total.items()
                    if v["variable"] == variable and v.get("filtre") == filtre
                }
                query_filtre_variable_opt = optimization_boosted(dict_query=query_filtre_variable, modalite=key_values())

                # Ajout explicite du filtre dans chaque requête
                query_filtre_variable_opt = {
                    k: {**v, "filtre": filtre}
                    for k, v in query_filtre_variable_opt.items()
                }

                # Fusion dans le dictionnaire final
                requetes_finales.update(query_filtre_variable_opt)

        return requetes_finales

    @reactive.Calc
    def conception_query_quantile() -> dict[str, dict[str, Any]]:

        data_query = dict_query()
        query_quantile = {k: v for k, v in data_query.items() if v["type"].lower() in ["quantile"]}

        # Récupération des filtres distincts
        filtres_uniques = set(v.get("filtre") for v in query_quantile.values())

        variables_uniques = set(v["variable"] for v in query_quantile.values())

        for filtre in filtres_uniques:
            for variable in variables_uniques:
                query_filtre_variable = {
                    k: v for k, v in query_quantile.items()
                    if v["variable"] == variable and v.get("filtre") == filtre
                }

                for key_query, query in query_filtre_variable.items():

                    budget_req_quantile = input.budget_total() * query["poids"]
                    epsilon = np.sqrt(8 * budget_req_quantile)

                    vrai_tableau = process_request(dataset().lazy(), query, use_bounds=False)

                    ic = intervalle_confiance_quantile(dataset().lazy(), query, epsilon, vrai_tableau)

                    query_quantile[key_query]["scale"] = ic
        return query_quantile

    @output
    @render.ui
    @reactive.event(input.confirm_validation)
    async def req_dp_display():

        data_query = dict_query()
        data_lazy = dataset().lazy()

        # 🔍 Extraire toutes les colonnes mentionnées dans les requêtes
        vars_by = {val for req in data_query.values() for val in req.get("by", [])}
        vars_variable = {v for v in (req.get("variable") for req in data_query.values()) if v is not None}
        selected_columns = set(vars_by | vars_variable)  # union des deux ensembles

        # 🐍 Sous-échantillon propre du LazyFrame
        if not selected_columns:
            filtered_lazy = data_lazy.with_columns(pl.lit(1).alias("__dummy")).select("__dummy").collect().lazy()

        else:
            filtered_lazy = data_lazy.select(selected_columns).collect().lazy()

        with ui.Progress(min=0, max=len(data_query)) as p:
            p.set(0, message="Traitement en cours...", detail="Analyse requête par requête...")

            context_param = {
                    "data": filtered_lazy,
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
        chemin = chemin_dataset.get(input.default_dataset())
        metadata = load_yaml_metadata(chemin)
        return yaml.dump(metadata, sort_keys=False, allow_unicode=True)

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

    # Extrait les modalités uniques des variables qualitatives
    @reactive.Calc
    def key_values():
        df = dataset()
        data_query = dict_query()
        list_var = set(val for v in data_query.values() for val in v.get("by", []))

        # Extraire les modalités uniques, triées, sans NaN
        return {
            col: sorted(df[col].drop_nulls().unique().to_list())
            for col in list_var
        }

    @reactive.Calc
    def nb_modalite_var():
        data_query = dict_query()
        metadata_dict = yaml.safe_load(yaml_metadata_str()) if yaml_metadata_str() else {}
        if 'columns' not in metadata_dict:
            return {}

        list_var = set(val for v in data_query.values() for val in v.get("by", []))

        nb_modalites = {}
        for col in list_var:
            nb = metadata_dict['columns'][col].get('unique_values', None)
            if nb is not None:
                nb_modalites[col] = [i for i in range(nb)]
        return nb_modalites

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
                ui.update_selectize("delete_req", choices=["Delete all"] + list(data.keys()))
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
        type_req = input.type_req()

        if type_req == "":
            ui.notification_show("❌ Aucun type de requête n'est spécifié", type="error")
            return

        # Charger les métadonnées YAML dans un dict (tu peux le faire une fois dans un Calc sinon)
        metadata_dict = yaml.safe_load(yaml_metadata_str()) if yaml_metadata_str() else {}
        variable = input.variable() if type_req != "Comptage" else None
        variable_denom = input.variable_denominateur() if type_req == "Ratio" else None

        nb_candidats = input.nb_candidats() if type_req == "Quantile" else None

        # Vérifie si variable est vide ou None
        if not variable and type_req != "Comptage":
            ui.notification_show("❌ Aucune variable sélectionnée", type="error")
            return

        if not variable_denom and type_req == "Ratio":
            ui.notification_show("❌ Aucune variable sélectionnée", type="error")
            return

        bounds = None
        bounds_denom = None

        # Essayer de récupérer min/max dans YAML pour la variable choisie
        if 'columns' in metadata_dict and variable in metadata_dict['columns']:
            col_meta = metadata_dict['columns'][variable]
            min_val = col_meta.get('min')
            max_val = col_meta.get('max')
            if min_val is not None and max_val is not None:
                bounds = [float(min_val), float(max_val)]

        if 'columns' in metadata_dict and variable_denom in metadata_dict['columns']:
            col_meta = metadata_dict['columns'][variable_denom]
            min_val = col_meta.get('min')
            max_val = col_meta.get('max')
            if min_val is not None and max_val is not None:
                bounds_denom = [float(min_val), float(max_val)]

        # 🧪 Vérification syntaxique du filtre
        filtre_str = input.filtre()
        if filtre_str:
            try:
                all_columns = extract_column_names_from_choices(variable_choices())
                _ = parse_filter_string(filtre_str, columns=all_columns)
            except Exception:
                ui.notification_show("❌ Erreur dans le format du filtre : vérifiez les opérateurs et les noms de variables", type="error")
                return

        base_dict = {
            "type": type_req,
            "variable": variable,
            "bounds": bounds,
            "by": sorted(input.group_by()),  # tri pour éviter doublons
            "filtre": input.filtre(),
        }

        if type_req == 'Quantile':
            alpha = sorted(input.alpha())

            if not nb_candidats:
                ui.notification_show("❌ Nombre de valeurs candidates au quantile manquant", type="error")
                return

            if nb_candidats < 5:
                ui.notification_show("❌ Nombre de valeurs candidates au quantile insuffisant", type="error")
                return

            if not alpha:
                ui.notification_show("❌ Pas de quantile sélectionné", type="error")
                return

            base_dict.update({
                "alpha": alpha,
                "nb_candidats": nb_candidats,
            })

        elif type_req == 'Ratio':
            if not variable_denom:
                ui.notification_show("❌ Aucune variable sélectionnée", type="error")
                return

            base_dict.update({
                "variable_denominateur": variable_denom,
                "bounds_denominateur": bounds_denom
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
                clean_dict.get("type") != "Quantile" or (
                    existing_req.get("alpha") == clean_dict.get("alpha") and
                    existing_req.get("nb_candidats") == clean_dict.get("nb_candidats")
                )
            ) and (
                clean_dict.get("type") != "Ratio" or (
                    existing_req.get("variable_denominateur") == clean_dict.get("variable_denominateur") and
                    existing_req.get("bounds_denominateur") == clean_dict.get("bounds_denominateur")
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
        ui.update_selectize("delete_req", choices=["Delete all"] + list(requetes().keys()))

    @reactive.effect
    @reactive.event(input.delete_btn)
    def _():
        current = requetes().copy()
        targets = input.delete_req()  # ceci est une liste ou un tuple de valeurs

        if not targets:
            ui.notification_show("❌ Aucune requête sélectionnée", type="error")
            return

        if "Delete all" in targets:
            current.clear()  # Vide toutes les requêtes
            requetes.set(current)  # Met à jour le reactive.Value
            ui.notification_show(f"🗑️ TOUTES les requêtes ont été supprimé", type="warning")
            ui.update_selectize("delete_req", choices=[])
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
            ui.update_selectize("delete_req", choices=["Delete all"] + list(requetes().keys()))
        if not_found:
            ui.notification_show(f"❌ Requête(s) introuvable(s) : {', '.join(not_found)}", type="error")

    @reactive.Calc
    def req_calcul():
        data_requetes = requetes()
        return calcul_requete(data_requetes, dataset())

    @output
    @render.ui
    def req_display():
        type_req_a_afficher = input.affichage_req()
        data_requetes = requetes()
        if "TOUTES" in type_req_a_afficher:
            if data_requetes == {}:
                return ui.p("Aucune requête entrée.")
            return affichage_requete(data_requetes, req_calcul())

        req = {k: v for k, v in data_requetes.items() if v["type"] in type_req_a_afficher}
        if req == {}:
            return ui.p("Aucune requête entrée.")
        return affichage_requete(req, req_calcul())

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
                    <li>Ou bien, (<strong>ε</strong> = <code>{eps:.3f}</code>, <strong>δ</strong> = <code>{delta}</code>)</li>
                </ul>
            </div>
        """)

    @render.plot
    def histo_plot():
        return create_histo_plot(data_example, input.alpha_slider())

    @render.plot
    def fc_emp_plot():
        return create_fc_emp_plot(data_example, input.alpha_slider())

    @reactive.Calc
    def score_proba_quantile():
        nb_candidat = input.candidat_slider()
        alpha = input.alpha_slider()
        epsilon = input.epsilon_slider()
        df = data_example

        L = df["body_mass_g"].min()
        U = df["body_mass_g"].max()

        candidats = np.linspace(L, U, nb_candidat).tolist()
        scores, sensi = manual_quantile_score(df['body_mass_g'], candidats, alpha=alpha, et_si=True)

        # Probabilités exponentielles (mécanisme exponentiel)
        proba_non_norm = np.exp(-epsilon * scores / (2 * sensi))
        proba = proba_non_norm / np.sum(proba_non_norm)

        # Top 95% des probabilités
        sorted_indices = np.argsort(proba)[::-1]
        sorted_proba = proba[sorted_indices]
        cumulative = np.cumsum(sorted_proba)
        top95_mask = cumulative <= 0.95
        if not np.all(top95_mask):
            top95_mask[np.argmax(cumulative > 0.95)] = True
        top95_indices = sorted_indices[top95_mask]

        # Marquages
        top95_cumul = [i in top95_indices for i in range(len(candidats))]

        return {
            "candidats": candidats,
            "scores": scores,
            "proba": proba,
            "top95_cumul": top95_cumul
        }

    @render.plot
    def score_plot():
        return create_score_plot(data=score_proba_quantile())

    @render.plot
    def proba_plot():
        return create_proba_plot(data=score_proba_quantile())

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

        elif input.dataset_name() == "":
            ui.notification_show(f"❌ Vous devez spécifier un nom au dataset.", type="error")

        elif input.echelle_geo() == "":
            ui.notification_show(f"❌ Vous devez spécifier l'échelle géographique de l'étude.", type="error")

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

    values_buttons_comptage = radio_buttons_server("Comptage", ["Comptage"], requetes, req_calcul)
    values_buttons_total = radio_buttons_server("Total", ["Total"], requetes, req_calcul)
    values_buttons_moyenne = radio_buttons_server("Moyenne", ["Moyenne"], requetes, req_calcul)
    values_buttons_ratio = radio_buttons_server("Ratio", ["Ratio"], requetes, req_calcul)
    values_buttons_quantile = radio_buttons_server("Quantile", ["Quantile"], requetes, req_calcul)

    budget_req_server("Comptage", dataset, requetes, conception_query_count, conception_query_sum, conception_query_quantile, "Comptage")
    budget_req_server("Total", dataset, requetes, conception_query_count, conception_query_sum, conception_query_quantile, "Total")
    budget_req_server("Moyenne", dataset, requetes, conception_query_count, conception_query_sum, conception_query_quantile, "Moyenne")
    budget_req_server("Ratio", dataset, requetes, conception_query_count, conception_query_sum, conception_query_quantile, "Ratio")
    budget_req_server("Quantile", dataset, requetes, conception_query_count, conception_query_sum, conception_query_quantile, "Quantile")

    @reactive.Calc
    def get_poids_req() -> dict[str, float]:
        values_buttons = {
            **values_buttons_comptage(), **values_buttons_total(),
            **values_buttons_moyenne(), **values_buttons_ratio(), **values_buttons_quantile()
        }
        poids = get_weights(requetes(), values_buttons)
        return poids

    @render.ui
    def ligne_conditionnelle():
        type_req = input.type_req()
        variables = variable_choices().copy()
        ui.update_selectize("group_by", choices=variables)

        if type_req == "Comptage" or type_req == "":
            return None

        contenu = []
        # Supprime le groupe "Qualitatives"
        del variables["🔤 Qualitatives"]

        label_variable = "Variable au numérateur:" if type_req == "Ratio" else "Variable:"
        contenu.append(ui.column(3, ui.input_selectize("variable", label_variable,
                                                    choices=variables, options={"plugins": ["clear_button"]})))

        if type_req == "Ratio":
            contenu.append(ui.column(3, ui.input_selectize("variable_denominateur", "Variable au dénominateur:",
                                                        choices=variables, options={"plugins": ["clear_button"]})))

        if type_req == "Quantile":
            contenu.append(ui.column(3, ui.input_selectize("alpha", "Choix des quantiles:", choices=choix_quantile, multiple=True)))
            contenu.append(ui.column(3, ui.input_numeric("nb_candidats", "Nombre de candidats:", 1000, min=5, max=1_000_000, step=5)))

        return ui.row(*contenu)


app = App(app_ui, server, static_assets=www_dir)
# shiny run --reload --launch-browser app.py
# shiny run --autoreload-port 8000 app.py

# shiny run --port 5000 --host 0.0.0.0 app.py
