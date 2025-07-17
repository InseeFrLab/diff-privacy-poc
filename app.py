# Imports
from src.plots import (
    create_histo_plot, create_fc_emp_plot,
    create_score_plot, create_proba_plot,
    create_barplot
)
from src.layout.introduction_dp import page_introduction_dp
from src.layout.donnees import page_donnees
from src.layout.preparer_requetes import (
    page_preparer_requetes, affichage_requete, affichage_bouton
)
from src.layout.conception_budget import page_conception_budget, make_radio_buttons
from src.layout.resultat_dp import page_resultat_dp, afficher_resultats
from src.layout.etat_budget_dataset import page_etat_budget_dataset
from src.process_tools import (
    process_request, calculer_toutes_les_requetes,
    df_comptage, df_total, df_moyenne, df_ratio, df_quantile
)
from src.fonctions import (
    eps_from_rho_delta, optimisation_chaine,
    update_context, parse_filter_string,
    get_weights, intervalle_confiance_quantile,
    load_data, manual_quantile_score, extract_column_names_from_choices,
    extract_bounds, same_base_request, same_quantile_params, same_ratio_params,
    load_yaml_metadata, assert_or_notify
)
from src.constant import (
    storage_options,
    contrib_individu,
    chemin_dataset,
    choix_quantile,
    borne_max_taille_dataset
)

from shiny import App, ui, render, reactive, module, Inputs, Outputs, Session
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
from typing import Optional, Union
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import textwrap

dp.enable_features("contrib")

www_dir = Path(__file__).parent / "www"

data_example = sns.load_dataset("penguins").dropna()

# 1. UI --------------------------------------
app_ui = ui.page_navbar(
    ui.head_content(
        ui.include_css(f"{www_dir}/my_style.css"),
        ui.tags.script(
            src="https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
        ),
        ui.tags.script("if (window.MathJax) MathJax.Hub.Queue(['Typeset', MathJax.Hub]);")
    ),
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
def radio_buttons_server(
    input: Inputs, output: Outputs, session: Session,
    requetes: reactive.Value[dict[str, dict[str, Any]]],
    req_calcul: reactive.calc
):
    type_req = session.ns

    @output
    @render.ui
    def radio_buttons() -> ui.TagList:
        return ui.layout_columns(
            *make_radio_buttons(requetes(), type_req, req_calcul()),
            col_widths=3
        )

    def selected_values() -> dict[str, str]:
        data_requetes = requetes()
        req_type = {k: v for k, v in data_requetes.items() if v["type"] == type_req}
        if not req_type:
            return {}
        return {key: input[key]() for key in req_type.keys()}

    return selected_values


@module.server
def budget_req_server(
    input: Inputs, output: Outputs, session: Session,
    dataset: reactive.calc,
    requetes: reactive.Value[dict[str, dict[str, Any]]],
    conception_count: reactive.Value[dict[str, dict[str, Any]]],
    conception_sum: reactive.Value[dict[str, dict[str, Any]]],
    conception_quantile: reactive.Value[dict[str, dict[str, Any]]]
):
    type_req = session.ns

    @reactive.calc
    def dataframe() -> pd.DataFrame:
        # Table de correspondance entre type_req et fonction de calcul
        dispatch = {
            "Comptage": lambda: df_comptage(requetes, conception_count),
            "Total": lambda: df_total(dataset, requetes, conception_count, conception_sum),
            "Moyenne": lambda: df_moyenne(dataset, requetes, conception_count, conception_sum),
            "Ratio": lambda: df_ratio(dataset, requetes, conception_count, conception_sum),
            "Quantile": lambda: df_quantile(conception_quantile),
        }

        return dispatch[type_req]()

    @output
    @render.data_frame
    def table_req() -> pd.DataFrame:
        df = dataframe()
        if not df.empty:
            if type_req != "Comptage":
                df = df.drop(columns="label")
            # Définir l'arrondi spécifique
            cols = [
                "cv moyen (%)", "biais relatif moyen (%)",
                "écart type comptage", "écart type bruit comptage"
            ]

            arrondi = {col: 1 if col in cols else 0 for col in df.columns}
            df = df.round(arrondi)
            return df

    @render_plotly
    def plot_req() -> go.Figure:
        df = dataframe()
        textcol = "label" if "label" in df.columns else "groupement"

        if not df.empty:
            if type_req == "Comptage":
                ycol = "écart type estimation"

            elif type_req == "Quantile":
                ycol = "taille moyenne IC 95%"
                cols_to_group = ["requête", "label", "groupement", "filtre"]
                existing_cols = [col for col in cols_to_group if col in df.columns]

                # Moyenne des tailles d'IC par groupe
                df = (
                    df.groupby(existing_cols, dropna=False)["taille moyenne IC 95%"]
                    .mean().reset_index().dropna(axis=1, how="all")
                )
            else:
                ycol = "cv moyen (%)"
            return create_barplot(df, x_col="requête", y_col=ycol, text=textcol, color="groupement")


@module.server
def bloc_budget_server(
    input: Inputs, output: Outputs, session: Session,
    requetes: reactive.Value[dict[str, dict[str, Any]]], header: str, type_req: str
):

    def bloc_visible() -> bool:
        return any(req["type"] == type_req for req in requetes().values())

    @output
    @render.ui
    def bloc_budget() -> ui.TagList:
        if bloc_visible():
            return ui.panel_well(
                ui.card(
                    ui.card_header(header),
                    ui.output_ui("radio_buttons"),
                    ui.layout_columns(
                        ui.card(ui.output_data_frame("table_req"), full_screen=True),
                        ui.card(output_widget("plot_req"), full_screen=True),
                        col_widths=[6, 6]
                    )
                )
            )


def server(input: Inputs, output: Outputs, session: Session):

    requetes: reactive.Value[dict[str, dict[str, Any]]] = reactive.Value({})
    page_autorisee: reactive.Value[bool] = reactive.Value(False)
    resultats_df: reactive.Value[dict[str, pd.Dataframe]] = reactive.Value({})
    onglet_actuel: reactive.Value[str] = reactive.Value("Conception du budget")  # Onglet par défaut
    trigger_update_budget: reactive.Value[int] = reactive.Value(0)

    bloc_budget_server(
        "Comptage", requetes, header="Répartition du budget pour les comptages", type_req="Comptage"
    )
    bloc_budget_server(
        "Total", requetes, header="Répartition du budget pour les totaux", type_req="Total"
    )
    bloc_budget_server(
        "Moyenne", requetes, header="Répartition du budget pour les moyennes", type_req="Moyenne"
    )
    bloc_budget_server(
        "Ratio", requetes, header="Répartition du budget pour les ratios", type_req="Ratio"
    )
    bloc_budget_server(
        "Quantile", requetes, header="Répartition du budget pour les quantiles", type_req="Quantile"
    )

    # To DO
    @reactive.calc
    def dict_query() -> dict[str, dict[str, Any]]:
        data_requetes = requetes()
        req_comptage_quantile = {
            k: v for k, v in data_requetes.items() if v["type"].lower() in ["comptage", "quantile"]}
        req_total_moyenne = {
            k: v for k, v in data_requetes.items() if v["type"].lower() in ["moyenne", "total"]}
        req_ratio = {
            k: v for k, v in data_requetes.items() if v["type"].lower() in ["ratio"]}
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

    @reactive.calc
    def conception_query_count() -> dict[str, dict[str, Any]]:
        data_query = dict_query()
        query_comptage = {k: v for k, v in data_query.items()if v["type"].lower() == "comptage"}
        return optimisation_chaine(query_comptage, key_values())

    @reactive.calc
    def conception_query_sum() -> dict[str, dict[str, Any]]:
        data_query = dict_query()
        query_total = {k: v for k, v in data_query.items() if v["type"].lower() == "total"}
        return optimisation_chaine(query_total, key_values())

    @reactive.calc
    def conception_query_quantile() -> dict[str, dict[str, Any]]:
        data_query = dict_query()
        query_quantile = {k: v for k, v in data_query.items() if v["type"].lower() == "quantile"}
        filtres_uniques = set(v.get("filtre") for v in query_quantile.values())
        variables_uniques = set(v.get("variable") for v in query_quantile.values())

        for filtre in filtres_uniques:
            for variable in variables_uniques:
                query_filtre_variable = {
                    k: v for k, v in query_quantile.items()
                    if v.get("variable") == variable and v.get("filtre") == filtre
                }

                for key_query, query in query_filtre_variable.items():

                    budget_req_quantile = input.budget_total() * query.get("poids", 0)
                    epsilon = np.sqrt(8 * budget_req_quantile)

                    vrai_tableau = process_request(dataset(), query, use_bounds=False)
                    ic = intervalle_confiance_quantile(dataset(), query, epsilon, vrai_tableau)
                    query_quantile[key_query]["scale"] = ic

        return query_quantile

    @output
    @render.ui
    @reactive.event(input.confirm_validation)
    async def req_dp_display() -> ui.TagList:

        data_query = dict_query()
        data_lazy = dataset()
        keys = key_values()

        # Extraire toutes les colonnes mentionnées dans les requêtes
        vars_by = {val for req in data_query.values() for val in req.get("by", [])}
        vars_variable = {
            v for v in (req.get("variable") for req in data_query.values())
            if v is not None
        }
        selected_columns = set(vars_by | vars_variable)  # union des deux ensembles

        # Sous-échantillon propre du LazyFrame
        if not selected_columns:
            filtered_lazy = (
                data_lazy.with_columns(pl.lit(1).alias("__dummy"))
                .select("__dummy")
                .collect()
                .lazy()
            )

        else:
            filtered_lazy = data_lazy.select(selected_columns).collect().lazy()

        with ui.Progress(min=0, max=len(data_query)) as p:
            p.set(0, message="Traitement en cours...", detail="Analyse requête par requête...")

            context_param = {
                    "data": filtered_lazy,
                    "privacy_unit": dp.unit_of(contributions=contrib_individu),
                    "margins": [dp.polars.Margin(max_partition_length=borne_max_taille_dataset)],
                }

            context_rho, context_eps = update_context(
                context_param, input.budget_total(), data_query
            )

            await calculer_toutes_les_requetes(
                context_rho, context_eps, keys, data_query, p, resultats_df
            )

        return afficher_resultats(resultats_df, requetes(), data_query, keys)

    @reactive.calc
    def dataset() -> pl.LazyFrame:
        """
        Charge un dataset depuis un fichier utilisateur (CSV ou Parquet)
        ou bien depuis un jeu de données par défaut.
        """
        file = input.dataset_input()
        if file is not None:
            ext = Path(file["name"]).suffix
            if ext == ".csv":
                return pl.read_csv(file["datapath"]).lazy()
            elif ext == ".parquet":
                return load_data(file["datapath"], storage_options)
            else:
                raise ValueError("❌ Format non supporté : utiliser CSV ou Parquet")

        # Si aucun fichier fourni, charger le jeu par défaut
        default = input.default_dataset()
        if default == "penguins":
            return pl.DataFrame(sns.load_dataset("penguins").dropna()).lazy()
        else:
            return load_data(default, storage_options)

    @reactive.calc
    def yaml_metadata_str() -> str | None:
        """
        Retourne les métadonnées YAML en chaîne formatée.
        """
        chemin = chemin_dataset.get(input.default_dataset())
        metadata = load_yaml_metadata(chemin)
        return yaml.dump(metadata, sort_keys=False, allow_unicode=True) if metadata else None

    @output
    @render.data_frame
    def data_view() -> pl.DataFrame:
        return dataset().limit(500).collect()

    @output
    @render.text
    def meta_data() -> ui.Tag:
        """
        Affiche les métadonnées YAML sous forme préformatée,
        ou un message si aucune métadonnée n’est disponible.
        """
        metadata = yaml_metadata_str()

        if not metadata:
            return ui.tags.em("Aucune métadonnée disponible.")

        return ui.tags.div(
            ui.tags.p("Métadonnées YAML :"),
            ui.tags.pre(metadata)
        )

    @reactive.calc
    def variable_choices() -> dict[str, Union[str, dict[str, str]]]:
        """
        Retourne un dictionnaire catégorisé des variables du dataset,
        distinguant qualitatives et quantitatives.
        """
        df = dataset().limit(1).collect()  # Uniquement 1 ligne pour les dtypes
        if df is None:
            return {}

        qualitative_types = {pl.Utf8, pl.Categorical, pl.Boolean}
        quantitative_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        }

        qualitative = [
            col for col, dtype in zip(df.columns, df.dtypes) if dtype in qualitative_types
        ]
        quantitative = [
            col for col, dtype in zip(df.columns, df.dtypes) if dtype in quantitative_types
        ]

        return {
            "": "",
            "🔤 Qualitatives": {col: col for col in qualitative},
            "🧮 Quantitatives": {col: col for col in quantitative}
        }

    # Extrait les modalités uniques des variables qualitatives
    @reactive.calc
    def key_values() -> dict[str, list[Any]]:
        """
        Pour chaque variable référencée dans les requêtes, retourne la liste triée
        de ses modalités uniques dans le dataset, en excluant les valeurs nulles.
        """
        df = dataset().collect()
        data_query = dict_query()
        variables = {val for v in data_query.values() for val in v.get("by", [])}

        # Extraire les modalités uniques, triées, sans NaN
        return {
            col: sorted(df[col].drop_nulls().unique().to_list())
            for col in variables
        }

    @reactive.calc
    def nb_modalite_var() -> dict[str, list[int]]:
        """
        Pour chaque variable dans les requêtes, retourne une liste
        d'entiers de 0 à (nombre de modalités - 1).
        Utilise les métadonnées YAML pour récupérer le nombre de modalités uniques par variable.
        """
        data_query = dict_query()
        metadata_str = yaml_metadata_str()
        metadata_dict = yaml.safe_load(metadata_str) if metadata_str else {}

        if not metadata_dict or 'columns' not in metadata_dict:
            return {}

        # Extraire l'ensemble des variables référencées dans les requêtes (clé 'by')
        variables = {val for v in data_query.values() for val in v.get("by", [])}

        nb_modalites = {}

        for var in variables:
            col_info = metadata_dict['columns'].get(var, {})
            nb = col_info.get('unique_values')
            if isinstance(nb, int) and nb > 0:
                nb_modalites[var] = list(range(nb))

        return nb_modalites

    @reactive.effect
    @reactive.event(input.request_input)
    def _() -> None:
        """
        Lit un fichier JSON chargé par l'utilisateur et met à jour la variable `requetes`.
        Affiche une notification en cas d'erreur de décodage JSON.
        """
        fileinfo = input.request_input()
        if not fileinfo:
            return

        filepath = Path(fileinfo[0]["datapath"])
        try:
            with filepath.open(encoding="utf-8") as f:
                data = json.load(f)

            requetes.set(data)
            ui.update_selectize("delete_req", choices=["TOUTES"] + list(data.keys()))
            ui.notification_show("✅ Requêtes importées avec succès", type="message")

        except json.JSONDecodeError:
            ui.notification_show("❌ Fichier JSON invalide", type="error")
        except Exception as e:
            ui.notification_show(f"❌ Erreur lors de l'import : {e}", type="error")

    @output
    @render.download(filename=lambda: "requetes_exportees.json")
    def download_json() -> io.StringIO:
        """
        Exporte les requêtes courantes au format JSON, encodé en UTF-8 avec indentation.
        """
        buffer = io.StringIO()
        json.dump(requetes(), buffer, indent=2, ensure_ascii=False)
        buffer.seek(0)
        return buffer

    @reactive.effect
    @reactive.event(input.add_req)
    def _() -> None:
        current = requetes().copy()
        type_req = input.type_req()

        if not assert_or_notify(type_req, "Aucun type de requête n'est spécifié"):
            return

        metadata_dict = yaml.safe_load(yaml_metadata_str()) if yaml_metadata_str() else {}
        variable = input.variable() if type_req != "Comptage" else None
        variable_denom = input.variable_denominateur() if type_req == "Ratio" else None
        nb_candidats = input.nb_candidats() if type_req == "Quantile" else None

        if not assert_or_notify(variable or type_req == "Comptage", "Aucune variable sélectionnée"):
            return

        if not assert_or_notify(
            variable_denom or type_req != "Ratio",
            "Aucune variable sélectionnée"
        ):
            return

        bounds = extract_bounds(metadata_dict, variable)
        bounds_denom = extract_bounds(metadata_dict, variable_denom)

        # 🧪 Vérification syntaxique du filtre
        filtre_str = input.filtre()
        if filtre_str:
            try:
                all_columns = extract_column_names_from_choices(variable_choices())
                _ = parse_filter_string(filtre_str, columns=all_columns)
            except Exception:
                text = (
                    "❌ Erreur dans le format du filtre : "
                    "vérifiez les opérateurs et les noms de variables"
                )
                ui.notification_show(text, type="error")
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

            if not assert_or_notify(
                nb_candidats,
                "Nombre de valeurs candidates au quantile manquant"
            ):
                return

            if not assert_or_notify(
                nb_candidats > 5,
                "Nombre de valeurs candidates au quantile insuffisant"
            ):
                return

            if not assert_or_notify(
                alpha,
                "Pas de quantile sélectionné"
            ):
                return

            base_dict.update({
                "alpha": alpha,
                "nb_candidats": nb_candidats,
            })

        elif type_req == 'Ratio':
            if not assert_or_notify(
                variable_denom,
                "Aucune variable sélectionnée"
            ):
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
            same_base_request(existing_req, clean_dict)
            and
            (clean_dict.get("type") != "Quantile" or same_quantile_params(existing_req, clean_dict))
            and
            (clean_dict.get("type") != "Ratio" or same_ratio_params(existing_req, clean_dict))
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
        ui.update_selectize("delete_req", choices=["TOUTES"] + list(requetes().keys()))

    @reactive.effect
    @reactive.event(input.delete_btn)
    def _() -> None:
        """
        Supprime les requêtes sélectionnées via le sélecteur 'delete_req'.
        Gère les cas : aucune sélection, suppression partielle, suppression totale.
        """
        current = requetes().copy()
        targets = input.delete_req()  # liste ou tuple de clés à supprimer

        if not assert_or_notify(targets, "Aucune requête sélectionnée"):
            return

        if "TOUTES" in targets:
            current.clear()  # Vide toutes les requêtes
            requetes.set(current)  # Met à jour le reactive.Value
            ui.notification_show("🗑️ TOUTES les requêtes ont été supprimées.", type="warning")
            ui.update_selectize("delete_req", choices=[])
            return

        removed, not_found = [], []
        for target in targets:
            if target in current:
                del current[target]
                removed.append(target)
            else:
                not_found.append(target)

        requetes.set(current)
        ui.update_selectize("delete_req", choices=["TOUTES"] + list(requetes().keys()))

        if removed:
            ui.notification_show(
                f"🗑️ Requête(s) supprimée(s) : {', '.join(removed)}", type="warning")

        if not_found:
            ui.notification_show(
                f"❌ Requête(s) introuvable(s) : {', '.join(not_found)}", type="error")

    @reactive.calc
    def req_calcul() -> dict[str, pd.DataFrame]:
        data_requetes = requetes()
        dict_results = {}

        for key, req in data_requetes.items():
            resultat = process_request(dataset(), req, use_bounds=False).to_pandas()
            dict_results[key] = resultat

        return dict_results

    @output
    @render.ui
    def req_display() -> ui.TagList:
        """
        Affiche les requêtes sélectionnées selon leur type.
        """
        types_selectionnes = input.affichage_req()
        data_requetes = requetes()

        # Pas de requêtes disponibles
        if not data_requetes:
            return ui.p("Aucune requête entrée.")

        # Toutes les requêtes ou filtrées par type
        if "TOUTES" in types_selectionnes:
            requetes_affichees = data_requetes
        else:
            requetes_affichees = {
                k: v for k, v in data_requetes.items()
                if v["type"] in types_selectionnes
            }

        if not requetes_affichees:
            return ui.p("Aucune requête entrée.")

        return affichage_requete(requetes_affichees, req_calcul())

    # TO DO
    @render.ui
    def interval_summary() -> ui.Tag:
        sigma = input.scale_gauss()
        quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
        result_lines = []

        for q in quantiles:
            z = norm.ppf(0.5 + q / 2)
            bound = round(z * sigma, 3)
            line = (
                f"<li><strong>{int(q * 100)}%</strong> de chances que le bruit soit entre "
                f"+/- <code>{round(bound, 1)}</code></li>"
            )
            result_lines.append(line)

        box_style = (
            "margin-top:20px; padding:10px; background-color:#f9f9f9; border-radius:12px; "
            'font-family: "Raleway", "Garamond", sans-serif; font-size:16px; color:#333'
        )

        content = textwrap.dedent(f"""
            <div style="{box_style}">
                <p style="margin-bottom:10px">
                    <strong>Résumé des intervalles de confiance :</strong>
                </p>
                <ul style="padding-left: 20px; margin: 0;">
                    {''.join(result_lines)}
                </ul>
            </div>
        """)

        return ui.HTML(content)

    @output
    @render.data_frame
    def cross_table() -> pd.DataFrame:
        """
        Calcule un tableau croisé entre 'species' et 'island'.
        """
        table = (
            data_example
            .groupby(["species", "island"])
            .size()
            .unstack(fill_value=0)
        )
        flat_table = (
            table
            .reset_index()
            .melt(id_vars="species", var_name="island", value_name="count")
            .sort_values(["species", "island"])
            .reset_index(drop=True)
            .sort_values(by=["species", "island"])
        )
        return flat_table

    @output
    @render.data_frame
    @reactive.event(input.scale_gauss)
    def cross_table_dp() -> pd.DataFrame:
        """
        Calcule un tableau croisé bruité entre 'species' et 'island',
        avec ajout de bruit gaussien sur les effectifs.
        """
        # Table originale sans bruit
        table = (
            data_example
            .groupby(["species", "island"])
            .size()
            .unstack(fill_value=0)
        )
        flat_table = (
            table
            .reset_index()
            .melt(id_vars="species", var_name="island", value_name="count")
            .sort_values(["species", "island"])
            .reset_index(drop=True)
        )

        # Ajout de bruit gaussien à la colonne 'count'
        sigma = input.scale_gauss()
        bruit = np.random.normal(loc=0, scale=sigma, size=len(flat_table))
        flat_table["count"] = (flat_table["count"] + bruit).round(0).clip(lower=0).astype(int)

        return flat_table

    @render.ui
    def dp_budget_summary() -> ui.Tag:
        rho = 1 / (2 * input.scale_gauss() ** 2)
        delta_exp = input.delta_slider()
        delta = f"10^{{{delta_exp}}}"
        eps = eps_from_rho_delta(rho, 10**delta_exp)

        box_style = (
            "margin-top:20px; padding:10px; background-color:#f9f9f9; border-radius:12px; "
            'font-family: "Raleway", "Garamond", sans-serif; font-size:16px; color:#333'
        )

        content = textwrap.dedent(f"""
            <div style="{box_style}">
                <p style="margin-bottom:10px">
                    <strong>Budget de confidentialité différentielle :</strong>
                </p>
                <ul style="padding-left: 20px; margin: 0;">
                    <li>En zCDP, \\( \\rho = {rho:.4f} \\)</li>
                    <li>
                        Ou bien, \\( \\varepsilon = {eps:.3f} \\), \\( \\delta = {delta} \\)
                    </li>
                </ul>
            </div>
        """)

        return ui.TagList(
            ui.HTML(content),
            ui.tags.script("if (window.MathJax) MathJax.Hub.Queue(['Typeset', MathJax.Hub]);")
        )

    @render.plot
    def histo_plot() -> plt.Figure:
        return create_histo_plot(data_example, input.alpha_slider())

    @render.plot
    def fc_emp_plot() -> plt.Figure:
        return create_fc_emp_plot(data_example, input.alpha_slider())

    @reactive.calc
    def score_proba_quantile() -> pd.DataFrame:
        nb_candidat = input.candidat_slider()
        alpha = input.alpha_slider()
        epsilon = input.epsilon_slider()
        L, U = input.min_max_slider()

        candidats = np.linspace(L, U, nb_candidat).tolist()
        scores, sensi = manual_quantile_score(data_example['body_mass_g'], candidats, alpha, True)

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

        df = pd.DataFrame({
            "Candidat": candidats,
            "Score": scores,
            "Probabilité": proba,
            "Top95": top95_cumul
        })

        return df

    @render.plot
    def score_plot() -> plt.Figure:
        return create_score_plot(df=score_proba_quantile())

    @render.plot
    def proba_plot() -> plt.Figure:
        return create_proba_plot(df=score_proba_quantile())

    @reactive.calc
    def budgets_par_dataset() -> pd.DataFrame:
        """
        Calcule le budget total par dataset :
        - somme sur 'France entière'
        - maximum des sommes sur les autres échelles géographiques
        Le total est la somme des deux.
        """
        _ = trigger_update_budget()
        try:
            df = pd.read_csv("data/budget_dp.csv")
        except FileNotFoundError:
            return pd.DataFrame(columns=["nom_dataset", "budget_dp_rho"])

        # Budget cumulé pour "France entière"
        df_france = (
            df[df["echelle_geographique"] == "France entière"]
            .groupby("nom_dataset", as_index=False)["budget_dp_rho"]
            .sum()
            .rename(columns={"budget_dp_rho": "budget_france"})
        )
        # Budget cumulé pour chaque autre échelle
        df_autres = (
            df[df["echelle_geographique"] != "France entière"]
            .groupby(["nom_dataset", "echelle_geographique"], as_index=False)["budget_dp_rho"]
            .sum()
        )

        # Pour chaque dataset, on garde le max des autres échelles
        df_max_autres = (
            df_autres.groupby("nom_dataset", as_index=False)["budget_dp_rho"]
            .max()
            .rename(columns={"budget_dp_rho": "budget_max_autres"})
        )

        # Fusion des deux sources, puis somme
        df_merge = pd.merge(df_france, df_max_autres, on="nom_dataset", how="outer").fillna(0)
        df_merge["budget_dp_rho"] = df_merge["budget_france"] + df_merge["budget_max_autres"]

        # Résultat final trié
        df_result = df_merge.sort_values("budget_dp_rho", ascending=False)

        return df_result

    @output
    @render.ui
    def budget_display() -> ui.TagList:
        """
        Affiche les budgets par dataset sous forme de value boxes,
        organisées en lignes de 4 colonnes maximum.
        """
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
    def data_budget_view() -> pd.DataFrame:
        _ = trigger_update_budget()  # Pour prendre en compte la mise à jour du csv
        fichier = Path("data/budget_dp.csv")

        if fichier.exists():
            return pd.read_csv(fichier)
        else:
            return pd.DataFrame(
                columns=["nom_dataset", "echelle_geographique", "date_ajout", "budget_dp_rho"]
            )

    @reactive.effect
    @reactive.event(input.confirm_validation)
    def _() -> None:
        """
        Valide les entrées, ajoute une ligne au CSV si tout est correct, et redirige vers résultats.
        """
        data_requetes = requetes()

        if not assert_or_notify(
            len(data_requetes) > 0,
            "Vous devez rentrer au moins une requête avant d'accéder aux résultats."
        ):
            return

        if not assert_or_notify(
            input.budget_total() > 0,
            "Vous devez valider un budget non nul avant d'accéder aux résultats."
        ):
            return

        if not assert_or_notify(
            input.dataset_name().strip(),
            "Vous devez spécifier un nom au dataset."
        ):
            return

        if not assert_or_notify(
            input.echelle_geo().strip(),
            "Vous devez spécifier l'échelle géographique de l'étude."
        ):
            return

        page_autorisee.set(True)
        ui.modal_remove()
        ui.update_navs("page", selected="Résultat DP")

        ligne = pd.DataFrame([{
            "nom_dataset": input.dataset_name(),
            "echelle_geographique": input.echelle_geo(),
            "date_ajout": datetime.now().strftime("%d/%m/%Y"),
            "budget_dp_rho": input.budget_total()
        }])

        fichier = Path("data/budget_dp.csv")
        if fichier.exists():
            ligne.to_csv(fichier, mode="a", header=False, index=False, encoding="utf-8")
        else:
            ligne.to_csv(fichier, mode="w", header=True, index=False, encoding="utf-8")

        ui.notification_show("✅ Ligne ajoutée à `budget_dp.csv`", type="message")
        trigger_update_budget.set(trigger_update_budget() + 1)  # 🔄 Déclenche la mise à jour

    @reactive.effect
    @reactive.event(input.page)
    def on_tab_change() -> None:
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

    @reactive.effect
    @reactive.event(input.valider_budget)
    def _() -> None:
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

    @reactive.effect
    @reactive.event(input.cancel_validation)
    def _() -> None:
        ui.modal_remove()

    @output
    @render.download(filename=lambda: "resultats_dp.xlsx")
    def download_xlsx() -> io.BytesIO:
        """
        Téléchargement des résultats au format Excel avec une feuille par clé.
        """
        resultats = resultats_df()
        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for key, df in resultats.items():
                nom_feuille = str(key)[:31]  # Limite Excel : 31 caractères max
                df.to_excel(writer, sheet_name=nom_feuille, index=False)

        buffer.seek(0)
        return buffer

    values_buttons_comptage = radio_buttons_server("Comptage", requetes, req_calcul)
    values_buttons_total = radio_buttons_server("Total", requetes, req_calcul)
    values_buttons_moyenne = radio_buttons_server("Moyenne", requetes, req_calcul)
    values_buttons_ratio = radio_buttons_server("Ratio", requetes, req_calcul)
    values_buttons_quantile = radio_buttons_server("Quantile", requetes, req_calcul)

    budget_req_server(
        "Comptage", dataset, requetes, conception_query_count,
        conception_query_sum, conception_query_quantile
    )
    budget_req_server(
        "Total", dataset, requetes, conception_query_count,
        conception_query_sum, conception_query_quantile
    )
    budget_req_server(
        "Moyenne", dataset, requetes, conception_query_count,
        conception_query_sum, conception_query_quantile
    )
    budget_req_server(
        "Ratio", dataset, requetes, conception_query_count,
        conception_query_sum, conception_query_quantile
    )
    budget_req_server(
        "Quantile", dataset, requetes, conception_query_count,
        conception_query_sum, conception_query_quantile
    )

    @reactive.calc
    def get_poids_req() -> dict[str, float]:
        data_requetes = requetes()
        values_buttons = {
            **values_buttons_comptage(), **values_buttons_total(),
            **values_buttons_moyenne(), **values_buttons_ratio(),
            **values_buttons_quantile()
        }
        poids = get_weights(data_requetes, values_buttons)
        return poids

    @render.ui
    def ligne_conditionnelle() -> Optional[ui.TagList]:
        type_req = input.type_req()
        variables = variable_choices().copy()
        ui.update_selectize("group_by", choices=variables)

        if type_req == "Comptage":
            return None

        contenu = affichage_bouton(type_req, variables, choix_quantile)

        return ui.row(*contenu)


app = App(app_ui, server, static_assets=www_dir)
# shiny run --reload --launch-browser app.py
# shiny run --autoreload-port 8000 app.py

# shiny run --port 5000 --host 0.0.0.0 app.py
