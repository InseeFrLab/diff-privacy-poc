# Imports
from src.plots import (
    create_histo_plot, create_fc_emp_plot,
    create_score_plot, create_proba_plot,
    create_barplot, create_grouped_barplot_cv,
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
    process_request, process_request_dp, calculer_toutes_les_requetes
)
from src.fonctions import (
    eps_from_rho_delta, optimization_boosted,
    sys_budget_dp, update_context,
    normalize_weights,
    organiser_par_by, load_data,
    generate_yaml_metadata_from_lazyframe_as_string
)
from src.constant import (
    storage_options
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
    def normalized_weights():
        return normalize_weights(requetes(), input)

    @reactive.Calc
    def X_count():
        if input.budget_total() == 0:
            return None, None, None, None
        data_requetes = requetes()
        poids = normalized_weights()  # Poids normalisé de toutes les requêtes
        nb_modalite = {k: len(v) for k, v in key_values().items()}
        req_comptage = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["count", "comptage"]}
        poids_comptage_req = {k: v for k, v in poids.items() if k in req_comptage.keys()}
        budget_comptage = input.budget_total() * sum(poids_comptage_req.values())
        poids_comptage, lien_comptage_req = organiser_par_by(req_comptage, poids)

        print(f"---------------------------------------------------------------------------------------------------")
        print(f"Cas du comptage")
        rho_estimation, rho_req, poids_estimateur = sys_budget_dp(budget_rho=budget_comptage, nb_modalite=nb_modalite, poids=poids_comptage)
        var_atteint = optimization_boosted(budget_rho=budget_comptage, nb_modalite=nb_modalite, poids=poids_comptage)

        rho_req_comptage = {lien_comptage_req[k]: rho for k, rho in rho_req.items() if k in lien_comptage_req}
        results = []
        for i, (key, request) in enumerate(lien_comptage_req.items()):
            scale = np.sqrt(var_atteint[key])
            results.append({"requête": request, "variable": key, "écart type": scale})

        return results, rho_req_comptage, poids_estimateur, lien_comptage_req

    @reactive.Calc
    def X_total():
        if input.budget_total() == 0:
            return None
        data_requetes = requetes()
        weights = normalized_weights()
        nb_modalite = {k: len(v) for k, v in key_values().items()}
        req_total = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["total", "sum", "somme"]}

        results = []
        rho_req_total = {}
        poids_estimateur_dict = {}
        lien_total_req_dict = {}

        # Récupération des variables distinctes
        variables_uniques = set(v["variable"] for v in req_total.values())

        # Boucle sur chaque variable unique
        for variable in variables_uniques:
            # Sous-ensemble des requêtes ayant cette variable
            req_variable = {
                k: v for k, v in req_total.items()
                if v["variable"] == variable
            }

            poids_total_variable = {k: v for k, v in weights.items() if k in req_variable.keys()}
            budget_total_variable = input.budget_total() * sum(poids_total_variable.values())
            poids_total, lien_total_req = organiser_par_by(req_variable, weights)

            print(f"---------------------------------------------------------------------------------------------------")
            print(f"Cas du total de la variable {variable}")

            rho_estimation, rho_req, poids_estimateur = sys_budget_dp(budget_rho=budget_total_variable, nb_modalite=nb_modalite, poids=poids_total)
            var_atteint = optimization_boosted(budget_rho=budget_total_variable, nb_modalite=nb_modalite, poids=poids_total)

            rho_req_total[variable] = {lien_total_req[k]: rho for k, rho in rho_req.items() if k in lien_total_req}
            poids_estimateur_dict[variable] = poids_estimateur
            lien_total_req_dict[variable] = lien_total_req

            for i, (key_tot, key) in enumerate(lien_total_req.items()):
                request = req_variable[key]
                L, U = request["bounds"]
                scale = max(abs(U), abs(L))*np.sqrt(var_atteint[key_tot])

                resultat = process_request(dataset().lazy(), request)
                resultat_non_biaise = process_request(dataset().lazy(), request, use_bounds=False)

                for row_biaise, row_non_biaise in zip(resultat.iter_rows(named=True), resultat_non_biaise.iter_rows(named=True)):
                    # Extraction des modalités croisées
                    modalites = [str(v) for k, v in row_biaise.items() if k != "sum"]

                    # Construction de l’étiquette
                    if not modalites:
                        label = f"{variable}"
                    else:
                        label = f"{variable} ({', '.join(modalites)})"

                    # Calcul du CV
                    cv = 100 * scale / row_biaise["sum"] if row_biaise["sum"] != 0 else float("inf")

                    biais = row_biaise["sum"] - row_non_biaise["sum"]
                    biais_relatif = 100 * biais / row_non_biaise["sum"]

                    results.append({
                        "requête": key,
                        "label": label,
                        "variable": variable,
                        "modalité": modalites,
                        "cv (%)": cv,
                        "biais relatif (%)": biais_relatif,
                        "écart type": scale,
                        'biais': biais
                    })

        return results, rho_req_total, poids_estimateur_dict, lien_total_req_dict

    @reactive.Calc
    def X_moyenne():
        if input.budget_total() == 0:
            return None

        data_requetes = requetes()
        weights = normalized_weights()
        results = []

        req_moyenne = {
            k: v for k, v in data_requetes.items()
            if v["type"].lower() in ["moyenne"]
        }

        for key, request in req_moyenne.items():
            poids = weights[key]
            v_min, v_max = request["bounds"]
            variable = request.get("variable", "Variable")

            scale_tot = max(abs(v_min), abs(v_max)) / np.sqrt(input.budget_total() * poids /2)
            scale_len = 1 / np.sqrt(input.budget_total() * poids /2)

            resultat = process_request(dataset().lazy(), request)
            resultat_non_biaise = process_request(dataset().lazy(), request, use_bounds=False)
            for row_biaise, row_non_biaise in zip(resultat.iter_rows(named=True), resultat_non_biaise.iter_rows(named=True)):
                total = row_biaise.get("sum", 0)
                total_non_biaise = row_non_biaise.get("sum", 0)
                count = row_biaise.get("count", 1)

                cv_total = scale_tot / total if total != 0 else float("inf")
                cv_count = scale_len / count if count != 0 else float("inf")
                cv = 100 * np.sqrt(cv_total**2 + cv_count**2)

                modalites = [str(v) for k, v in row_biaise.items() if k not in ["sum", "count", "mean"]]

                if not modalites:
                    label = f"{variable}"
                else:
                    label = f"{variable} ({', '.join(modalites)})"

                biais = (total - total_non_biaise) / count
                biais_relatif = 100 * biais / (total_non_biaise/count)

                results.append({
                    "requête": key,
                    "label": label,
                    "variable": variable,
                    "modalité": modalites,
                    "cv (%)": cv,
                    "biais relatif (%)": biais_relatif,
                    "biais": biais,
                    "cv total (%)": 100*cv_total,
                    "écart type total": scale_tot,
                    "cv comptage (%)": 100*cv_count,
                    "écart type comptage": scale_len
                })

            results.sort(key=lambda x: x["cv (%)"])

        return results

    @reactive.Calc
    def X_quantile():
        if input.budget_total() == 0:
            return None
        data_requetes = requetes()
        weights = normalized_weights()
        results = []
        req_quantile = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["quantile"]}
        for i, (key, request) in enumerate(req_quantile.items()):
            poids = weights[key]

            context_param = {
                "data": dataset().lazy(),
                "privacy_unit": dp.unit_of(contributions=1),
                "margins": [dp.polars.Margin(max_partition_length=10000)],
            }

            context_comptage, context_tot, context_moy, context_quantile = update_context(context_param, input.budget_total(), 0, 0, [], [], [], [1])

            resultat_dp = process_request_dp(context_comptage, context_tot, context_moy, context_quantile, key_values(), request)
            intervalle_candidats = resultat_dp.precision(
                data=dataset().lazy(),
                epsilon=np.sqrt(8 * input.budget_total() * poids)
            )
            results.append({"requête": key, "candidats": intervalle_candidats})

        return results

    @output
    @render.ui
    @reactive.event(input.confirm_validation)
    async def req_dp_display():
        # --- Barre de progression ---
        data_requetes = requetes()
        with ui.Progress(min=0, max=len(data_requetes)) as p:
            p.set(0, message="Traitement en cours...", detail="Analyse requête par requête...")
            weights = normalized_weights()

            _, rho_req_comptage, poids_estimateur, lien_comptage_req = X_count()

            poids_rho_req_comptage = [rho for rho in rho_req_comptage.values()]

            _, rho_req_total_dict, poids_estimateur_dict, lien_total_req_dict = X_total()

            flat_dict = {}
            for subdict in rho_req_total_dict.values():
                flat_dict.update(subdict)

            # Trier par ordre des clés 'req_N' selon le numéro
            poids_rho_req_total = [
                flat_dict[k] for k in sorted(flat_dict, key=lambda x: int(x.split("_")[1]))
            ]

            poids_requetes_comptage = [
                weights[clef]
                for clef, requete in data_requetes.items()
                if requete["type"].lower() in ["count", "comptage"]
            ]

            poids_requetes_total = [
                weights[clef]
                for clef, requete in data_requetes.items()
                if requete["type"].lower() in ["total", "somme", "sum"]
            ]

            poids_requetes_moyenne = [
                weights[clef]
                for clef, requete in data_requetes.items()
                if requete["type"].lower() in ["moyenne", "mean"]
            ]

            poids_requetes_quantile = [
                weights[clef]
                for clef, requete in data_requetes.items()
                if requete["type"].lower() == "quantile"
            ]

            context_param = {
                "data": dataset().lazy(),
                "privacy_unit": dp.unit_of(contributions=1),
                "margins": [dp.polars.Margin(max_partition_length=70_000_000)],
            }

            budget_comptage = sum(poids_requetes_comptage) * input.budget_total()

            budget_totaux = sum(poids_requetes_total) * input.budget_total()

            context_comptage, context_tot, context_moy, context_quantile = update_context(
                context_param, input.budget_total(), budget_comptage, budget_totaux, poids_rho_req_comptage, poids_rho_req_total, poids_requetes_moyenne, poids_requetes_quantile
            )
            await calculer_toutes_les_requetes(context_comptage, context_tot, context_moy, context_quantile, key_values(), data_requetes, p, resultats_df, dataset(), rho_req_comptage, rho_req_total_dict)

        return afficher_resultats(resultats_df, requetes(), poids_estimateur, poids_estimateur_dict, lien_comptage_req, lien_total_req_dict)

    @output
    @render.data_frame
    def table_comptage():
        result, _, _, _ = X_count()
        df = pd.DataFrame(result).round(1)
        return df

    @output
    @render.data_frame
    def table_total():
        result, _, _, _ = X_total()
        df = pd.DataFrame(result).drop(columns="label").round(0)
        return df

    @output
    @render.data_frame
    def table_moyenne():
        df = pd.DataFrame(X_moyenne()).drop(columns="label").round(0)
        return df

    @output
    @render.data_frame
    def table_quantile():
        df = pd.DataFrame(X_quantile()).round(1)
        return df

    @render_widget
    def plot_comptage():
        result, _, _, _ = X_count()
        df = pd.DataFrame(result)
        return create_barplot(df, x_col="requête", y_col="écart type", hoover="variable")

    @render.plot
    def graphe_plot():
        if input.budget_total() == 0:
            return None, None, None, None
        data_requetes = requetes()
        poids = normalized_weights()
        req_comptage = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["count", "comptage"]}
        poids_comptage, lien_comptage_req = organiser_par_by(req_comptage, poids)
        poids_set = {
            frozenset() if k == 'Total' else frozenset([k]) if isinstance(k, str) else frozenset(k): v
            for k, v in poids_comptage.items()
        }
        liste_request = [s for s in poids_set.keys()]
        return plot_subset_tree(liste_request, taille_noeuds=2000, keep_gris=False)

    @render_widget
    def plot_total():
        result, _, _, _ = X_total()
        df = pd.DataFrame(result)
        return create_grouped_barplot_cv(df)

    @render_widget
    def plot_moyenne():
        df = pd.DataFrame(X_moyenne())
        return create_grouped_barplot_cv(df)

    @render_widget
    def plot_quantile():
        df = pd.DataFrame(X_quantile())
        return create_barplot(df, x_col="requête", y_col="candidats", hoover=None)


    # Page 1 ----------------------------------

    # Lire le dataset si importé sinon dataset déjà en mémoire
    @reactive.Calc
    def dataset():
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
        # Charger les métadonnées depuis ta chaîne YAML
        metadata_dict = yaml.safe_load(yaml_metadata_str()) if yaml_metadata_str() else {}
        if 'columns' not in metadata_dict:
            return {}

        # Extraire les colonnes qualitatives
        qualitative_cols = [
            col for col, meta in metadata_dict['columns'].items()
            if meta.get('type') in ('Utf8', 'Categorical', 'Boolean', 'str', 'String')
        ]

        # Récupérer les modalités uniques stockées dans 'unique_values_list'
        modalities = {}
        for col in qualitative_cols:
            modal_list = metadata_dict['columns'][col].get('unique_values_list', [])
            print(modal_list)
            if modal_list:
                modalities[col] = sorted(modal_list)
            else:
                modalities[col] = []
        return modalities

    @reactive.Effect
    def update_variable_choices():
        if input.type_req() == "Comptage":
            ui.update_selectize("variable", choices={})  # pas de choix possible
        else:
            # Met à jour dynamiquement les choix de la selectize input
            ui.update_selectize("variable", choices=variable_choices())
            ui.update_selectize("group_by", choices=variable_choices())

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
                "candidat": input.nb_candidat(),
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
                    existing_req.get("candidat") == clean_dict.get("candidat")
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

    @reactive.effect
    def disable_variable_if_comptage():
        # Crée un message JS pour désactiver ou activer l'élément
        is_comptage = input.type_req() == "Comptage"
        session.send_input_message("variable", {"disabled": is_comptage})

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
            # Afficher modal pour prévenir
            ui.modal_show(
                ui.modal(
                    "Vous devez valider le budget avant d'accéder aux résultats.",
                    title="Accès refusé",
                    easy_close=True,
                    footer=None
                )
            )
            # Remettre l'onglet actif sur l'onglet précédent (empêche le changement)
            ui.update_navs("page", selected=onglet_actuel())
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
