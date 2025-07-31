import numpy as np
import pandas as pd
from src.constant import (
    choix_quantile
)
from src.fonctions import (
    calcul_MCG
)


def df_comptage(requetes, conception_query_count) -> pd.DataFrame:
    query_comptage = conception_query_count
    data_requetes = requetes
    req_comptage = {k: v for k, v in data_requetes.items() if v.__class__.__name__ == "Comptage"}
    results = []
    for query in query_comptage.values():

        for req in query.id_req:

            if req in req_comptage:

                results.append({
                    "requête": req,
                    "groupement": query.groupement_style,
                    "filtre": query.filtre,
                    "écart type estimation": query.scale,
                    "écart type bruit": np.sqrt(query.sigma2)
                })
    return pd.DataFrame(results).dropna(axis=1, how="all").round(1)


def df_total(dataset, requetes, conception_query_count, conception_query_sum) -> pd.DataFrame:
    query_comptage = conception_query_count
    query_total = conception_query_sum
    data_requetes = requetes
    req_total = {k: v for k, v in data_requetes.items() if v.__class__.__name__ == "Total"}

    results = []

    for key, req in req_total.items():

        for query in query_comptage.values():

            if key in query.id_req:
                sigma2_comptage = query.sigma2
                scale_comptage = query.scale
                break

        for query in query_total.values():

            if key in query.id_req and query.variable == req.variable:

                sigma2_total_centre = query.sigma2
                scale_total_centre = query.scale

                L, U = query.bounds

                m = (U + L)/2

                var_comptage = (scale_comptage * m)**2

                scale = np.sqrt(var_comptage + scale_total_centre**2)

                label = f"{req.variable}<br>groupement: {query.groupement_style}"

                resultat = req.execute(dataset, use_bounds=True)
                resultat_non_biaise = req.execute(dataset, use_bounds=False)

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
                    "requête": key,
                    "label": label,
                    "variable": req.variable,
                    "groupement": query.groupement_style,
                    "filtre": query.filtre,
                    "cv moyen (%)": cv_moyen,
                    "biais relatif moyen (%)": biais_relatif_moyen,
                    "écart type estimation": scale,
                    "écart type total centré": scale_total_centre,
                    "écart type comptage": scale_comptage,
                    "écart type bruit total centré": np.sqrt(sigma2_total_centre),
                    "écart type bruit comptage": np.sqrt(sigma2_comptage),

                })

                break
    return pd.DataFrame(results).dropna(axis=1, how="all").round(1)


def df_moyenne(dataset, requetes, conception_query_count, conception_query_sum):
    query_comptage = conception_query_count
    query_total = conception_query_sum
    data_requetes = requetes
    req_moyenne = {k: v for k, v in data_requetes.items() if v.__class__.__name__ == "Moyenne"}

    results = []

    for key, req in req_moyenne.items():

        for query in query_comptage.values():

            if key in query.id_req:
                sigma2_comptage = query.sigma2
                scale_comptage = query.scale
                break

        for query in query_total.values():

            if key in query.id_req and query.variable == req.variable:

                sigma2_total_centre = query.sigma2
                scale_total_centre = query.scale

                L, U = query.bounds
                m = (U + L)/2

                var_comptage = (scale_comptage * m)**2

                scale_total = np.sqrt(var_comptage + scale_total_centre**2)

                label = f"{req.variable}<br>groupement: {query.groupement_style}"

                resultat = req.execute(dataset, use_bounds=True)
                resultat_non_biaise = req.execute(dataset, use_bounds=False)

                list_var = []
                list_cv = []
                list_biais_relatif = []

                for row_biaise, row_non_biaise in zip(resultat.iter_rows(named=True), resultat_non_biaise.iter_rows(named=True)):
                    total_biaise = row_biaise.get("sum", 0)
                    total = row_non_biaise.get("sum", 0)
                    count = row_non_biaise.get("count", 1)

                    var = (((count * m - total)**2) * (scale_comptage**2) + (count * scale_total_centre)**2) / count**4 if count != 0 else float("inf")
                    cv = 100 * np.sqrt(var) / (total_biaise / count) if count != 0 else float("inf")

                    biais = (total_biaise - total) / count
                    biais_relatif = 100 * biais / (total/count)

                    list_var.append(var)
                    list_cv.append(cv)
                    list_biais_relatif.append(biais_relatif)

                var_moyenne = np.mean(var)
                cv_moyen = np.mean(list_cv)
                biais_relatif_moyen = np.mean(list_biais_relatif)

                results.append({
                    "requête": key,
                    "label": label,
                    "variable": req.variable,
                    "groupement": query.groupement_style,
                    "filtre": query.filtre,
                    "cv moyen (%)": cv_moyen,
                    "biais relatif moyen (%)": biais_relatif_moyen,
                    "écart type moyen estimation ": np.sqrt(var_moyenne),
                    "écart type total": scale_total,
                    "écart type comptage": scale_comptage,
                    "écart type total centré": scale_total_centre,
                    "écart type bruit total centré": np.sqrt(sigma2_total_centre),
                    "écart type bruit comptage": np.sqrt(sigma2_comptage),
                })

                break

    return pd.DataFrame(results).dropna(axis=1, how="all").round(1)


def df_ratio(dataset, requetes, conception_query_count, conception_query_sum):
    query_comptage = conception_query_count
    query_total = conception_query_sum
    data_requetes = requetes
    req_ratio = {k: v for k, v in data_requetes.items() if v.__class__.__name__ == "Ratio"}

    results = []

    for key, req in req_ratio.items():

        for query in query_comptage.values():

            if key in query.id_req:
                sigma2_comptage = query.sigma2
                scale_comptage = query.scale
                break

        variable_num = req.variable_numerateur

        for query in query_total.values():

            if key in query.id_req and query.variable == variable_num:

                sigma2_total_num_centre = query.sigma2
                scale_total_num_centre = query.scale

                L, U = query.bounds
                m_num = (U + L)/2

                var_num_comptage = (scale_comptage * m_num)**2

                scale_total_num = np.sqrt(var_num_comptage + scale_total_num_centre**2)

                break

        variable_denom = req.variable_denominateur

        for query in query_total.values():

            if key in query.id_req and query.variable == variable_denom:

                sigma2_total_denom_centre = query.sigma2
                scale_total_denom_centre = query.scale

                L, U = query.bounds
                m_denom = (U + L)/2

                var_denom_comptage = (scale_comptage * m_denom)**2

                scale_total_denom = np.sqrt(var_denom_comptage + scale_total_denom_centre**2)

                label = f"{variable_num}<br>sur {variable_denom}<br>groupement: {query.groupement_style}"

                resultat = req.execute(dataset, use_bounds=True)
                resultat_non_biaise = req.execute(dataset, use_bounds=False)

                list_var = []
                list_cv = []
                list_biais_relatif = []

                for row_biaise, row_non_biaise in zip(resultat.iter_rows(named=True), resultat_non_biaise.iter_rows(named=True)):
                    total_num_biaise = row_biaise.get("sum_num", 0)
                    total_num = row_non_biaise.get("sum_num", 0)
                    total_denom_biaise = row_biaise.get("sum_denom", 1)
                    total_denom = row_non_biaise.get("sum_denom", 1)

                    var = (((total_denom * m_num - total_num * m_denom)**2) * (scale_comptage**2) + (total_denom * scale_total_num_centre)**2 + (total_num * scale_total_denom_centre)**2) / total_denom**4 if total_denom != 0 else float("inf")
                    cv = 100 * np.sqrt(var) / (total_num_biaise / total_denom_biaise) if total_denom_biaise != 0 else float("inf")

                    biais = total_num_biaise/total_denom_biaise - total_num/total_denom
                    biais_relatif = 100 * biais / (total_num/total_denom)

                    list_var.append(var)
                    list_cv.append(cv)
                    list_biais_relatif.append(biais_relatif)

                var_moyenne = np.mean(var)
                cv_moyen = np.mean(list_cv)
                biais_relatif_moyen = np.mean(list_biais_relatif)

                results.append({
                    "requête": key,
                    "label": label,
                    "variable numérateur": variable_num,
                    "variable dénominateur": variable_denom,
                    "groupement": query.groupement_style,
                    "filtre": query.filtre,
                    "cv moyen (%)": cv_moyen,
                    "biais relatif moyen (%)": biais_relatif_moyen,
                    "écart type moyen estimation ": np.sqrt(var_moyenne),
                    "écart type total numérateur": scale_total_num,
                    "écart type total dénominateur": scale_total_denom,
                    "écart type total numérateur centré": scale_total_num_centre,
                    "écart type total dénominateur centré": scale_total_denom_centre,
                    "écart type comptage": scale_comptage,
                    "écart type bruit total numérateur centré": np.sqrt(sigma2_total_num_centre),
                    "écart type bruit total dénominateur centré": np.sqrt(sigma2_total_denom_centre),
                    "écart type bruit comptage": np.sqrt(sigma2_comptage),
                })

                break

    return pd.DataFrame(results).dropna(axis=1, how="all").round(1)


def df_quantile(conception_query_quantile) -> pd.DataFrame:
    query_quantile = conception_query_quantile

    results = []
    for query in query_quantile.values():
        variable = query.variable
        label = f"{variable}<br>groupement: {query.groupement_style}"

        for quantile_key, taille_ic in query.scale.items():
            alpha = float(quantile_key.removeprefix("quantile_"))

            results.append({
                "requête": query.id_req[0],
                "label": label,
                "quantile": choix_quantile[alpha],
                "groupement": query.groupement_style,
                "filtre": query.filtre,
                "taille moyenne IC 95%": taille_ic,
            })

    return pd.DataFrame(results).dropna(axis=1, how="all").round(1)


def calculer_toutes_les_requetes(context_rho, context_eps, key_values, dict_query, results_store, progress=None):
    current_results = {}

    for i, (key, query) in enumerate(dict_query.items(), start=1):

        type_req = query.__class__.__name__

        if progress:
            progress.set(i, message=f"Requête {key} — {type_req}", detail="Calcul en cours...")

        if type_req == "Quantile":
            context_use = context_eps
        else:
            context_use = context_rho

        if "centre" in query.execute_dp.__code__.co_varnames:
            df_result = query.execute_dp(context_use, key_values, centre=True)
        else:
            df_result = query.execute_dp(context_use, key_values)

        by = query.by
        if by and df_result.shape[1] > 1:
            # Colonnes restantes (dans l'ordre d'origine, sauf celles de `by`)
            remaining_cols = [col for col in df_result.columns if col not in by]

            # Réordonner les colonnes
            df_result = df_result[by + remaining_cols]
            df_result = df_result.sort(by=by)

        current_results[key] = df_result.to_pandas()

    results_store.update(current_results)


def optimisation_et_assemblage_results(results_store, requetes, data_query, modalite):
    current_results = results_store
    final_results = {}
    intermed_results = {}

    query_comptage = {k: v for k, v in data_query.items() if v.__class__.__name__ == "Comptage"}
    filtres_uniques = set(query.filtre for query in query_comptage.values())

    for filtre in filtres_uniques:
        query_filtre = {
            k: v for k, v in query_comptage.items()
            if v.filtre == filtre
        }

        results_filtre = {k: v for k, v in current_results.items() if k in query_filtre.keys()}
        results_filtre = calcul_MCG(results_filtre, modalite, query_comptage, "count")

        intermed_results.update(results_filtre)

    query_total = {k: v for k, v in data_query.items() if v.__class__.__name__ == "Total"}
    filtres_uniques = set(query.filtre for query in query_total.values())
    variables_uniques = set(getattr(query, "variable", None) for query in query_total.values())

    for filtre in filtres_uniques:
        for variable in variables_uniques:
            query_filtre_variable = {
                k: v for k, v in query_total.items()
                if getattr(v, "variable", None) == variable and v.filtre == filtre
            }
            results_filtre = {k: v for k, v in current_results.items() if k in query_filtre_variable.keys()}
            results_filtre = calcul_MCG(results_filtre, modalite, query_filtre_variable, "sum", pos=False)
            intermed_results.update(results_filtre)

    for key, req in requetes.items():

        if req.__class__.__name__ == "Total":
            key_query_comptage = next(
                (k for k, v in data_query.items() if key in v.id_req and v.__class__.__name__ == "Comptage"),
                None  # valeur par défaut si rien n'est trouvé
            )
            key_query_total = next(
                (k for k, v in data_query.items() if key in v.id_req and v.__class__.__name__ == "Total"),
                None
            )
            L, U = req.bounds
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

        elif req.__class__.__name__ == "Moyenne":
            key_query_comptage = next(
                (k for k, v in data_query.items() if key in v.id_req and v.__class__.__name__ == "Comptage"),
                None  # valeur par défaut si rien n'est trouvé
            )
            key_query_total = next(
                (k for k, v in data_query.items() if key in v.id_req and v.__class__.__name__ == "Total"),
                None
            )
            L, U = req.bounds
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

        elif req.__class__.__name__ == "Ratio":
            key_query_comptage = next(
                (k for k, v in data_query.items() if key in v.id_req and v.__class__.__name__ == "Comptage"),
                None  # valeur par défaut si rien n'est trouvé
            )
            variable_num = req.variable_numerateur
            variable_denom = req.variable_denominateur

            L, U = req.bounds_numerateur
            m_num = (U + L) / 2

            L, U = req.bounds_denominateur
            m_denom = (U + L) / 2

            key_query_total_num = next(
                (k for k, v in data_query.items() if key in v.id_req and v.__class__.__name__ == "Total" and v.variable == variable_num),
                None
            )
            key_query_total_denom = next(
                (k for k, v in data_query.items() if key in v.id_req and v.__class__.__name__ == "Total" and v.variable == variable_denom),
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
            key_query = next((k for k, v in data_query.items() if key in v.id_req), None)

            if req.__class__.__name__ == "Comptage":
                df_result = intermed_results[key_query]

            if req.__class__.__name__ == "Quantile":
                df_result = current_results[key_query]

        df_result = df_result.round(1)

        # Remplace -0.0 par 0.0 dans toutes les colonnes numériques à virgule
        for col in df_result.select_dtypes(include=["float"]).columns:
            df_result[col] = df_result[col].apply(lambda x: 0.0 if x == -0.0 else x)

        if "count" in df_result.columns:
            df_result["count"] = df_result["count"].clip(lower=0)

        final_results[key] = df_result

    results_store.update(final_results)
