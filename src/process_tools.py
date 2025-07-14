from src.request_class import (
    count_dp, mean_dp, sum_centered_dp, quantile_dp
)
from src.fonctions import parse_filter_string
import polars as pl
import numpy as np
import pandas as pd
from src.constant import (
    choix_quantile
)


def df_comptage(requetes, conception_query_count) -> pd.DataFrame:
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
                    "filtre": query["filtre"],
                    "écart type estimation": query["scale"],
                    "écart type bruit": np.sqrt(query["sigma2"])
                })
    return pd.DataFrame(results).dropna(axis=1, how="all").round(1)


def df_total(dataset, requetes, conception_query_count, conception_query_sum) -> pd.DataFrame:
    query_comptage = conception_query_count()
    query_total = conception_query_sum()
    data_requetes = requetes()
    req_total = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["total", "sum", "somme"]}

    results = []

    for key, req in req_total.items():

        for query in query_comptage.values():

            if key in query["req"]:
                sigma2_comptage = query["sigma2"]
                scale_comptage = query["scale"]
                break

        variable = req["variable"]

        for query in query_total.values():

            if key in query["req"] and query["variable"] == variable:

                sigma2_total_centre = query["sigma2"]
                scale_total_centre = query["scale"]

                L, U = query["bounds"]

                m = (U + L)/2

                var_comptage = (scale_comptage * m)**2

                scale = np.sqrt(var_comptage + scale_total_centre**2)

                label = f"{variable}<br>groupement: {query["groupement_style"]}"

                resultat = process_request(dataset().lazy(), req)
                resultat_non_biaise = process_request(dataset().lazy(), req, use_bounds=False)

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
                    "variable": variable,
                    "groupement": query["groupement_style"],
                    "filtre": query["filtre"],
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
    query_comptage = conception_query_count()
    query_total = conception_query_sum()
    data_requetes = requetes()
    req_moyenne = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["moyenne", "mean"]}

    results = []

    for key, req in req_moyenne.items():

        for query in query_comptage.values():

            if key in query["req"]:
                sigma2_comptage = query["sigma2"]
                scale_comptage = query["scale"]
                break

        variable = req["variable"]

        for query in query_total.values():

            if key in query["req"] and query["variable"] == variable:

                sigma2_total_centre = query["sigma2"]
                scale_total_centre = query["scale"]

                L, U = query["bounds"]
                m = (U + L)/2

                var_comptage = (scale_comptage * m)**2

                scale_total = np.sqrt(var_comptage + scale_total_centre**2)

                label = f"{variable}<br>groupement: {query["groupement_style"]}"

                resultat = process_request(dataset().lazy(), req)
                resultat_non_biaise = process_request(dataset().lazy(), req, use_bounds=False)

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
                    "variable": variable,
                    "groupement": query["groupement_style"],
                    "filtre": query["filtre"],
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
    query_comptage = conception_query_count()
    query_total = conception_query_sum()
    data_requetes = requetes()
    req_ratio = {k: v for k, v in data_requetes.items() if v["type"].lower() in ["ratio"]}

    results = []

    for key, req in req_ratio.items():

        for query in query_comptage.values():

            if key in query["req"]:
                sigma2_comptage = query["sigma2"]
                scale_comptage = query["scale"]
                break

        variable = req["variable"]

        for query in query_total.values():

            if key in query["req"] and query["variable"] == variable:

                sigma2_total_num_centre = query["sigma2"]
                scale_total_num_centre = query["scale"]

                L, U = query["bounds"]
                m_num = (U + L)/2

                var_num_comptage = (scale_comptage * m_num)**2

                scale_total_num = np.sqrt(var_num_comptage + scale_total_num_centre**2)

                break

        variable_denom = req["variable_denominateur"]

        for query in query_total.values():

            if key in query["req"] and query["variable"] == variable_denom:

                sigma2_total_denom_centre = query["sigma2"]
                scale_total_denom_centre = query["scale"]

                L, U = query["bounds"]
                m_denom = (U + L)/2

                var_denom_comptage = (scale_comptage * m_denom)**2

                scale_total_denom = np.sqrt(var_denom_comptage + scale_total_denom_centre**2)

                label = f"{variable}<br>groupement: {query["groupement_style"]}"

                resultat = process_request(dataset().lazy(), req)
                resultat_non_biaise = process_request(dataset().lazy(), req, use_bounds=False)

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
                    "variable numérateur": variable,
                    "variable dénominateur": variable_denom,
                    "groupement": query["groupement_style"],
                    "filtre": query["filtre"],
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
    query_quantile = conception_query_quantile()

    results = []
    for query in query_quantile.values():
        variable = query["variable"]
        label = f"{variable}<br>groupement: {query['groupement_style']}"

        for quantile_key, taille_ic in query["scale"].items():
            alpha = float(quantile_key.removeprefix("quantile_"))

            results.append({
                "requête": query["req"][0],
                "label": label,
                "quantile": choix_quantile[alpha],
                "groupement": query["groupement_style"],
                "filtre": query.get("filtre"),
                "taille moyenne IC 95%": taille_ic,
            })

    return pd.DataFrame(results).dropna(axis=1, how="all").round(1)


async def calculer_toutes_les_requetes(context_rho, context_eps, key_values, dict_query, progress, results_store):
    current_results = {}

    for i, (key, query) in enumerate(dict_query.items(), start=1):

        type_req = query.get("type", "—")
        progress.set(i, message=f"Requête {key} — {type_req}", detail="Calcul en cours...")

        dp_result = process_request_dp(context_rho, context_eps, key_values, query)
        dp_result = dp_result.execute()
        # print(dp_result.summarize())
        df_result = dp_result.release().collect()

        by = query.get("by")
        if by and df_result.shape[1] > 1:
            # Colonnes restantes (dans l'ordre d'origine, sauf celles de `by`)
            remaining_cols = [col for col in df_result.columns if col not in by]

            # Réordonner les colonnes
            df_result = df_result[by + remaining_cols]
            df_result = df_result.sort(by=by)

        current_results[key] = df_result.to_pandas()

    results_store.set(current_results)


def calcul_requete(requetes, dataset):
    df = dataset.lazy()
    dict_results = {}

    for key, req in requetes.items():
        # Colonne de gauche : paramètres
        resultat = process_request(df, req, use_bounds=False)

        if req.get("by") is not None:
            resultat = resultat.sort(by=req.get("by"))

        resultat = resultat.to_pandas()

        dict_results[key] = resultat

    return dict_results


def process_request_dp(context_rho, context_eps, key_values, req):
    variable = req.get("variable")
    by = req.get("by")
    bounds = req.get("bounds")
    filtre = req.get("filtre")
    alpha = req.get("alpha")
    nb_candidats = req.get("nb_candidats")
    type_req = req["type"]

    mapping = {
            "Comptage": lambda: count_dp(context_rho, key_values, by=by, variable=None, filtre=filtre),
            "Moyenne": lambda: mean_dp(context_rho, key_values, by=by, variable=variable, bounds=bounds, filtre=filtre),
            "Total": lambda: sum_centered_dp(context_rho, key_values, by=by, variable=variable, bounds=bounds, filtre=filtre),
            "Quantile": lambda: quantile_dp(context_eps, key_values, by=by, variable=variable, bounds=bounds, filtre=filtre, alpha=alpha, nb_candidats=nb_candidats)
        }
    class_mapping = {
        "Comptage": count_dp,
        "Moyenne": mean_dp,
        "Total": sum_centered_dp,
        "Quantile": quantile_dp
    }

    if type_req not in mapping:
        raise ValueError(f"Type de requête non supporté : {type_req}")

    return mapping[type_req]() # class_mapping[type_req](**req)


def process_request(df: pl.LazyFrame, req: dict, use_bounds=True) -> pl.LazyFrame:
    variable = req.get("variable")
    variable_denom = req.get("variable_denominateur")
    by = req.get("by")
    bounds = req.get("bounds")
    bounds_denom = req.get("bounds_denominateur")
    filtre = req.get("filtre")
    list_alpha = req.get("alpha")
    type_req = req.get("type", "").lower()

    if filtre:
        df = df.filter(parse_filter_string(filtre))

    # Appliquer les bornes sur la variable principale
    if use_bounds and variable and bounds:
        L, U = bounds
        df = df.with_columns(pl.col(variable).clip(lower_bound=L, upper_bound=U).alias(variable))

    # Appliquer les bornes sur le dénominateur (si ratio)
    if use_bounds and variable_denom and bounds_denom:
        L, U = bounds_denom
        df = df.with_columns(pl.col(variable_denom).clip(lower_bound=L, upper_bound=U).alias(variable_denom))

    # Fonctions de traitement
    if type_req in {"count", "comptage"}:
        agg_exprs = [pl.count().alias("count")]

    elif type_req in {"mean", "moyenne"}:
        agg_exprs = [
            pl.col(variable).sum().alias("sum"),
            pl.count().alias("count"),
            pl.col(variable).mean().alias("mean")
        ]

    elif type_req in {"sum", "total"}:
        agg_exprs = [pl.col(variable).sum().alias("sum")]

    elif type_req in {"ratio"}:
        agg_exprs = [
            pl.col(variable).sum().alias("sum_num"),
            pl.col(variable_denom).sum().alias("sum_denom")
        ]

    elif type_req in {"quantile"}:
        agg_exprs = [
            pl.col(variable).quantile(float(alpha), interpolation="nearest").alias(f"quantile_{float(alpha)}") for alpha in list_alpha
        ]

    else:
        raise ValueError(f"Type de requête inconnu : {req.get('type')}")

    # Appliquer aggregation selon `by`
    if by:
        df = df.group_by(by).agg(agg_exprs).sort(by=by)
    else:
        df = df.select(agg_exprs)

    # Si ratio, ajouter la colonne "ratio"
    if type_req == "ratio":
        df = df.with_columns((pl.col("sum_num") / pl.col("sum_denom")).alias("ratio"))

    return df.collect()
