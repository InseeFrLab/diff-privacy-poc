from src.request_class import (
    count_dp, sum_centered_dp, quantile_dp
)
from src.fonctions import parse_filter_string
import polars as pl
import numpy as np
import pandas as pd
from src.constant import (
    choix_quantile
)
from typing import Any


def df_comptage(requetes, conception_query_count) -> pd.DataFrame:
    query_comptage = conception_query_count()
    data_requetes = requetes()
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
    query_comptage = conception_query_count()
    query_total = conception_query_sum()
    data_requetes = requetes()
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

                resultat = req.execute(dataset(), use_bounds=True)
                resultat_non_biaise = req.execute(dataset(), use_bounds=False)

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
    query_comptage = conception_query_count()
    query_total = conception_query_sum()
    data_requetes = requetes()
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

                resultat = req.execute(dataset(), use_bounds=True)
                resultat_non_biaise = req.execute(dataset(), use_bounds=False)

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
    query_comptage = conception_query_count()
    query_total = conception_query_sum()
    data_requetes = requetes()
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

                resultat = req.execute(dataset(), use_bounds=True)
                resultat_non_biaise = req.execute(dataset(), use_bounds=False)

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
    query_quantile = conception_query_quantile()

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


async def calculer_toutes_les_requetes(context_rho, context_eps, key_values, dict_query, progress, results_store):
    current_results = {}

    for i, (key, query) in enumerate(dict_query.items(), start=1):

        type_req = query.__class__.__name__
        progress.set(i, message=f"Requête {key} — {type_req}", detail="Calcul en cours...")

        if type_req == "Quantile":
            context_use = context_eps
        else:
            context_use = context_rho

        df_result = query.execute_dp(context_use, key_values)
        by = query.by
        if by and df_result.shape[1] > 1:
            # Colonnes restantes (dans l'ordre d'origine, sauf celles de `by`)
            remaining_cols = [col for col in df_result.columns if col not in by]

            # Réordonner les colonnes
            df_result = df_result[by + remaining_cols]
            df_result = df_result.sort(by=by)

        current_results[key] = df_result.to_pandas()

    results_store.set(current_results)


def process_request(df: pl.LazyFrame, req: dict[str, Any], use_bounds: bool = True) -> pl.LazyFrame:
    """
    Produit le résultat de la requête (sans confidentialité différentielle) sous forme de lazyframe.

    Args:
        df (pl.LazyFrame): Données requêtées.
        req (dict): Dictionnaire contenant les paramètres nécessaires à la requête.
        use_bounds (bool): Booléen indiquant si le clipping doit être appliqué ou non.

    Returns:
        pl.LazyFrame: Résultat de la requête.
    """

    def apply_bounds(df: pl.LazyFrame, var: str, bounds: tuple[float, float]) -> pl.LazyFrame:
        """
        Applique les bornes à une variable si elles sont définies.

        Args:
            frame (pl.LazyFrame): Données requêtées.
            var (str): Nom de la variable clippée.
            bounds (tuple): Bornes min et max de l'intervalle du clipping.

        Returns:
            pl.LazyFrame: Données après clipping de la variable
        """
        if var and bounds:
            lower, upper = bounds
            return df.with_columns(pl.col(var).clip(lower_bound=lower, upper_bound=upper).alias(var))
        return df

    # Extraction des paramètres
    type_req = req.__class__.__name__.lower()
    variable = getattr(req, "variable", None)
    variable_denom = getattr(req, "variable_denominateur", None)
    by = getattr(req, "by", None)
    bounds = getattr(req, "bounds", None)
    bounds_denom = getattr(req, "bounds_denominateur", None)
    filtre = getattr(req, "filtre", None)
    list_alpha = getattr(req, "alpha", None)

    if type_req == "ratio":
        variable = getattr(req, "variable_numerateur", None)
        bounds = getattr(req, "bounds_numerateur", None)

    if filtre:
        df = df.filter(parse_filter_string(filtre))

    # Application des bornes
    if use_bounds:
        df = apply_bounds(df, variable, bounds)
        df = apply_bounds(df, variable_denom, bounds_denom)

    # Construction du corps de la requête
    match type_req:
        case "comptage":
            agg_exprs = [pl.count().alias("count")]

        case "moyenne":
            agg_exprs = [
                pl.col(variable).sum().alias("sum"),
                pl.count().alias("count"),
                pl.col(variable).mean().alias("mean")
            ]

        case "total":
            agg_exprs = [pl.col(variable).sum().alias("sum")]

        case "ratio":
            agg_exprs = [
                pl.col(variable).sum().alias("sum_num"),
                pl.col(variable_denom).sum().alias("sum_denom")
            ]

        case "quantile":
            if not list_alpha:
                raise ValueError("Liste des quantiles `alpha` manquante pour le type 'quantile'")
            agg_exprs = [
                pl.col(variable)
                .quantile(float(alpha), interpolation="nearest")
                .alias(f"quantile_{float(alpha)}")
                for alpha in list_alpha
            ]

        case _:
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
