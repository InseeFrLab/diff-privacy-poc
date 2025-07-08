from src.request_class import (
    count_dp, mean_dp, sum_centered_dp, quantile_dp
)
from src.fonctions import parse_filter_string
import polars as pl


async def calculer_toutes_les_requetes(context_rho, context_eps, key_values, dict_query, progress, results_store):
    current_results = {}

    for i, (key, query) in enumerate(dict_query.items(), start=1):
        type_req = query.get("type", "—")
        progress.set(i, message=f"Requête {key} — {type_req}", detail="Calcul en cours...")

        # Traitement DP
        dp_result: request_dp = process_request_dp(context_rho, context_eps, key_values, query)
        # print(resultat_dp.precision())
        dp_result = dp_result.execute()
        df_result = dp_result.release().collect()

        # Tri et réordonnancement si groupement
        by = query.get("by")
        if by and df_result.shape[1] > 1:
            df_result = df_result.sort(by=by)

            first_col = df_result.columns[0]
            other_cols = df_result.columns[1:]
            df_result = df_result.select(other_cols + [first_col])

        current_results[key] = df_result.to_pandas()

    results_store.set(current_results)


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
        "Quantile": lambda: quantile_dp(context_eps, key_values, by=by, variable=variable, bounds=bounds, filtre=filtre, alpha=alpha, nb_candidats=nb_candidats)
    }

    if type_req not in mapping:
        raise ValueError(f"Type de requête non supporté : {type_req}")

    return mapping[type_req](**req)


def process_request(df: pl.LazyFrame, req: dict, use_bounds=True) -> pl.LazyFrame:
    variable = req.get("variable")
    variable_denom = req.get("variable_denominateur")
    by = req.get("by")
    bounds = req.get("bounds")
    bounds_denom = req.get("bounds_denominateur")
    filtre = req.get("filtre")
    alpha = req.get("alpha")
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
            pl.col(variable).quantile(alpha, interpolation="nearest").alias(f"quantile_{alpha}")
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
