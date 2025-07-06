from src.request_class import (
    count_dp, mean_dp, sum_centered_dp, quantile_dp
)
from src.fonctions import parse_filter_string
import polars as pl


async def calculer_toutes_les_requetes(context_rho, context_eps, key_values, dict_query, progress, results_store):
    current_results = {}

    for i, (key, query) in enumerate(dict_query.items(), start=1):
        progress.set(i, message=f"Requête {key} — {query.get('type', '—')}", detail="Calcul en cours...")
        # await asyncio.sleep(0.05)

        # resultat_dp = process_request_dp(context_rho, context_eps, key_values, query).execute()
        resultat_dp = process_request_dp(context_rho, context_eps, key_values, query)
        # print(resultat_dp.precision())
        resultat_dp = resultat_dp.execute()
        df_result = resultat_dp.release().collect()

        if query.get("type") == "Moyenne":
            df_result = df_result.with_columns(mean=pl.col.sum / pl.col.count)

        if query.get("by") is not None:
            df_result = df_result.sort(by=query.get("by"))

            if query.get("type") == "Moyenne":
                first_cols = df_result.columns[:2]            # Colonnes 0 et 1
                middle_cols = df_result.columns[2:-1]         # Toutes les colonnes sauf les 2 premières et la dernière
                last_col = df_result.columns[-1:]             # Colonne finale (doit être un Index, pas une chaîne)
                new_order = middle_cols + first_cols + last_col
                df_result = df_result.select(new_order)
            else:
                first_col = df_result.columns[0]
                new_order = df_result.columns[1:] + [first_col]
                df_result = df_result.select(new_order)

        # On stocke tous les résultats dans le dictionnaire
        current_results[key] = df_result.to_pandas()

    # Mise à jour une fois que tous les résultats sont prêts
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

    if type_req not in mapping:
        raise ValueError(f"Type de requête non supporté : {type_req}")

    return mapping[type_req]()


def process_request(df: pl.LazyFrame, req: dict, use_bounds=True) -> pl.LazyFrame:

    variable = req.get("variable")
    by = req.get("by")
    bounds = req.get("bounds")
    filtre = req.get("filtre")
    alpha = req.get("alpha")
    type_req = req["type"]

    if filtre:
        df = df.filter(parse_filter_string(filtre))

    # Appliquer les bornes si variable et bounds sont présents
    if variable and bounds and use_bounds:
        L, U = bounds
        df = df.filter((pl.col(variable) >= L) & (pl.col(variable) <= U))

    # Traitement par type de requête
    if type_req == "count" or type_req == "Comptage":
        if by:
            df = df.group_by(by).agg(pl.count().alias("count"))
        else:
            df = df.select(pl.count())

    elif type_req == "mean" or type_req == "Moyenne":
        if by:
            df = df.group_by(by).agg(
                pl.col(variable).sum().alias("sum"),
                pl.count().alias("count"),
                pl.col(variable).mean().alias("mean"))
        else:
            df = df.select(
                pl.col(variable).sum().alias("sum"),
                pl.count().alias("count"),
                pl.col(variable).mean().alias("mean"))

    elif type_req == "sum" or type_req == "Total":
        if by:
            df = df.group_by(by).agg(pl.col(variable).sum().alias("sum"))
        else:
            df = df.select(pl.col(variable).sum().alias("sum"))

    elif type_req == "quantile" or type_req == "Quantile":

        if by:
            df = df.group_by(by).agg(
                pl.col(variable).quantile(alpha, interpolation="nearest").alias(f"quantile_{alpha}")
            )
        else:
            df = df.select(
                pl.col(variable).quantile(alpha, interpolation="nearest").alias(f"quantile_{alpha}")
            )

    else:
        raise ValueError(f"Type de requête inconnu : {type_req}")

    if by:
        df = df.sort(by=by)

    return df.collect()
