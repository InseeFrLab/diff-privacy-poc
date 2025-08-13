import numpy as np
import pandas as pd
import polars as pl
from shiny import ui

from .constant import choix_quantile
from .fonctions import calcul_MCG
from .request_class import Count, DatasetInfo, Mean, Quantile, Query, Ratio, Sum


def find_count_query(query_id: str, internal_queries: dict[str, Query]) -> Count | None:
    return next(
        (q for q in internal_queries.values() if query_id in q.query_ids),
        None
    )


def find_sum_query(query_id: str, variable: str, internal_queries: dict[str, Query]) -> Sum | None:
    return next(
        (q for q in internal_queries.values()
            if query_id in q.query_ids and q.variable == variable),
        None
    )


def compute_count_diagnostics(
    queries: dict[str, Query],
    internal_count_queries: dict[str, Count]
) -> pl.DataFrame:
    """
    Compute diagnostics for differentially private count queries.

    Parameters
    ----------
    queries : dict[str, Query]
        Dictionary of user queries.
    internal_count_queries : dict[str, Count]
        Dictionary of internal Count queries, with noise and group info.

    Returns
    -------
    pl.DataFrame
        A summary table showing, for each count query:
        - Grouping label
        - Filter expression
        - Estimation std deviation
        - Noise std deviation
    """

    count_queries: dict[str, Query] = {
        query_id: query
        for query_id, query in queries.items()
        if isinstance(query, Count)
    }

    results = []

    for query_id in count_queries:
        internal_count_query = find_count_query(query_id, internal_count_queries)
        results.append({
            "requête": query_id,
            "groupement": str(internal_count_query.grouping_label),
            "filtre": internal_count_query.filter_expr,
            "écart type estimation": internal_count_query.scale,
            "écart type bruit": np.sqrt(internal_count_query.sigma2)
        })

    results = pl.DataFrame(results)
    results = results.with_columns([
        pl.col(col).round(1)
        for col, dtype in zip(results.columns, results.dtypes, strict=True)
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
    ])
    return results


def compute_sum_diagnostics(
    lf: pl.LazyFrame,
    queries: dict[str, Query],
    internal_count_queries: dict[str, Count],
    internal_sum_queries: dict[str, Sum]
) -> pl.DataFrame:
    """
    Compute diagnostics for differentially private sum queries, including:
    - Mean coefficient of variation (CV)
    - Mean relative bias
    - Standard deviations of estimation and noise components

    Parameters
    ----------
    lf : pl.LazyFrame
        The input LazyFrame containing the data.
    queries : dict[str, Query]
        Dictionary of all user queries.
    internal_count_queries : dict[str, Count]
        Internal count queries (used for noise on counts).
    internal_sum_queries : dict[str, Sum]
        Internal centered sum queries (used for noise on sum residuals).

    Returns
    -------
    pl.DataFrame
        A summary table with diagnostics per query.
    """

    sum_queries = {k: v for k, v in queries.items() if isinstance(v, Sum)}
    results = []

    for query_id, query in sum_queries.items():
        internal_count_query = find_count_query(query_id, internal_count_queries)
        internal_sum_query = find_sum_query(query_id, query.variable, internal_sum_queries)

        # Noise and scale components
        count_sigma2 = internal_count_query.sigma2
        count_scale = internal_count_query.scale

        sum_sigma2 = internal_sum_query.sigma2
        sum_scale = internal_sum_query.scale

        lower, upper = query.bounds
        midpoint = (lower + upper) / 2

        # Total scale = combination of both noise sources
        count_variance = (count_scale * midpoint) ** 2
        total_scale = np.sqrt(count_variance + sum_scale ** 2)

        # Execute noisy and unbiased queries
        biased_df = query.execute(lf, use_bounds=True)
        unbiased_df = query.execute(lf, use_bounds=False)

        # Compute diagnostics
        cv_list = []
        relative_bias_list = []

        for row_biased, row_unbiased in zip(
            biased_df.iter_rows(named=True),
            unbiased_df.iter_rows(named=True)
        ):
            sum_biased = row_biased["sum"]
            sum_unbiased = row_unbiased["sum"]

            cv = float("inf") if sum_biased == 0 else 100 * total_scale / sum_biased

            bias = sum_biased - sum_unbiased
            relative_bias = 100 * bias / sum_unbiased

            cv_list.append(cv)
            relative_bias_list.append(relative_bias)

        mean_cv = np.mean(cv_list)
        mean_relative_bias = np.mean(relative_bias_list)

        results.append({
            "requête": query_id,
            "variable": query.variable,
            "groupement": str(internal_sum_query.grouping_label),
            "filtre": internal_sum_query.filter_expr,
            "cv moyen (%)": mean_cv,
            "biais relatif moyen (%)": mean_relative_bias,
            "écart type estimation": total_scale,
            "écart type total centré": sum_scale,
            "écart type comptage": count_scale,
            "écart type bruit total centré": np.sqrt(sum_sigma2),
            "écart type bruit comptage": np.sqrt(count_sigma2),

        })

    results = pl.DataFrame(results)
    results = results.with_columns([
        pl.col(col).round(1) for col, dtype in zip(results.columns, results.dtypes, strict=True)
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
    ])
    return results


def compute_mean_diagnostics(
    lf: pl.LazyFrame,
    queries: dict[str, Query],
    internal_count_queries: dict[str, Count],
    internal_sum_queries: dict[str, Sum]
) -> pl.DataFrame:
    """
    Compute diagnostics for differentially private mean queries.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input data.
    queries : dict[str, Query]
        Dictionary of user queries.
    internal_count_queries : dict[str, Count]
        Internal queries used for noisy counts.
    internal_sum_queries : dict[str, Sum]
        Internal queries used for noisy sums.

    Returns
    -------
    pl.DataFrame
        A table with per-query diagnostics:
        - Mean CV
        - Relative bias
        - Decomposition of estimation std deviations
    """

    mean_queries = {
        query_id: query
        for query_id, query in queries.items()
        if isinstance(query, Mean)
    }

    results = []

    for query_id, query in mean_queries.items():

        internal_count_query = find_count_query(query_id, internal_count_queries)
        internal_sum_query = find_sum_query(query_id, query.variable, internal_sum_queries)

        # Noise and scale components
        count_sigma2 = internal_count_query.sigma2
        count_scale = internal_count_query.scale

        sum_sigma2 = internal_sum_query.sigma2
        sum_scale = internal_sum_query.scale

        # Bounds and midpoint
        L, U = query.bounds
        midpoint = (U + L) / 2

        # Estimation std dev
        count_variance = (count_scale * midpoint) ** 2
        total_scale = np.sqrt(count_variance + sum_scale ** 2)

        # Execute biased and unbiased
        biased_df = query.execute(lf, use_bounds=True)
        unbiased_df = query.execute(lf, use_bounds=False)

        list_cv = []
        list_bias_relative = []
        list_variance = []

        for biased_row, unbiased_row in zip(
            biased_df.iter_rows(named=True),
            unbiased_df.iter_rows(named=True), strict=True
        ):
            total_biased = biased_row.get("sum", 0)
            total_true = unbiased_row.get("sum", 0)
            count = unbiased_row.get("count", 1)

            if count == 0:
                list_variance.append(float("inf"))
                list_cv.append(float("inf"))
                list_bias_relative.append(float("inf"))
                continue

            # Variance of noisy mean
            var = (
                ((count * midpoint - total_true) ** 2) * (count_scale ** 2)
                + (count * sum_scale) ** 2
            ) / (count ** 4)

            mean_biased = total_biased / count
            mean_true = total_true / count
            bias = mean_biased - mean_true
            bias_relative = 100 * bias / mean_true if mean_true != 0 else float("inf")
            cv = 100 * np.sqrt(var) / mean_biased if mean_biased != 0 else float("inf")

            list_variance.append(var)
            list_cv.append(cv)
            list_bias_relative.append(bias_relative)

        mean_var = np.mean(list_variance)
        mean_cv = np.mean(list_cv)
        mean_bias_relative = np.mean(list_bias_relative)

        results.append({
            "requête": query_id,
            "variable": query.variable,
            "groupement": str(query.grouping_label),
            "filtre": query.filter_expr,
            "cv moyen (%)": mean_cv,
            "biais relatif moyen (%)": mean_bias_relative,
            "écart type moyen estimation": np.sqrt(mean_var),
            "écart type total": total_scale,
            "écart type comptage": count_scale,
            "écart type total centré": sum_scale,
            "écart type bruit total centré": np.sqrt(count_sigma2),
            "écart type bruit comptage": np.sqrt(sum_sigma2),
        })

    results = pl.DataFrame(results)
    results = results.with_columns([
        pl.col(col).round(1) for col, dtype in zip(results.columns, results.dtypes, strict=True)
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
    ])
    return results


def compute_ratio_diagnostics(
    lf: pl.LazyFrame,
    queries: dict[str, Query],
    internal_count_queries: dict[str, Count],
    internal_sum_queries: dict[str, Sum]
) -> pl.DataFrame:

    ratio_queries = {k: v for k, v in queries.items() if isinstance(v, Ratio)}

    results = []

    for query_id, query in ratio_queries.items():

        internal_count_query = find_count_query(query_id, internal_count_queries)
        internal_sum_num_query = find_sum_query(
            query_id, query.numerator_variable, internal_sum_queries
        )
        internal_sum_denom_query = find_sum_query(
            query_id, query.denominator_variable, internal_sum_queries
        )

        sigma2_comptage = internal_count_query.sigma2
        scale_comptage = internal_count_query.scale

        # Numérateur
        L_num, U_num = internal_sum_num_query.bounds
        m_num = (L_num + U_num) / 2
        var_num_comptage = (scale_comptage * m_num)**2
        scale_total_num = np.sqrt(var_num_comptage + internal_sum_num_query.scale**2)

        # Dénominateur
        L_denom, U_denom = internal_sum_denom_query.bounds
        m_denom = (L_denom + U_denom) / 2
        scale_total_denom = np.sqrt((scale_comptage * m_denom)**2 + internal_sum_denom_query.scale**2)

        # Exécution
        resultat = query.execute(lf, use_bounds=True)
        resultat_non_biaise = query.execute(lf, use_bounds=False)

        list_var = []
        list_cv = []
        list_biais_relatif = []

        for row_biaise, row_non_biaise in zip(
            resultat.iter_rows(named=True),
            resultat_non_biaise.iter_rows(named=True), strict=True
        ):
            num_b = row_biaise.get("sum_numerator", 0)
            denom_b = row_biaise.get("sum_denominator", 1)
            num = row_non_biaise.get("sum_numerator", 0)
            denom = row_non_biaise.get("sum_denominator", 1)

            if denom == 0:
                var = float("inf")
                cv = float("inf")
                biais_relatif = float("inf")
            else:
                var = (((denom * m_num - num * m_denom)**2) * (scale_comptage**2) +
                       (denom * internal_sum_num_query.scale)**2 +
                       (num * internal_sum_denom_query.scale)**2) / denom**4

                ratio_b = num_b / denom_b if denom_b != 0 else float("inf")
                ratio = num / denom
                cv = 100 * np.sqrt(var) / ratio_b if ratio_b != 0 else float("inf")
                biais = ratio_b - ratio
                biais_relatif = 100 * biais / ratio if ratio != 0 else float("inf")

            list_var.append(var)
            list_cv.append(cv)
            list_biais_relatif.append(biais_relatif)

        var_moyenne = np.mean(list_var)
        cv_moyen = np.mean(list_cv)
        biais_relatif_moyen = np.mean(list_biais_relatif)

        results.append({
            "requête": query_id,
            "variable numérateur": query.numerator_variable,
            "variable dénominateur": query.denominator_variable,
            "groupement": str(query.grouping_label),
            "filtre": query.filter_expr,
            "cv moyen (%)": cv_moyen,
            "biais relatif moyen (%)": biais_relatif_moyen,
            "écart type moyen estimation ": np.sqrt(var_moyenne),
            "écart type total numérateur": scale_total_num,
            "écart type total dénominateur": scale_total_denom,
            "écart type total numérateur centré": internal_sum_num_query.scale,
            "écart type total dénominateur centré": internal_sum_denom_query.scale,
            "écart type comptage": scale_comptage,
            "écart type bruit total numérateur centré": np.sqrt(internal_sum_num_query.sigma2),
            "écart type bruit total dénominateur centré": np.sqrt(internal_sum_denom_query.sigma2),
            "écart type bruit comptage": np.sqrt(sigma2_comptage),
        })

    results = pl.DataFrame(results)
    results = results.with_columns([
        pl.col(col).round(1)
        for col, dtype in zip(results.columns, results.dtypes, strict=True)
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
    ])
    return results


def compute_quantile_diagnostics(quantile_queries: dict[str, Quantile]) -> pl.DataFrame:

    results = []
    for query in quantile_queries.values():

        for quantile_key, taille_ic in query.scale.items():
            alpha = float(quantile_key.removeprefix("quantile_"))

            results.append({
                "requête": query.query_ids[0],
                "variable": query.variable,
                "quantile": choix_quantile[alpha],
                "groupement": str(query.grouping_label),
                "filtre": query.filter_expr,
                "taille moyenne IC 95%": taille_ic,
            })

    results = pl.DataFrame(results)
    results = results.with_columns([
        pl.col(col).round(1) for col, dtype in zip(results.columns, results.dtypes, strict=True)
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
    ])
    return results


def run_all_queries(
    dataset_info: DatasetInfo,
    internal_queries: dict[str, Query],
    key_values: dict[str, list[str]] | None = None,
    show_progress: ui.Progress | None = None
) -> dict[str, pd.DataFrame]:

    current_results = {}

    for i, (query_id, query) in enumerate(internal_queries.items(), start=1):

        if show_progress:
            show_progress.set(
                i, message=f"Requête {query_id} — {query.__class__.__name__}",
                detail="Calcul en cours..."
            )

        if "center" in query.execute_dp.__code__.co_varnames:
            df_result = query.execute_dp(dataset_info, key_values=key_values, center=True)
        else:
            df_result = query.execute_dp(dataset_info, key_values=key_values)

        group_by = query.group_by
        if group_by and df_result.shape[1] > 1:
            # Colonnes restantes (dans l'ordre d'origine, sauf celles de `by`)
            remaining_cols = [col for col in df_result.columns if col not in group_by]

            # Réordonner les colonnes
            df_result = df_result[group_by + remaining_cols]
            df_result = df_result.sort(by=group_by)

        current_results[query_id] = df_result.to_pandas()

    return current_results


def finalize_and_optimize_results(
    results_store: dict[str, pd.DataFrame],
    queries: dict[str, Query],
    internal_queries: dict[str, Count | Sum | Quantile],
    key_values: dict[str, list[str]]
) -> dict[str, pl.DataFrame]:

    current_results = results_store
    final_results = {}
    intermed_results = {}

    # Traitement Count
    internal_count_queries = {k: v for k, v in internal_queries.items() if isinstance(v, Count)}
    filtres_uniques = set(query.filter_expr for query in internal_count_queries.values())

    for filtre in filtres_uniques:
        query_filtre = {k: v for k, v in internal_count_queries.items() if v.filter_expr == filtre}
        results_filtre = {k: v for k, v in current_results.items() if k in query_filtre}
        results_filtre = calcul_MCG(results_filtre, key_values, query_filtre, "count")
        intermed_results.update(results_filtre)

    # Traitement Sum
    internal_sum_queries = {k: v for k, v in internal_queries.items() if isinstance(v, Sum)}
    filtres_uniques = set(query.filter_expr for query in internal_sum_queries.values())
    variables_uniques = set(
        getattr(query, "variable", None) for query in internal_sum_queries.values()
    )

    for filtre in filtres_uniques:
        for variable in variables_uniques:
            query_filtre_variable = {
                k: v for k, v in internal_sum_queries.items()
                if getattr(v, "variable", None) == variable and v.filter_expr == filtre
            }
            results_filtre = {
                k: v for k, v in current_results.items() if k in query_filtre_variable
            }
            results_filtre = calcul_MCG(
                results_filtre, key_values, query_filtre_variable, "sum", pos=False
            )
            intermed_results.update(results_filtre)

    for query_id, query in queries.items():
        if isinstance(query, Sum):
            key_query_comptage = next(
                (
                    k for k, v in internal_queries.items()
                    if query_id in v.query_ids and isinstance(v, Count)
                ),
                None
            )
            key_query_total = next(
                (
                    k for k, v in internal_queries.items()
                    if query_id in v.query_ids and isinstance(v, Sum)
                ),
                None
            )
            L, U = query.bounds
            m = (U + L) / 2

            df_result_comptage = intermed_results[key_query_comptage]
            df_result_total = intermed_results[key_query_total]

            # On concatène horizontalement sur l’index (corrigé)
            df_result = pd.concat(
                [df_result_total.reset_index(drop=True), df_result_comptage.reset_index(drop=True)],
                axis=1
            )
            # Supprimer les colonnes en doublon éventuelles
            df_result = df_result.loc[:, ~df_result.columns.duplicated()]

            df_result["sum"] = df_result["sum"] + df_result["count"] * m

        elif isinstance(query, Mean):
            key_query_comptage = next(
                (
                    k for k, v in internal_queries.items()
                    if query_id in v.query_ids and isinstance(v, Count)
                ),
                None
            )
            key_query_total = next(
                (
                    k for k, v in internal_queries.items()
                    if query_id in v.query_ids and isinstance(v, Sum)
                ),
                None
            )
            L, U = query.bounds
            m = (U + L) / 2

            df_result_comptage = intermed_results[key_query_comptage]
            df_result_total = intermed_results[key_query_total]

            # On concatène horizontalement sur l’index (corrigé)
            df_result = pd.concat(
                [df_result_total.reset_index(drop=True), df_result_comptage.reset_index(drop=True)],
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

        elif isinstance(query, Ratio):
            key_query_comptage = next(
                (
                    k for k, v in internal_queries.items()
                    if query_id in v.query_ids and isinstance(v, Count)
                ),
                None
            )
            variable_num = query.numerator_variable
            variable_denom = query.denominator_variable

            L, U = query.numerator_bounds
            m_num = (U + L) / 2
            L, U = query.denominator_bounds
            m_denom = (U + L) / 2

            key_query_total_num = next(
                (
                    k for k, v in internal_queries.items()
                    if query_id in v.query_ids and isinstance(v, Sum) and v.variable == variable_num
                ),
                None
            )
            key_query_total_denom = next(
                (
                    k for k, v in internal_queries.items()
                    if query_id in v.query_ids and isinstance(v, Sum) and v.variable == variable_denom
                ),
                None
            )

            df_result_comptage = intermed_results[key_query_comptage]
            df_result_total_num = intermed_results[key_query_total_num].copy()
            df_result_total_num.rename(columns={"sum": "sum_num"}, inplace=True)
            df_result_total_denom = intermed_results[key_query_total_denom].copy()
            df_result_total_denom.rename(columns={"sum": "sum_denom"}, inplace=True)

            # On concatène horizontalement sur l’index (corrigé)
            df_result = pd.concat(
                [
                    df_result_total_num.reset_index(drop=True),
                    df_result_total_denom.reset_index(drop=True),
                    df_result_comptage.reset_index(drop=True)
                ],
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
            key_query = next(
                (k for k, v in internal_queries.items() if query_id in v.query_ids),
                None
            )

            if isinstance(query, Count):
                df_result = intermed_results[key_query]

            if isinstance(query, Quantile):
                df_result = current_results[key_query]

        df_result = df_result.round(1)

        # Remplace -0.0 par 0.0 dans toutes les colonnes numériques à virgule
        for col in df_result.select_dtypes(include=["float"]).columns:
            df_result[col] = df_result[col].apply(lambda x: 0.0 if x == -0.0 else x)

        if "count" in df_result.columns:
            df_result["count"] = df_result["count"].clip(lower=0)

        final_results[query_id] = pl.from_pandas(df_result)

    return final_results
