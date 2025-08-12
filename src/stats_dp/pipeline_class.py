import polars as pl
import opendp.prelude as dp
import copy
from .process_tools import (
    run_all_queries, finalize_and_optimize_results,
    compute_count_diagnostics, compute_sum_diagnostics,
    compute_mean_diagnostics, compute_ratio_diagnostics,
    compute_quantile_diagnostics
)
from .fonctions import (
    add_variance, add_confidence_interval, extract_columns_from_filter
)
from .request_class import Count, Sum, Quantile, Query, DatasetInfo
import time
dp.enable_features("contrib")


class QueryPipeline():
    def __init__(
        self,
        queries: dict[str, Query],
        dataset_info: DatasetInfo
    ):
        self.queries = queries
        self.dataset_info = dataset_info
        self.lf = dataset_info.lf
        self.max_individual_contribution = dataset_info.max_individual_contribution
        self.max_dataset_size_bound = dataset_info.max_dataset_size_bound

        variables = {
            var for query in self.queries.values() if query.group_by for var in query.group_by
        }
        df = self.lf.select([pl.col(v).drop_nulls() for v in variables]).collect()
        self.key_values = {
            v: sorted(df[v].unique().to_list())
            for v in variables
        }

        # Deep copy and build internal query dict
        copied_queries = {k: copy.deepcopy(v) for k, v in self.queries.items()}
        self.internal_queries = {}
        i = 1
        for query_id, query in copied_queries.items():
            if not isinstance(query, (Count, Quantile)):
                internal_queries = query.transformation()
            else:
                internal_queries = (query,)

            for internal_query in internal_queries:
                if internal_query not in self.internal_queries.values():
                    internal_query.query_ids.append(query_id)
                    query_key = f"query_{i}"
                    self.internal_queries[query_key] = internal_query
                    i += 1
                else:
                    existing_key = next(
                        k for k, v in self.internal_queries.items() if v == internal_query
                    )
                    self.internal_queries[existing_key].query_ids.append(query_id)

    def execute(
        self,
        use_bounds: bool = False
    ) -> dict[str, pl.DataFrame]:
        """
        Execute all registered requests in the pipeline and return the results.

        Each request is executed on the current LazyFrame `self.lf`, optionally
        using bounds if `use_bounds` is set to True.

        Parameters
        ----------
        use_bounds : bool, optional
            If True, bounds will be applied during request execution. Defaults to False.

        Returns
        -------
        dict[str, pl.DataFrame]
            A dictionary mapping each request key to its result as a Polars DataFrame.
        """
        print("==> Starting execute")
        global_start = time.time()
        result_dict = {}

        for query_id, query in self.queries.items():
            result_dict[query_id] = query.execute(lf=self.lf, use_bounds=use_bounds)

        total_duration = time.time() - global_start
        print(f"<== Finished execute (total time: {total_duration:.2f} seconds)")
        return result_dict

    def execute_dp(
        self,
        rho_budget: float,
        weights: dict[str, float],
        use_gls: bool = True,
        show_progress: bool = False
    ) -> dict[str, pl.DataFrame]:
        """
        Run all differentially private queries using allocated weights and a rho budget.

        This method:
        - Filters only the necessary columns from the LazyFrame
        - Creates privacy contexts for DP mechanisms (rho for count/sum, epsilon for quantile)
        - Computes all queries and applies final formatting and optimization if needed

        Parameters
        ----------
        rho_budget : float
            Total privacy budget to be distributed across all queries.
        weights : dict[str, float]
            Dictionary assigning a weight to each query key.
        use_gls : bool, optional
        show_progress : bool, optional
            Whether to display progress during execution (default: False).

        Returns
        -------
        dict[str, pl.DataFrame]
            A dictionary of Polars DataFrames with results for each type of query.
        """
        print("==> Starting run_dp_queries")
        start_global = time.time()

        internal_queries = self._weighted_queries(rho_budget=rho_budget, weights=weights)
        lazy_data = self.lf

        # Identify required columns from all query objects
        by_vars = {
            var for query in internal_queries.values() if query.group_by for var in query.group_by
        }
        main_vars = {
            v for v in (getattr(query, "variable", None) for query in internal_queries.values())
            if v is not None
        }
        numerator_vars = {
            v for v in (getattr(query, "numerator_variable", None) for query in internal_queries.values())
            if v is not None
        }
        denominator_vars = {
            v for v in (getattr(query, "denominator_variable", None) for query in internal_queries.values())
            if v is not None
        }
        filter_vars = {
            col
            for query in internal_queries.values()
            if getattr(query, "filter_expr", None)
            for col in extract_columns_from_filter(query.filter_expr)
        }

        selected_columns = set(
            by_vars | main_vars | numerator_vars | denominator_vars | filter_vars
        )

        # Minimal LazyFrame filtering
        if not selected_columns:
            filtered_lazy = (
                lazy_data.with_columns(pl.lit(1).alias("__dummy"))
                .select("__dummy")
                .collect()
                .lazy()
            )
        else:
            filtered_lazy = lazy_data.select(selected_columns).collect().lazy()

        # Split weights for rho (count/sum) and epsilon (quantile)
        quantile_queries_weights = [
            query.weight for query in internal_queries.values() if isinstance(query, Quantile)
        ]
        other_queries_weights = [
            query.weight for query in internal_queries.values() if not isinstance(query, Quantile)
        ]

        start_context = time.time()

        self.dataset_info.lf = filtered_lazy
        self.dataset_info.create_rho_context(
            rho_budget * sum(other_queries_weights),
            other_queries_weights
        )
        self.dataset_info.create_epsilon_context(
            rho_budget * sum(quantile_queries_weights),
            quantile_queries_weights
        )

        context_duration = time.time() - start_context
        print(f"✅ Privacy context initialized in {context_duration:.2f} seconds.")

        result_dfs = run_all_queries(
            self.dataset_info, internal_queries, self.key_values, show_progress
        )

        result_dfs = finalize_and_optimize_results(
            result_dfs, self.queries, internal_queries, self.key_values
        )

        total_duration = time.time() - start_global
        print(f"<== Finished run_dp_queries (total time: {total_duration:.2f} seconds)")
        return result_dfs

    def precision_dp(
        self,
        rho_budget: float,
        weights: dict[str, float]
    ) -> dict[str, pl.DataFrame]:
        """
        Compute differentially private statistics (count, sum, mean, ratio, quantile)
        using allocated weights and a rho privacy budget.

        This method:
        - Applies weights to queries
        - Adds variance estimates to Count and Sum queries
        - Computes confidence intervals for Quantile queries using the analytical bound
        with Gaussian noise

        Parameters
        ----------
        rho_budget : float
            Total privacy budget to be distributed across all queries.
        weights : dict[str, float]
            Dictionary assigning a weight to each query identifier.

        Returns
        -------
        dict[str, pl.DataFrame]
            Dictionary containing result DataFrames for each type of query:
            "Count", "Total", "Mean", "Ratio", and "Quantile".
        """
        print("==> Starting compute_dp_precision")
        start_global = time.time()

        internal_queries = self._weighted_queries(rho_budget=rho_budget, weights=weights)

        # Count queries
        internal_count_queries = {k: v for k, v in internal_queries.items() if isinstance(v, Count)}
        internal_count_queries = add_variance(internal_count_queries, self.key_values, rho_budget)

        # Sum queries
        internal_sum_queries = {k: v for k, v in internal_queries.items() if isinstance(v, Sum)}
        internal_sum_queries = add_variance(internal_sum_queries, self.key_values, rho_budget)

        # Quantile queries
        quantile_queries = {k: v for k, v in internal_queries.items() if isinstance(v, Quantile)}
        quantile_queries = add_confidence_interval(self.lf, quantile_queries, rho_budget)

        results = {
            "Count": compute_count_diagnostics(self.queries, internal_count_queries),
            "Sum": compute_sum_diagnostics(self.lf, self.queries, internal_count_queries, internal_sum_queries),
            "Mean": compute_mean_diagnostics(self.lf, self.queries, internal_count_queries, internal_sum_queries),
            "Ratio": compute_ratio_diagnostics(self.lf, self.queries, internal_count_queries, internal_sum_queries),
            "Quantile": compute_quantile_diagnostics(quantile_queries),
        }

        total_duration = time.time() - start_global
        print(f"<== Finished compute_dp_precision (total time: {total_duration:.2f} seconds)")
        return results

    def _weighted_queries(
        self,
        rho_budget: float,
        weights: dict[str, float]
    ) -> dict[str, Query]:
        """
        Assign weights and apply differential privacy precision to each query.

        For each query in the pipeline:
        - The weight is computed as the sum of weights associated with its requested IDs.
        - If the query is of type Count or Sum, the method applies differential privacy
        using the rho budget.

        Parameters
        ----------
        rho_budget : float
            Total privacy budget to be distributed across queries.
        weights_dict : dict[str, float]
            Dictionary mapping query identifiers to their relative weights.

        Returns
        -------
        dict[str, Query]
            The updated dictionary of queries with assigned weights and applied precision
            where appropriate.
        """
        for internal_query in self.internal_queries.values():
            internal_query.weight = sum(
                weights.get(query_id, 0) for query_id in internal_query.query_ids
            )

            if isinstance(internal_query, (Count, Sum)):
                internal_query.precision_dp(rho_budget)

        return self.internal_queries
