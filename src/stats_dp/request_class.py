import polars as pl
from itertools import product
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any, Union
import re
import operator
import opendp.prelude as dp
from opendp.extras.polars import LazyFrameQuery


OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


def clip_variable_within_bounds(
    lf: pl.LazyFrame,
    variable: str,
    bounds: tuple[float, float]
) -> pl.LazyFrame:
    """
    Clips a variable within given bounds if specified.

    Args:
        lf (pl.LazyFrame): Queried data.
        variable (str): Name of the variable to be clipped.
        bounds (tuple): Minimum and maximum bounds for clipping.

    Returns:
        pl.LazyFrame: Data after clipping the variable.
    """
    if variable and bounds:
        lower, upper = bounds
        return lf.with_columns(
            pl.col(variable).clip(lower_bound=lower, upper_bound=upper).alias(variable)
        )
    return lf


def parse_single_condition(
    condition: str
) -> pl.Expr:
    """
    Parses a string condition such as 'age > 18' into a Polars expression.

    Args:
        condition (str): A condition string in the format 'column operator value'.

    Returns:
        pl.Expr: A Polars expression representing the condition.

    Raises:
        ValueError: If the condition string is not valid or cannot be parsed.
    """
    for op_str, op_func in OPS.items():
        if op_str in condition:
            left, right = condition.split(op_str, 1)
            left = left.strip()
            right = right.strip()
            # Handle quoted strings
            if re.match(r"^['\"].*['\"]$", right):
                right = right[1:-1]
            elif re.match(r"^\d+(\.\d+)?$", right):  # numeric value
                right = float(right) if '.' in right else int(right)
            return op_func(pl.col(left), right)
    raise ValueError(f"Invalid condition: {condition}")


def parse_filter_expression(
    filter_string: str,
    available_columns: Optional[list[str]] = None
) -> pl.Expr:
    """
    Parses a combined filter string into a single Polars expression.
    If `available_columns` is provided, checks that the referenced columns exist.

    Args:
        filter_string (str): A filter string using '&' and '|' operators,
            e.g. "age > 18 & gender == 'M'".
        available_columns (Optional[list[str]]): A list of valid column names to validate against.

    Returns:
        pl.Expr: A Polars expression representing the combined filters.

    Raises:
        ValueError: If the filter string is empty, malformed, or references unknown columns.
    """
    tokens = re.split(r'(\s+\&\s+|\s+\|\s+)', filter_string)
    expressions = []
    operators = []

    for token in tokens:
        token = token.strip()
        if token == "&":
            operators.append("&")
        elif token == "|":
            operators.append("|")
        elif token:
            # Check column name before parsing the condition
            for op_str in OPS:
                if op_str in token:
                    left, _ = token.split(op_str, 1)
                    column = left.strip()
                    if available_columns is not None and column not in available_columns:
                        raise ValueError(f"Unknown column in filter: '{column}'")
                    break
            expressions.append(parse_single_condition(token))

    if not expressions:
        raise ValueError("The filter is empty or malformed.")

    expr = expressions[0]
    for op, next_expr in zip(operators, expressions[1:]):
        if op == "&":
            expr = expr & next_expr
        elif op == "|":
            expr = expr | next_expr
    return expr


def generate_public_keys(
    group_by_columns: list[str],
    value_options: dict[str, list[str]]
) -> pl.LazyFrame:
    """
    Generates all combinations of public keys based on the Cartesian product
    of possible values for each grouping column.

    Args:
        group_by_columns (List[str]): The list of column names used for grouping.
        value_options (dict[str, List[str]]): A dictionary mapping each column name
            to the list of its possible values.

    Returns:
        pl.LazyFrame: A lazy Polars DataFrame containing one row per combination
            of values across the specified columns.
    """
    selected_values = [value_options[col] for col in group_by_columns if col in value_options]
    combinations = list(product(*selected_values))  # Cartesian product of values
    public_keys = pl.DataFrame([dict(zip(group_by_columns, comb)) for comb in combinations]).lazy()
    return public_keys


class DatasetInfo():
    """
    Encapsulates dataset metadata and provides methods to create differential privacy contexts
    using either (ε, 0)- or (p)-differential privacy settings.
    """
    def __init__(
        self,
        lf: pl.LazyFrame,
        max_individual_contribution: int = 1,
        max_dataset_size_bound: int = 1
    ):
        """
        Initialize DatasetInfo with dataset and contribution parameters.

        Args:
            lf (pl.LazyFrame): The input lazy Polars DataFrame.
            max_individual_contribution (int): Maximum contribution per individual.
            max_dataset_size (int): Maximum allowed dataset partition size.
        """
        self.lf = lf
        self.max_individual_contribution = max_individual_contribution
        self.max_dataset_size_bound = max_dataset_size_bound
        self.rho_context: Optional[dp.Context] = None
        self.epsilon_context: Optional[dp.Context] = None

    def create_rho_context(
        self,
        rho_budget: float,
        weight_list: list[float]
    ) -> Optional[dp.Context]:
        """
        Create a (p)-differential privacy context using a given rho budget and weights.

        Args:
            rho_budget (float): Total privacy budget under p-DP.
            weight_list (list[float]): Budget split weights per query.

        Returns:
            dp.Context: A differential privacy context object based on p.
        """
        if rho_budget > 0:
            self.rho_context = dp.Context.compositor(
                data=self.lf,
                privacy_loss=dp.loss_of(rho=rho_budget),
                privacy_unit=dp.unit_of(contributions=self.max_individual_contribution),
                split_by_weights=weight_list,
                margins=[dp.polars.Margin(max_partition_length=self.max_dataset_size_bound)]
            )
        return self.rho_context

    def create_epsilon_context(
        self,
        rho_budget: float,
        weight_list: list[float]
    ) -> Optional[dp.Context]:
        """
        Create a (ε, 0)-differential privacy context from a given p budget
        using the conversion ε = √(8p).

        Args:
            rho_budget (float): Total privacy budget under p-DP.
            weight_list (list[float]): Budget split weights per query.

        Returns:
            dp.Context: A differential privacy context object based on ε.
        """
        if rho_budget > 0:
            epsilon_budget = np.sqrt(8 * rho_budget)
            self.epsilon_context = dp.Context.compositor(
                data=self.lf,
                privacy_loss=dp.loss_of(epsilon=epsilon_budget),
                privacy_unit=dp.unit_of(contributions=self.max_individual_contribution),
                split_by_weights=weight_list,
                margins=[dp.polars.Margin(max_partition_length=self.max_dataset_size_bound)]
            )
        return self.epsilon_context


class Query(ABC):
    """
    Abstract base class for a statistical or differentially private query on a dataset.
    """

    def __init__(
        self,
        group_by: Optional[list[str]] = None,
        filter_expr: Optional[str] = None
    ):
        """
        Initialize a query with optional grouping and filtering.

        Args:
            group_by (list[str] | None): List of column names to group by.
            filter_expr (str | None): Optional filter condition as a string expression.
        """
        self.group_by = group_by
        self.filter_expr = filter_expr
        self.query_ids = []
        self.weight = 1

        if group_by is None:
            print(1)
            self.grouping_set = frozenset()
            self.grouping_label = "None"
        else:
            if isinstance(group_by, str):
                self.grouping_set = frozenset([group_by])
                self.grouping_label = group_by
            elif isinstance(group_by, list):
                self.grouping_set = frozenset(group_by)
                self.grouping_label = group_by[0] if len(group_by) == 1 else tuple(group_by)

    @abstractmethod
    def execute(
        self,
        lf: pl.LazyFrame,
        use_bounds: bool
    ) -> pl.DataFrame:
        """
        Execute the non-private query on a LazyFrame.

        Args:
            lf (pl.LazyFrame): The input LazyFrame.
            use_bounds (bool): Whether to clip values using bounds.

        Returns:
            pl.DataFrame: The result of the query.
        """
        pass

    @abstractmethod
    def plan_dp(
        self,
        dataset_info: DatasetInfo,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> LazyFrameQuery:
        """
        Build the differentially private query plan.

        Args:
            dataset_info (DatasetInfo): The dataset metadata and context.
            key_values (dict[str, list[str]] | None): Optional keys for public outputs.

        Returns:
            LazyFrameQuery: The privacy-aware query plan.
        """
        pass

    def precision_opendp(
        self,
        dataset_info: DatasetInfo,
        key_values: Optional[dict[str, list[str]]] = None,
        alpha: float = 0.05
    ) -> pl.DataFrame:
        """
        Return the confidence interval summary of the DP query.

        Args:
            dataset_info (DatasetInfo): Dataset and privacy context.
            key_values (dict[str, list[str]] | None): Optional key filtering.
            alpha (float): Confidence level.

        Returns:
            pl.DataFrame: Confidence intervals for the result.
        """
        return self.plan_dp(dataset_info, key_values).summarize(alpha=alpha)

    def execute_dp(
        self,
        dataset_info: DatasetInfo,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> pl.DataFrame:
        """
        Execute the DP query and return the released results.

        Args:
            dataset_info (DatasetInfo): Dataset and privacy context.
            key_values (dict[str, list[str]] | None): Optional key filtering.

        Returns:
            pl.DataFrame: The final released query result.
        """
        return self.plan_dp(dataset_info, key_values).release().collect()

    def to_query_dict(self) -> dict[str, Any]:
        """
        Convert the query object to a serializable dictionary (excluding technical fields).

        Returns:
            dict[str, Any]: A dictionary representation of the query.
        """
        exclude = {"grouping_set", "grouping_label", "query_ids", "weight", "sigma2", "scale"}
        return {
            "type": self.__class__.__name__,
            **{k: v for k, v in self.__dict__.items() if v is not None and k not in exclude}
        }

    def __repr__(self) -> str:
        """
        String representation of the query object.

        Returns:
            str: A human-readable string version.
        """
        cls_name = self.__class__.__name__
        exclude = {"grouping_set", "grouping_label", "query_ids", "weight", "sigma2", "scale"}
        args = [
            f"{key}={value!r}"
            for key, value in self.__dict__.items()
            if value is not None and key not in exclude
        ]
        return f"{cls_name}({', '.join(args)})"

    def __eq__(self, other: "Query") -> bool:
        """
        Check for equality between two queries, ignoring internal fields.

        Args:
            other (Query): Another query object.

        Returns:
            bool: True if equivalent, False otherwise.
        """
        exclude = {"grouping_set", "grouping_label", "query_ids", "weight", "sigma2", "scale"}
        if not isinstance(other, self.__class__):
            return False

        def normalize(obj):
            if isinstance(obj, list):
                return sorted(obj)
            return obj

        for key in set(self.__dict__) | set(other.__dict__):
            val_self = normalize(self.__dict__.get(key))
            val_other = normalize(other.__dict__.get(key))

            if key not in exclude and val_self != val_other:
                return False
        return True

    def filter_and_group_with_bounds(
        self,
        lf: Union[pl.LazyFrame, LazyFrameQuery],
        *expressions: pl.Expr,
        use_bounds: bool,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> Union[pl.LazyFrame, LazyFrameQuery]:
        """
        Apply filter, bounds clipping, grouping, and optional key join to a LazyFrame.

        Args:
            lf (pl.LazyFrame | LazyFrameQuery): Input frame to transform.
            *expressions (pl.Expr): Expressions to compute.
            use_bounds (bool): Whether to apply bounds clipping.
            key_values (dict[str, list[str]] | None): Optional key filtering.

        Returns:
            pl.LazyFrame | LazyFrameQuery: Transformed LazyFrame.
        """
        if self.filter_expr:
            lf = lf.filter(parse_filter_expression(self.filter_expr))

        if use_bounds:
            if isinstance(self, Ratio):
                lf = clip_variable_within_bounds(
                    lf, self.numerator_variable, self.numerator_bounds
                )
                lf = clip_variable_within_bounds(
                    lf, self.denominator_variable, self.denominator_bounds
                )
            else:
                lf = clip_variable_within_bounds(
                    lf, self.variable, self.bounds
                )

        if self.group_by:
            lf = lf.group_by(self.group_by).agg(*expressions)
            if key_values:
                lf = lf.join(
                    generate_public_keys(group_by_columns=self.group_by, value_options=key_values),
                    on=self.group_by, how="right"
                )

            # Tri après group_by (et join si key_values est présent)
            lf = lf.sort(by=self.group_by)
        else:
            lf = lf.select(*expressions)

        return lf


class Count(Query):

    def plan_dp(
        self,
        dataset_info: DatasetInfo,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> LazyFrameQuery:
        """
        Create the DP query plan for count query.
        """
        context = dataset_info.rho_context
        query = context.query().with_columns(pl.lit(1).alias("count_column"))
        expr = (
            pl.col("count_column")
            .fill_null(0)
            .dp.sum((0, 1))
            .alias("count")
        )

        if key_values is None and self.group_by is not None:
            variables = {val for val in self.group_by}
            df = dataset_info.lf.select([pl.col(v).drop_nulls() for v in variables]).collect()
            key_values = {
                v: sorted(df[v].unique().to_list())
                for v in variables
            }

        query = self.filter_and_group_with_bounds(
            query, expr, use_bounds=False, key_values=key_values
        )
        return query

    def execute(
        self,
        lf: pl.LazyFrame,
        use_bounds: bool = False
    ) -> pl.DataFrame:
        """
        Execute the non-private count query on LazyFrame.
        """
        expr = pl.count().alias("count")
        lf = self.filter_and_group_with_bounds(lf, expr, use_bounds=use_bounds)
        return lf.collect()

    def precision_dp(
        self,
        rho_budget: float
    ) -> float:
        """
        Calculate the variance (sigma squared) of the noise given a privacy budget.
        """
        self.sigma2 = 1 / (2 * rho_budget * self.weight)
        return self.sigma2


class Sum(Query):
    def __init__(
        self,
        variable: str,
        bounds: tuple[float, float],
        group_by: Optional[list[str]] = None,
        filter_expr: Optional[str] = None
    ):
        """
        Initialize a sum query with optional grouping and filtering.

        Args:
            variable (str): Name of the variable to be sum.
            bounds (tuple[float, float]): Minimum and maximum bounds for clipping.
            group_by (list[str] | None): List of column names to group by.
            filter_expr (str | None): Optional filter condition as a string expression.
        """
        super().__init__(group_by=group_by, filter_expr=filter_expr)
        self.variable = variable
        self.bounds = bounds

    def plan_dp(
        self,
        dataset_info: DatasetInfo,
        key_values: Optional[dict[str, list[str]]] = None,
        center: bool = False
    ) -> LazyFrameQuery:
        """
        Plan the differentially private sum query with optional centering.
        """
        lower, upper = self.bounds
        context = dataset_info.rho_context
        query = context.query()

        if center:
            mid = (lower + upper) / 2
            half_range = (upper - lower) / 2
            expr = (
                (pl.col(self.variable) - mid)
                .fill_null(0)
                .fill_nan(0)
                .dp.sum((-half_range, half_range))
                .alias("sum")
            )
        else:
            expr = (
                pl.col(self.variable)
                .fill_null(0)
                .fill_nan(0)
                .dp.sum((lower, upper))
                .alias("sum")
            )

        if key_values is None and self.group_by is not None:
            variables = {val for val in self.group_by}
            df = dataset_info.lf.select([pl.col(v).drop_nulls() for v in variables]).collect()
            key_values = {
                v: sorted(df[v].unique().to_list())
                for v in variables
            }

        query = self.filter_and_group_with_bounds(
            query, expr, use_bounds=False, key_values=key_values
        )
        return query

    def execute_dp(
        self,
        dataset_info: DatasetInfo,
        key_values: Optional[dict[str, list[str]]] = None,
        center: bool = False
    ) -> pl.DataFrame:
        """
        Execute the DP sum query.
        """
        return self.plan_dp(dataset_info, key_values, center=center).release().collect()

    def execute(
        self,
        lf: pl.LazyFrame,
        use_bounds: bool
    ) -> pl.DataFrame:
        """
        Execute a non-private sum and count query.
        """
        expr = (
            pl.col(self.variable).sum().alias("sum"),
            pl.count().alias("count")
        )
        lf = self.filter_and_group_with_bounds(lf, expr, use_bounds=use_bounds)
        return lf.collect()

    def transformation(self) -> tuple[Count, "Sum"]:
        """
        Return a transformation composed of a Count and the Sum itself.
        """
        count_query = Count(group_by=self.group_by, filter_expr=self.filter_expr)
        return (count_query, self)

    def precision_dp(
        self,
        rho_budget: float
    ) -> float:
        """
        Calculate the variance (sigma squared) of the noise for DP sum.
        """
        lower, upper = self.bounds
        self.sigma2 = (upper - lower) ** 2 / (8 * rho_budget * self.weight)
        return self.sigma2


class Mean(Query):
    def __init__(
        self,
        variable: str,
        bounds: tuple[float, float],
        group_by: Optional[list[str]] = None,
        filter_expr: Optional[str] = None
    ):
        """
        Initialize a mean query with optional grouping and filtering.

        Args:
            variable (str): Name of the variable to be average.
            bounds (tuple[float, float]): Minimum and maximum bounds for clipping.
            group_by (list[str] | None): List of column names to group by.
            filter_expr (str | None): Optional filter condition as a string expression.
        """
        super().__init__(group_by=group_by, filter_expr=filter_expr)
        self.variable = variable
        self.bounds = bounds

    def plan_dp(
        self,
        dataset_info: DatasetInfo,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> LazyFrameQuery:
        """
        Plan the differentially private mean query using a centered sum and count.
        """
        lower, upper = self.bounds
        center = (upper + lower) / 2
        half_range = (upper - lower) / 2

        context = dataset_info.rho_context
        query = context.query()

        expr = (
            (pl.col(self.variable) - center)
            .fill_null(0)
            .fill_nan(0)
            .dp.sum(bounds=(-half_range, half_range))
            .alias("centered_sum"),

            pl.col(self.variable)
            .fill_null(1)
            .fill_nan(1)
            .dp.sum(bounds=(1, 1))
            .alias("count")
        )

        if key_values is None and self.group_by is not None:
            variables = {val for val in self.group_by}
            df = dataset_info.lf.select([pl.col(v).drop_nulls() for v in variables]).collect()
            key_values = {
                v: sorted(df[v].unique().to_list())
                for v in variables
            }

        query = self.filter_and_group_with_bounds(
            query, expr, use_bounds=False, key_values=key_values
        )
        return query

    def execute(
        self,
        lf: pl.LazyFrame,
        use_bounds: bool
    ) -> pl.DataFrame:
        """
        Execute the non-private mean query: sum, count and mean.
        """
        expr = (
            pl.col(self.variable).sum().alias("sum"),
            pl.count().alias("count"),
            pl.col(self.variable).mean().alias("mean")
        )
        lf = self.filter_and_group_with_bounds(lf, expr, use_bounds=use_bounds)
        return lf.collect()

    def execute_dp(
        self,
        dataset_info: DatasetInfo,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> pl.DataFrame:
        """
        Execute the differentially private mean query by combining centered sum and count.
        """
        lower, upper = self.bounds
        center = (upper + lower) / 2

        results = (
            self.plan_dp(dataset_info, key_values)
            .release()
            .collect()
            .with_columns(
                mean=(pl.col("centered_sum") / pl.col("count")) + center
            )
        )
        return results

    def transformation(self) -> tuple[Count, Sum]:
        """
        Return a transformation composed of a Count and a Sum.
        """
        count_query = Count(group_by=self.group_by, filter_expr=self.filter_expr)
        sum_query = Sum(
            group_by=self.group_by, filter_expr=self.filter_expr,
            variable=self.variable, bounds=self.bounds
        )
        return (count_query, sum_query)


class Ratio(Query):
    def __init__(
        self,
        numerator_variable: str,
        denominator_variable: str,
        numerator_bounds: tuple[float, float],
        denominator_bounds: tuple[float, float],
        group_by: Optional[list[str]] = None,
        filter_expr: Optional[str] = None
    ):
        super().__init__(group_by=group_by, filter_expr=filter_expr)
        self.numerator_variable = numerator_variable
        self.numerator_bounds = numerator_bounds
        self.denominator_variable = denominator_variable
        self.denominator_bounds = denominator_bounds

    def plan_dp(
        self,
        dataset_info: DatasetInfo,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> LazyFrameQuery:
        """
        Plans the differentially private query for the ratio of sums.
        """
        lower_num, upper_num = self.numerator_bounds
        lower_denom, upper_denom = self.denominator_bounds

        context = dataset_info.rho_context
        query = context.query()

        expr = (
            pl.col(self.numerator_variable)
            .fill_null(0)
            .fill_nan(0)
            .dp.sum(bounds=(lower_num, upper_num))
            .alias("sum_numerator"),

            pl.col(self.denominator_variable)
            .fill_null(0)
            .fill_nan(0)
            .dp.sum(bounds=(lower_denom, upper_denom))
            .alias("sum_denominator")
        )

        if key_values is None and self.group_by is not None:
            variables = {val for val in self.group_by}
            df = dataset_info.lf.select([pl.col(v).drop_nulls() for v in variables]).collect()
            key_values = {
                v: sorted(df[v].unique().to_list())
                for v in variables
            }

        query = self.filter_and_group_with_bounds(
            query, expr, use_bounds=False, key_values=key_values
        )
        return query

    def execute(
        self,
        lf: pl.LazyFrame,
        use_bounds: bool
    ) -> pl.DataFrame:
        """
        Executes the non-private ratio query by computing sums and ratio.
        """
        expr = (
            pl.col(self.numerator_variable).sum().alias("sum_numerator"),
            pl.col(self.denominator_variable).sum().alias("sum_denominator")
        )
        lf = self.filter_and_group_with_bounds(lf, expr, use_bounds=use_bounds)
        lf = lf.with_columns(
            (pl.col("sum_numerator") / pl.col("sum_denominator")).alias("ratio")
        )
        return lf.collect()

    def execute_dp(
        self,
        dataset_info: DatasetInfo,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> pl.DataFrame:
        """
        Executes the differentially private ratio query by dividing noisy sums.
        """
        results = (
            self.plan_dp(dataset_info, key_values)
            .release()
            .collect()
            .with_columns(
                (pl.col("sum_numerator") / pl.col("sum_denominator")).alias("ratio")
            )
        )
        return results

    def transformation(self) -> tuple[Count, Sum, Sum]:
        """
        Returns a transformation composed of a Count and two Sum queries.
        """
        count_query = Count(group_by=self.group_by, filter_expr=self.filter_expr)
        sum_num = Sum(
            group_by=self.group_by,
            filter_expr=self.filter_expr,
            variable=self.numerator_variable,
            bounds=self.numerator_bounds
        )
        sum_denom = Sum(
            group_by=self.group_by,
            filter_expr=self.filter_expr,
            variable=self.denominator_variable,
            bounds=self.denominator_bounds
        )
        return (count_query, sum_num, sum_denom)


class Quantile(Query):
    def __init__(
        self,
        variable: str,
        bounds: tuple[float, float],
        alphas: list[float],
        num_candidates: int,
        group_by: Optional[list[str]] = None,
        filter_expr: Optional[str] = None
    ):
        """
        Initialize Quantile query.

        Parameters:
        - variable: variable on which to compute quantiles
        - bounds: (min, max) bounds for DP quantile mechanism
        - alphas: list of quantile levels (e.g., [0.25, 0.5, 0.75])
        - num_candidates: number of candidate quantile points to consider
        - group_by: optional grouping variables
        - filter_expr: optional filter expression
        """
        super().__init__(group_by=group_by, filter_expr=filter_expr)
        self.variable = variable
        self.bounds = bounds
        self.alphas = alphas if isinstance(alphas, list) else [alphas]
        self.num_candidates = num_candidates

    def plan_dp(
        self,
        dataset_info: DatasetInfo,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> LazyFrameQuery:
        """
        Plans the differentially private quantile queries.
        """
        bounds_min, bounds_max = self.bounds
        candidates = np.linspace(bounds_min, bounds_max, int(self.num_candidates))
        context = dataset_info.epsilon_context
        query = context.query()

        exprs = [
            pl.col(self.variable)
            .fill_null(0)
            .dp.quantile(float(alpha), candidates)
            .alias(f"quantile_{float(alpha)}")
            for alpha in self.alphas
        ]

        query = self.filter_and_group_with_bounds(
            query, *exprs, use_bounds=False, key_values=key_values
        )
        return query

    def execute(
        self,
        lf: pl.LazyFrame,
        use_bounds: bool
    ) -> pl.DataFrame:
        """
        Executes the non-private quantile queries.
        """
        exprs = tuple(
            pl.col(self.variable)
            .quantile(float(alpha), interpolation="nearest")
            .alias(f"quantile_{float(alpha)}")
            for alpha in self.alphas
        )
        lf = self.filter_and_group_with_bounds(lf, *exprs, use_bounds=use_bounds)
        return lf.collect()
