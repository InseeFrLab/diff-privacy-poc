import polars as pl
from itertools import product
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any, Union
import re
import operator
import opendp.prelude as dp
from opendp.extras.polars import LazyFrameQuery


# Map des opérateurs Python vers leurs fonctions correspondantes
OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


def apply_bounds(
    lf: pl.LazyFrame,
    var: str,
    bounds: tuple[float, float]
) -> pl.LazyFrame:
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
        return lf.with_columns(pl.col(var).clip(lower_bound=lower, upper_bound=upper).alias(var))
    return lf


def parse_single_condition(
    condition: str
) -> pl.Expr:
    """Transforme une condition string comme 'age > 18' en pl.Expr."""
    for op_str, op_func in OPS.items():
        if op_str in condition:
            left, right = condition.split(op_str, 1)
            left = left.strip()
            right = right.strip()
            # Gère les chaînes entre guillemets simples ou doubles
            if re.match(r"^['\"].*['\"]$", right):
                right = right[1:-1]
            elif re.match(r"^\d+(\.\d+)?$", right):  # nombre
                right = float(right) if '.' in right else int(right)
            return op_func(pl.col(left), right)
    raise ValueError(f"Condition invalide : {condition}")


def parse_filter_string(
    filter_str: str,
    columns: Optional[list[str]] = None
) -> pl.Expr:
    """Transforme une chaîne de filtres combinés en une unique pl.Expr.
    Si `columns` est fourni, vérifie que les colonnes mentionnées existent."""
    tokens = re.split(r'(\s+\&\s+|\s+\|\s+)', filter_str)
    exprs = []
    ops = []

    for token in tokens:
        token = token.strip()
        if token == "&":
            ops.append("&")
        elif token == "|":
            ops.append("|")
        elif token:
            # Avant d'appeler parse_single_condition, on vérifie le nom de la colonne
            for op_str in OPS:
                if op_str in token:
                    left, _ = token.split(op_str, 1)
                    col = left.strip()
                    if columns is not None and col not in columns:
                        raise ValueError(f"Colonne inconnue dans le filtre : '{col}'")
                    break
            exprs.append(parse_single_condition(token))

    if not exprs:
        raise ValueError("Le filtre est vide ou mal formé")

    expr = exprs[0]
    for op, next_expr in zip(ops, exprs[1:]):
        if op == "&":
            expr = expr & next_expr
        elif op == "|":
            expr = expr | next_expr

    return expr


def generate_public_keys(
    by_keys: list[str],
    key_values: dict[str, list[str]]
) -> pl.LazyFrame:
    # Ne garder que les colonnes utiles pour le group_by
    values = [key_values[key] for key in by_keys if key in key_values]
    combinaisons = list(product(*values))  # Produit cartésien des valeurs
    public_keys = pl.DataFrame([dict(zip(by_keys, comb)) for comb in combinaisons]).lazy()
    return public_keys


class InfoDataset():
    def __init__(
        self,
        lf: pl.LazyFrame,
        contribution_individu_max: int = 1,
        borne_max_taille_dataset: int = 1
    ):

        self.lf = lf
        self.contribution_individu_max = contribution_individu_max
        self.borne_max_taille_dataset = borne_max_taille_dataset
        self.context_rho = None
        self.context_eps = None

    def create_context_rho(
        self,
        budget_rho: float,
        list_poids: list[float]
    ) -> dp.Context:

        if budget_rho > 0:

            self.context_rho = dp.Context.compositor(
                data=self.lf,
                privacy_loss=dp.loss_of(rho=budget_rho),
                privacy_unit=dp.unit_of(contributions=self.contribution_individu_max),
                split_by_weights=list_poids,
                margins=[dp.polars.Margin(max_partition_length=self.borne_max_taille_dataset)]
            )

        return self.context_rho

    def create_context_eps(
        self,
        budget_rho: float,
        list_poids: list[float]
    ) -> dp.Context:

        if budget_rho > 0:

            budget_eps = np.sqrt(8*budget_rho)

            self.context_eps = dp.Context.compositor(
                data=self.lf,
                privacy_loss=dp.loss_of(epsilon=budget_eps),
                privacy_unit=dp.unit_of(contributions=self.contribution_individu_max),
                split_by_weights=list_poids,
                margins=[dp.polars.Margin(max_partition_length=self.borne_max_taille_dataset)]
            )

        return self.context_eps


class Query(ABC):
    def __init__(
        self,
        by: Optional[list[str]] = None,
        filtre: Optional[str] = None
    ):
        self.by = by
        self.filtre = filtre
        self.id_req = []
        self.poids = 1
        if by is None:
            self.groupement = frozenset()
            self.groupement_style = 'Aucun'
        else:
            if isinstance(by, str):
                self.groupement = frozenset([by])
                self.groupement_style = by
            elif isinstance(by, list):
                self.groupement = frozenset(by)
                self.groupement_style = by[0] if len(by) == 1 else tuple(by)

    @abstractmethod
    def execute(
        self,
        lf: pl.LazyFrame,
        use_bounds: bool
    ) -> pl.DataFrame:
        pass

    @abstractmethod
    def plan_dp(
        self,
        info_dataset: InfoDataset,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> LazyFrameQuery:
        pass

    def precision_opendp(
        self,
        info_dataset: InfoDataset,
        key_values: Optional[dict[str, list[str]]] = None,
        alpha: float = 0.05
    ) -> pl.DataFrame:
        return self.plan_dp(info_dataset, key_values).summarize(alpha=alpha)

    def execute_dp(
        self,
        info_dataset: InfoDataset,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> pl.DataFrame:
        return self.plan_dp(info_dataset, key_values).release().collect()

    def to_query_dict(self) -> dict[str, Any]:
        exclure = {"groupement", "groupement_style", "id_req", "poids", "sigma2", "scale"}
        return {
            "type": self.__class__.__name__,
            **{k: v for k, v in self.__dict__.items() if v is not None and k not in exclure}
        }

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        exclure = {"groupement", "groupement_style", "id_req", "poids", "sigma2", "scale"}
        args = [
            f"{key}={value!r}"
            for key, value in self.__dict__.items()
            if value is not None and key not in exclure
        ]
        return f"{cls_name}({', '.join(args)})"

    def __eq__(
        self,
        other: "Query"
    ) -> bool:
        exclure = {"groupement", "groupement_style", "id_req", "poids", "sigma2", "scale"}
        if not isinstance(other, self.__class__):
            return False

        def normalize(obj):
            if isinstance(obj, list):
                return sorted(obj)
            return obj

        for key in set(self.__dict__) | set(other.__dict__):
            val_self = normalize(self.__dict__.get(key))
            val_other = normalize(other.__dict__.get(key))

            if key not in exclure and val_self != val_other:
                return False
        return True

    def filtre_bounds_by(
        self,
        lf: Union[pl.LazyFrame, LazyFrameQuery],
        *expr,
        use_bounds: bool,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> Union[pl.LazyFrame, LazyFrameQuery]:

        if self.filtre:
            lf = lf.filter(parse_filter_string(self.filtre))
        if use_bounds:
            if isinstance(self, Ratio):
                lf = apply_bounds(lf, self.variable_numerateur, self.bounds_numerateur)
                lf = apply_bounds(lf, self.variable_denominateur, self.bounds_denominateur)
            else:
                lf = apply_bounds(lf, self.variable, self.bounds)
        if self.by:
            lf = lf.group_by(self.by).agg(*expr)
            if key_values:
                lf = lf.join(
                    generate_public_keys(by_keys=self.by, key_values=key_values),
                    on=self.by, how="right"
                )
        else:
            lf = lf.select(*expr)
        return lf


class Count(Query):

    def plan_dp(
        self,
        info_dataset: InfoDataset,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> LazyFrameQuery:

        context = info_dataset.context_rho
        query = context.query().with_columns(pl.lit(1).alias("colonne_comptage"))
        expr = (
            pl.col("colonne_comptage")
            .fill_null(0)
            .dp.sum((0, 1))
            .alias("count")
        )
        if key_values is None and self.by is not None:
            variables = {val for val in self.by}
            df = info_dataset.lf.select([pl.col(v).drop_nulls() for v in variables]).collect()
            key_values = {
                v: sorted(df[v].unique().to_list())
                for v in variables
            }
        query = self.filtre_bounds_by(query, expr, use_bounds=False, key_values=key_values)
        return query

    def execute(
        self,
        lf: pl.DataFrame,
        use_bounds: bool = False
    ) -> pl.DataFrame:

        expr = (
            pl.count().alias("count")
        )
        print(type(lf))
        lf = self.filtre_bounds_by(lf, expr, use_bounds=use_bounds)
        return lf.collect()

    def precision_dp(
        self,
        budget: float
    ) -> float:

        self.sigma2 = 1/(2 * budget * self.poids)
        return self.sigma2


class Sum(Query):
    def __init__(
        self,
        variable: str,
        bounds: tuple[float, float],
        by: Optional[list[str]] = None,
        filtre: Optional[str] = None
    ):
        super().__init__(by=by, filtre=filtre)
        self.variable = variable
        self.bounds = bounds

    def plan_dp(
        self,
        info_dataset: InfoDataset,
        key_values: Optional[dict[str, list[str]]] = None,
        centre: bool = False
    ) -> LazyFrameQuery:

        l, u = self.bounds
        context = info_dataset.context_rho
        query = context.query()

        if centre:
            center = (l + u)/2
            half_range = (u - l)/2
            expr = (
                (pl.col(self.variable) - center)
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
                .dp.sum((l, u))
                .alias("sum")
            )

        if key_values is None and self.by is not None:
            variables = {val for val in self.by}
            df = info_dataset.lf.select([pl.col(v).drop_nulls() for v in variables]).collect()
            key_values = {
                v: sorted(df[v].unique().to_list())
                for v in variables
            }

        query = self.filtre_bounds_by(query, expr, use_bounds=False, key_values=key_values)
        return query

    def execute_dp(
        self,
        info_dataset: InfoDataset,
        key_values: Optional[dict[str, list[str]]] = None,
        centre: bool = False
    ) -> pl.DataFrame:
        return self.plan_dp(info_dataset, key_values, centre=centre).release().collect()

    def execute(
        self,
        lf: pl.LazyFrame,
        use_bounds: bool
    ) -> pl.DataFrame:

        expr = (
            pl.col(self.variable).sum().alias("sum"),
            pl.count().alias("count")
        )
        lf = self.filtre_bounds_by(lf, expr, use_bounds=use_bounds)
        return lf.collect()

    def transformation(self) -> (Count, "Sum"):
        """
        Retourne une transformation composée d’un Comptage et du Total lui-même.
        """
        comptage = Count(by=self.by, filtre=self.filtre)
        return (comptage, self)

    def precision_dp(
        self,
        budget: float
    ) -> float:
        l, u = self.bounds
        self.sigma2 = (u - l)**2/(4 * 2 * budget * self.poids)
        return self.sigma2


class Mean(Query):
    def __init__(
        self,
        variable: str,
        bounds: tuple[float, float],
        by: Optional[list[str]] = None,
        filtre: Optional[str] = None
    ):
        super().__init__(by=by, filtre=filtre)
        self.variable = variable
        self.bounds = bounds

    def plan_dp(
        self,
        info_dataset: InfoDataset,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> LazyFrameQuery:

        l, u = self.bounds
        center = (u + l) / 2
        half_range = (u - l) / 2
        context = info_dataset.context_rho
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

        if key_values is None and self.by is not None:
            variables = {val for val in self.by}
            df = info_dataset.lf.select([pl.col(v).drop_nulls() for v in variables]).collect()
            key_values = {
                v: sorted(df[v].unique().to_list())
                for v in variables
            }

        query = self.filtre_bounds_by(query, expr, use_bounds=False, key_values=key_values)
        return query

    def execute(
        self,
        lf: pl.LazyFrame,
        use_bounds: bool
    ) -> pl.DataFrame:

        expr = (
            pl.col(self.variable).sum().alias("sum"),
            pl.count().alias("count"),
            pl.col(self.variable).mean().alias("mean")
        )
        lf = self.filtre_bounds_by(lf, expr, use_bounds=use_bounds)
        return lf.collect()

    def execute_dp(
        self,
        info_dataset: InfoDataset,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> pl.DataFrame:

        l, u = self.bounds
        center = (u + l) / 2

        results = (
            self.plan_dp(info_dataset, key_values)
            .release()
            .collect()
            .with_columns(
                mean=(pl.col("centered_sum") / pl.col("count")) + center
            )
        )
        return results

    def transformation(self) -> (Count, Sum):
        """
        Retourne une transformation composée d’un Comptage et d'un Total.
        """
        comptage = Count(by=self.by, filtre=self.filtre)
        total = Sum(by=self.by, filtre=self.filtre, variable=self.variable, bounds=self.bounds)
        return (comptage, total)


class Ratio(Query):
    def __init__(
        self,
        variable: str,
        variable_denominateur: str,
        bounds: tuple[float, float],
        bounds_denominateur: tuple[float, float],
        by: Optional[list[str]] = None,
        filtre: Optional[str] = None
    ):
        super().__init__(by=by, filtre=filtre)
        self.variable_numerateur = variable
        self.bounds_numerateur = bounds
        self.variable_denominateur = variable_denominateur
        self.bounds_denominateur = bounds_denominateur

    def plan_dp(
        self,
        info_dataset: InfoDataset,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> LazyFrameQuery:

        l_num, u_num = self.bounds_numerateur
        l_denom, u_denom = self.bounds_denominateur
        context = info_dataset.context_rho
        query = context.query()
        expr = (
            pl.col(self.variable_numerateur)
            .fill_null(0)
            .fill_nan(0)
            .dp.sum(bounds=(l_num, u_num))
            .alias("sum_numerateur"),

            pl.col(self.variable_denominateur)
            .fill_null(0)
            .fill_nan(0)
            .dp.sum(bounds=(l_denom, u_denom))
            .alias("sum_denominateur")
        )

        if key_values is None and self.by is not None:
            variables = {val for val in self.by}
            df = info_dataset.lf.select([pl.col(v).drop_nulls() for v in variables]).collect()
            key_values = {
                v: sorted(df[v].unique().to_list())
                for v in variables
            }

        query = self.filtre_bounds_by(query, expr, use_bounds=False, key_values=key_values)
        return query

    def execute(
        self,
        lf: pl.LazyFrame,
        use_bounds: bool
    ) -> pl.DataFrame:

        expr = (
            pl.col(self.variable_numerateur).sum().alias("sum_num"),
            pl.col(self.variable_denominateur).sum().alias("sum_denom")
        )
        lf = self.filtre_bounds_by(lf, expr, use_bounds=use_bounds)
        lf = lf.with_columns((pl.col("sum_num") / pl.col("sum_denom")).alias("ratio"))
        return lf.collect()

    def execute_dp(
        self,
        info_dataset: InfoDataset,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> pl.DataFrame:

        results = (
            self.plan_dp(info_dataset, key_values)
            .release()
            .collect()
            .with_columns(
                (pl.col("sum_numerateur") / pl.col("sum_denominateur")).alias("ratio")
            )
        )
        return results

    def transformation(self) -> (Count, Sum, Sum):
        """
        Retourne une transformation composée d’un Comptage et de deux classes Total.
        """
        comptage = Count(by=self.by, filtre=self.filtre)
        total_num = Sum(
            by=self.by, filtre=self.filtre, variable=self.variable_numerateur,
            bounds=self.bounds_numerateur
        )
        total_denom = Sum(
            by=self.by, filtre=self.filtre, variable=self.variable_denominateur,
            bounds=self.bounds_denominateur
        )
        return (comptage, total_num, total_denom)


class Quantile(Query):
    def __init__(
        self,
        variable: str,
        bounds: tuple[float, float],
        alpha: list,
        nb_candidats: int,
        by: Optional[list[str]] = None,
        filtre: Optional[str] = None
    ):
        super().__init__(by=by, filtre=filtre)
        self.variable = variable
        self.bounds = bounds
        if isinstance(alpha, list):
            self.alpha = alpha
        else:
            self.alpha = [alpha]
        self.nb_candidats = nb_candidats

    def plan_dp(
        self,
        info_dataset: InfoDataset,
        key_values: Optional[dict[str, list[str]]] = None
    ) -> LazyFrameQuery:

        bounds_min, bounds_max = self.bounds
        candidats = np.linspace(bounds_min, bounds_max, int(self.nb_candidats))
        context = info_dataset.context_eps
        query = context.query()
        exprs = [
            pl.col(self.variable)
            .fill_null(0)
            .dp.quantile(float(a), candidats)
            .alias(f"quantile_{float(a)}")
            for a in self.alpha
        ]
        query = self.filtre_bounds_by(query, *exprs, use_bounds=False, key_values=key_values)
        return query

    def execute(
        self,
        lf: pl.LazyFrame,
        use_bounds: bool
    ) -> pl.DataFrame:

        expr = (
            pl.col(self.variable)
            .quantile(float(alpha), interpolation="nearest")
            .alias(f"quantile_{float(alpha)}")
            for alpha in self.alpha
        )
        lf = self.filtre_bounds_by(lf, expr, use_bounds=use_bounds)
        return lf.collect()
