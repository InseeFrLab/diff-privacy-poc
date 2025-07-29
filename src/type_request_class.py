import polars as pl
from src.fonctions import (
    parse_filter_string
)
from itertools import product
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any


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


def generate_public_keys(by_keys: list[str], key_values) -> pl.LazyFrame:
    # Ne garder que les colonnes utiles pour le group_by
    values = [key_values[key] for key in by_keys if key in key_values]
    combinaisons = list(product(*values))  # Produit cartésien des valeurs
    public_keys = pl.DataFrame([dict(zip(by_keys, comb)) for comb in combinaisons]).lazy()
    return public_keys


class Requete(ABC):
    def __init__(self, by: Optional[list[str]] = None, filtre: Optional[str] = None):
        self.by = by
        self.filtre = filtre
        self.id_req = []
        self.poids = 0
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
    def execute(self, df: pl.DataFrame, use_bounds: bool) -> pl.DataFrame:
        pass

    @abstractmethod
    def plan_dp(self, context, key_values):
        pass

    def precision_opendp(self, context, key_values, alpha=0.05):
        return self.plan_dp(context, key_values).summarize(alpha=alpha)

    def execute_dp(self, context, key_values):
        return self.plan_dp(context, key_values).release().collect()

    def to_query_dict(self) -> dict[str, Any]:
        exclure = {"groupement", "groupement_style", "id_req", "poids", "sigma2", "scale"}
        return {
            "type": self.__class__.__name__,
            **{k: v for k, v in self.__dict__.items() if v is not None and k not in exclure}
        }

    def __repr__(self):
        cls_name = self.__class__.__name__
        exclure = {"groupement", "groupement_style", "id_req", "poids", "sigma2", "scale"}
        args = [
            f"{key}={value!r}"
            for key, value in self.__dict__.items()
            if value is not None and key not in exclure
        ]
        return f"{cls_name}({', '.join(args)})"

    def __eq__(self, other):
        exclure = {"id_req", "poids", "sigma2", "scale"}
        if not isinstance(other, self.__class__):
            return False

        def normalize(obj):
            if isinstance(obj, list):
                return sorted(obj)
            return obj

        for key in set(self.__dict__) | set(other.__dict__):
            if normalize(self.__dict__.get(key)) != normalize(other.__dict__.get(key)) and key not in exclure:
                return False
        return True

    def filtre_bounds_by(self, df, *expr, use_bounds, key_values=None):
        if self.filtre:
            df = df.filter(parse_filter_string(self.filtre))

        if use_bounds:
            if self.__class__.__name__ == "Ratio":
                df = apply_bounds(df, self.variable_numerateur, self.bounds_numerateur)
                df = apply_bounds(df, self.variable_denominateur, self.bounds_denominateur)
            else:
                df = apply_bounds(df, self.variable, self.bounds)

        if self.by:
            df = df.group_by(self.by).agg(*expr)
            if key_values:
                df = df.join(
                    generate_public_keys(by_keys=self.by, key_values=key_values),
                    on=self.by, how="right"
                )
        else:
            df = df.select(*expr)

        return df


class Comptage(Requete):

    def plan_dp(self, context, key_values):
        query = context.query().with_columns(pl.lit(1).alias("colonne_comptage"))
        expr = (
            pl.col("colonne_comptage")
            .fill_null(0)
            .dp.sum((0, 1))
            .alias("count")
        )
        query = self.filtre_bounds_by(query, expr, use_bounds=False, key_values=key_values)
        return query

    def execute(self, df, use_bounds):
        expr = (
            pl.count().alias("count")
        )
        df = self.filtre_bounds_by(df, expr, use_bounds=False)
        return df.collect()

    def precision_dp(self, budget_global):
        self.sigma2 = 1/(2 * budget_global * self.poids)
        return self.sigma2


class Total(Requete):
    def __init__(self, variable, bounds, by=None, filtre=None):
        super().__init__(by=by, filtre=filtre)
        self.variable = variable
        self.bounds = bounds

    def plan_dp(self, context, key_values, centre: bool = True):
        l, u = self.bounds
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

        query = self.filtre_bounds_by(query, expr, use_bounds=False, key_values=key_values)
        return query

    def execute_dp(self, context, key_values, centre: bool = True):
        return self.plan_dp(context, key_values, centre=centre).release().collect()

    def execute(self, df, use_bounds):
        expr = (
            pl.col(self.variable).sum().alias("sum"),
            pl.count().alias("count")
        )
        df = self.filtre_bounds_by(df, expr, use_bounds=use_bounds)
        return df.collect()

    def transformation(self):
        """
        Retourne une transformation composée d’un Comptage et du Total lui-même.
        """
        comptage = Comptage(by=self.by, filtre=self.filtre)
        return (comptage, self)

    def precision_dp(self, budget_global):
        l, u = self.bounds
        self.sigma2 = (u - l)**2/(4 * 2 * budget_global * self.poids)
        return self.sigma2


class Moyenne(Requete):
    def __init__(self, variable, bounds, by=None, filtre=None):
        super().__init__(by=by, filtre=filtre)
        self.variable = variable
        self.bounds = bounds

    def plan_dp(self, context, key_values):
        l, u = self.bounds
        center = (u + l) / 2
        half_range = (u - l) / 2
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
        query = self.filtre_bounds_by(query, expr, use_bounds=False, key_values=key_values)
        # Calcul de la moyenne centrée bruitée, puis recentrage
        query = query.with_columns(
            ((pl.col("centered_sum") / pl.col("count")) + center).alias("mean")
        )
        return query

    def execute(self, df, use_bounds):
        expr = (
            pl.col(self.variable).sum().alias("sum"),
            pl.count().alias("count"),
            pl.col(self.variable).mean().alias("mean")
        )
        df = self.filtre_bounds_by(df, expr, use_bounds=use_bounds)
        return df.collect()

    def transformation(self):
        """
        Retourne une transformation composée d’un Comptage et d'un Total.
        """
        comptage = Comptage(by=self.by, filtre=self.filtre)
        total = Total(by=self.by, filtre=self.filtre, variable=self.variable, bounds=self.bounds)
        return (comptage, total)


class Ratio(Requete):
    def __init__(
        self, variable, variable_denominateur, bounds,
        bounds_denominateur, by=None, filtre=None
    ):
        super().__init__(by=by, filtre=filtre)
        self.variable_numerateur = variable
        self.bounds_numerateur = bounds
        self.variable_denominateur = variable_denominateur
        self.bounds_denominateur = bounds_denominateur

    def plan_dp(self, context, key_values):
        l_num, u_num = self.bounds_numerateur
        l_denom, u_denom = self.bounds_denominateur
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
        query = self.filtre_bounds_by(query, expr, use_bounds=False, key_values=key_values)
        # Calcul de la moyenne centrée bruitée, puis recentrage
        query = query.with_columns(
            (pl.col("sum_numerateur") / pl.col("sum_denominateur")).alias("ratio")
        )
        return query

    def execute(self, df, use_bounds):
        expr = (
            pl.col(self.variable_numerateur).sum().alias("sum_num"),
            pl.col(self.variable_denominateur).sum().alias("sum_denom")
        )
        df = self.filtre_bounds_by(df, expr, use_bounds=use_bounds)
        df = df.with_columns((pl.col("sum_num") / pl.col("sum_denom")).alias("ratio"))
        return df.collect()

    def transformation(self):
        """
        Retourne une transformation composée d’un Comptage et de deux classes Total.
        """
        comptage = Comptage(by=self.by, filtre=self.filtre)
        total_num = Total(by=self.by, filtre=self.filtre, variable=self.variable_numerateur, bounds=self.bounds_numerateur)
        total_denom = Total(by=self.by, filtre=self.filtre, variable=self.variable_denominateur, bounds=self.bounds_denominateur)
        return (comptage, total_num, total_denom)


class Quantile(Requete):
    def __init__(self, variable, bounds, alpha, nb_candidats, by=None, filtre=None):
        super().__init__(by=by, filtre=filtre)
        self.variable = variable
        self.bounds = bounds
        if isinstance(alpha, list):
            self.alpha = alpha
        else:
            self.alpha = [alpha]
        self.nb_candidats = nb_candidats

    def plan_dp(self, context, key_values):
        bounds_min, bounds_max = self.bounds
        candidats = np.linspace(bounds_min, bounds_max, int(self.nb_candidats))
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

    def execute(self, df, use_bounds):
        expr = (
            pl.col(self.variable)
            .quantile(float(alpha), interpolation="nearest")
            .alias(f"quantile_{float(alpha)}")
            for alpha in self.alpha
        )
        df = self.filtre_bounds_by(df, expr, use_bounds=use_bounds)
        return df.collect()
