import polars as pl
from src.fonctions import (
    parse_filter_string
)
from itertools import product
from typing import Any
from abc import ABC, abstractmethod
import numpy as np


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

    # Produit cartésien des valeurs
    combinaisons = list(product(*values))

    # Construction du LazyFrame public
    public_keys = pl.DataFrame([dict(zip(by_keys, comb)) for comb in combinaisons]).lazy()
    return public_keys


class Requete(ABC):
    def __init__(self, by=None, filtre=None):
        self.by = by
        self.filtre = filtre

    @abstractmethod
    def plan_dp(self, context, key_values):
        pass

    def precision_dp(self, context, key_values, alpha=0.05):
        return self.plan_dp(context, key_values).summarize(alpha=alpha)

    def execute_dp(self, context, key_values):
        return self.plan_dp(context, key_values).release().collect()

    @abstractmethod
    def execute(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    def generate_public_keys(self, by_keys, key_values: dict):
        """Méthode utilitaire à partager : crée les clés publiques"""
        df_keys = {
            key: key_values.get(key, [])
            for key in by_keys
            if key in key_values
        }
        return pl.DataFrame(df_keys)

    def filtre_bounds_by(self, df, expr, use_bounds, key_values=None):
        if self.filtre:
            df = df.filter(parse_filter_string(self.filtre))

        if use_bounds:
            df = apply_bounds(df, self.variable, self.bounds)

        if self.by:
            df = df.group_by(self.by).agg(expr)
            if key_values:
                df = df.join(generate_public_keys(by_keys=self.by, key_values=key_values), on=self.by, how="right")
        else:
            df = df.select(expr)

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

        query = self.filtre_bounds_by(self, query, expr, use_bounds=False, key_values=key_values)

        return query

    def execute(self, df):
        expr = (
            pl.count().alias("count")
        )

        df = self.filtre_bounds_by(df, expr, use_bounds=False)

        return df.collect()

    def to_query_dict(self) -> dict[str, Any]:
        query = {
            "type": "Comptage"
        }
        if self.by is not None:
            query["by"] = self.by
        if self.filtre is not None:
            query["filtre"] = self.filtre
        return query


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

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(expr)
                .join(self.generate_public_keys(by_keys=self.by, key_values=key_values), on=self.by, how="right")
            )
        else:
            query = query.select(expr)
        return query

    def execute_dp(self, context, key_values, centre: bool = True):
        return self.plan_dp(context, key_values, centre=centre).release().collect()

    def execute(self, df, use_bounds):
        expr = (
            pl.col(self.variable).sum().alias("sum"),
            pl.count().alias("count")
        )

        df = self.filtre_bounds_and_by(self, df, expr, use_bounds)

        return df.collect()

    def to_query_dict(self) -> dict[str, Any]:
        query = {
            "type": "Total",
            "variable": self.variable,
            "bounds": self.bounds
        }
        if self.by is not None:
            query["by"] = self.by
        if self.filtre is not None:
            query["filtre"] = self.filtre
        return query


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

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(expr)
                .join(self.generate_public_keys(by_keys=self.by, key_values=key_values), on=self.by, how="right")
            )
        else:
            query = query.select(expr)

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

        df = self.filtre_bounds_and_by(df, expr, use_bounds)

        return df.collect()


class Ratio(Requete):
    def __init__(self, variable_numerateur, variable_denominateur, bounds, by=None, filtre=None):
        super().__init__(by=by, filtre=filtre)
        self.variable_numerateur = variable_numerateur
        self.variable_denominateur = variable_denominateur
        self.bounds = bounds

    def plan_dp(self, context, key_values):
        pass

    def execute(self, df, use_bounds):
        expr = (
            pl.col(self.variable_numerateur).sum().alias("sum_num"),
            pl.col(self.variable_denominateur).sum().alias("sum_denom")
        )

        df = self.filtre_bounds_and_by(df, expr, use_bounds)

        return df.collect()


class Quantile(Requete):
    def __init__(self, variable, bounds, alpha, nb_candidats, by=None, filtre=None):
        super().__init__(by=by, filtre=filtre)
        self.variable = variable
        self.bounds = bounds
        bounds_min, bounds_max = bounds
        self.candidats = np.linspace(bounds_min, bounds_max, int(nb_candidats))

        if isinstance(alpha, list):
            self.list_alpha = alpha
        else:
            self.list_alpha = [alpha]

    def plan_dp(self, context, key_values):
        query = context.query()
        expr = [
            pl.col(self.variable)
            .fill_null(0)
            .dp.quantile(float(a), self.candidats)
            .alias(f"quantile_{float(a)}")
            for a in self.list_alpha
        ]

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(*expr)
                .join(self.generate_public_keys(by_keys=self.by, key_values=key_values), on=self.by, how="right")
            )
        else:
            query = query.select(*expr)

        return query

    def execute(self, df, use_bounds):
        expr = (
            pl.col(self.variable)
            .quantile(float(alpha), interpolation="nearest")
            .alias(f"quantile_{float(alpha)}")
            for alpha in self.list_alpha
        )

        df = self.filtre_bounds_and_by(self, df, expr, use_bounds)

        return df.collect()
