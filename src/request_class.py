from src.fonctions import (
    parse_filter_string
)

from abc import ABC, abstractmethod
import polars as pl
import numpy as np


# Classe mère
class request_dp(ABC):
    def __init__(self, context, key_values, by=None, variable=None, bounds=None, filtre=None):
        self.context = context
        self.key_values = key_values
        self.by = by
        self.variable = variable
        self.bounds = bounds
        self.filtre = filtre

    def generate_public_keys(self, by_keys: list[str]) -> pl.LazyFrame:
        from itertools import product

        # Ne garder que les colonnes utiles pour le group_by
        values = [self.key_values[key] for key in by_keys if key in self.key_values]

        # Produit cartésien des valeurs
        combinaisons = list(product(*values))

        # Construction du LazyFrame public
        public_keys = pl.DataFrame([dict(zip(by_keys, comb)) for comb in combinaisons]).lazy()
        return public_keys

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def precision(self):
        pass


class count_dp(request_dp):
    """
    Comptage différentiellement privé - Notes de la fonction d'OpenDP

    - Sensibilité = privacy_unit

    - Ne pas exécuter la requête si la marge est supposée connue (argument lengths dans un margin)
        → sinon : ValueError: unable to infer bounds
        → la contourner via une requête DP jointe à celle-ci (pas possible via ma fonction)

    - La requête peut se faire en (epsilon)-DP (Laplace) ou (rho)-zCDP (Gaussien)

    - Si group_by :
        - L'argument keys doit être inclus dans un margin sinon (epsilon, delta)-DP avec seuil
        → Autre solution : jointure avec un ensemble de clés (complet ou partiel)
            → Si budget en (epsilon, delta), aucun seuil n'est appliqué (pas delta dépensé)
        - epsilon affecte l'écart-type du bruit + le seuil
        - delta affecte uniquement le seuil
        - Formule classique pour l'écart-type, mais le seuil reste mal compris

    - Si filter :
        - Le margin n'est pas prise en compte
        → Problème potentiel avec group_by
        → Solution : jointure avec un ensemble de clés (complet ou partiel)
            → Si budget en (epsilon, delta), aucun seuil n'est appliqué (pas delta dépensé)
    """
    def execute(self):
        query = self.context.query().with_columns(pl.lit(1).alias("colonne_comptage"))

        expr = (
            pl.col("colonne_comptage")
            .fill_null(0)
            .dp.sum((0, 1))
            .alias("count")
        )

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(expr)
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(expr)

        return query

    def precision(self, alpha=0.05):
        return self.execute().summarize(alpha=alpha)


class sum_dp(request_dp):
    """
    Somme différentiellement privé - Notes de la fonction d'OpenDP

    - La sensibilité dépend de la `privacy_unit` :
        - Cas général : sensibilité = privacy_unit * max(abs(L), abs(U))
        - Si une marge sur la variable est connue : sensibilité = (privacy_unit // 2) * (U - L)

    - Nécéssaire :
        - Définir dans la marge `dp.polars.Margin(max_partition_length=?)`
        - Définir un `.fill_null(?)` dans la chaîne
        - Définir un `.fill_nan(?)` dans la chaîne si et seulement si entrée de type float

    - La requête peut se faire en (epsilon)-DP (Laplace) ou (rho)-zCDP (Gaussien)

    - Si group_by :
        - L'argument keys doit être inclus dans un margin sinon requête jointe len (epsilon, delta)-DP avec seuil
        - Si entrée type float, nécessité de définir `max_num_partitions` dans le margin (must be known when the metric is not sensitive to ordering (SymmetricDistance))
        → Autre solution : jointure avec un ensemble de clés (complet ou partiel)
            → Si budget en (epsilon, delta), aucun seuil n'est appliqué (pas delta dépensé) et pas de max_num_partitions

    - Si filter :
        - Le margin n'est pas prise en compte
        → Problème potentiel avec group_by et formule maximum pour sensibilité
        → Solution (règle pas sensibilité) : jointure avec un ensemble de clés (complet ou partiel)
            → Si budget en (epsilon, delta), aucun seuil n'est appliqué (pas delta dépensé)

    - Remarques privacy unit :
        - Peut être fixée à 1 si :
            - La marge (lengths) est connue sur l'ensemble du dataset
            - Group_by nécessaire avec clés connues (mais pas lengths)
            - Entrée de type int (si float, scale suspicieusement faible)
        - Sinon, une erreur est levée.
    """
    def execute(self):
        l, u = self.bounds
        query = self.context.query()

        query = self.context.query()
        expr = pl.col(self.variable).fill_null(0).fill_nan(0).dp.sum((l, u)).alias("sum")

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(expr)
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(expr)
        return query

    def precision(self, alpha=0.05):
        return self.execute().summarize(alpha=alpha)


class sum_centered_dp(request_dp):
    def execute(self):
        l, u = self.bounds
        m = (l + u)/2
        sensi = (u - l)/2
        query = self.context.query()
        expr = (pl.col(self.variable) - m).fill_null(0).fill_nan(0).dp.sum((-sensi, sensi)).alias("sum")

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(expr)
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(expr)
        return query

    def precision(self, alpha=0.05):
        return self.execute().summarize(alpha=alpha)


class mean_dp(request_dp):
    """
    Moyenne différentiellement privé - Notes de la fonction d'OpenDP

    - Il s'agit de l'utilisation des deux dernières fonctions avec un budget divisé équitablement

    - La sensibilité : voir cas comptage et somme

    - Nécéssaire :
        - Définir dans la marge `dp.polars.Margin(max_partition_length=?)`
        - Définir un `.fill_null(?)` dans la chaîne
        - Définir un `.fill_nan(?)` dans la chaîne si et seulement si entrée de type float

    - La requête peut se faire en (epsilon)-DP (Laplace) ou (rho)-zCDP (Gaussien)

    - Si group_by :
        - L'argument keys doit être inclus dans un margin
        - Si entrée type float, nécessité de définir `max_num_partitions` dans le margin (must be known when the metric is not sensitive to ordering (SymmetricDistance))
        → Autre solution : jointure avec un ensemble de clés (complet ou partiel)

    - Si filter :
        - Le margin n'est pas prise en compte
        → Problème potentiel avec group_by et formule maximum pour sensibilité
        → Solution (règle pas sensibilité) : jointure avec un ensemble de clés (complet ou partiel)
            → Si budget en (epsilon, delta), aucun seuil n'est appliqué (pas delta dépensé)

    - Remarques privacy unit :
        - Peut être fixée à 1 si :
            - La marge (lengths) est connue sur l'ensemble du dataset
            - Group_by nécessaire avec clés connues (mais pas lengths)
            - Entrée de type int (si float, scale suspicieusement faible)
        - Sinon, une erreur est levée.
    """
    def execute(self):
        l, u = self.bounds
        center = (u + l) / 2
        half_range = (u - l) / 2
        query = self.context.query()
        dtype = query.collect_schema().get(self.variable)
        query = self.context.query()
        is_float = dtype in (pl.Float32, pl.Float64)

        # Construction conditionnelle des expressions
        col_var = pl.col(self.variable)
        centered_col = (col_var - center).fill_null(0)
        len_expr = col_var.fill_null(1)

        if is_float:
            centered_col = centered_col.fill_nan(0)
            len_expr = len_expr.fill_nan(1)

        expr = (
            centered_col.dp.sum(bounds=(-half_range, half_range)).alias("centered_sum"),
            len_expr.dp.sum(bounds=(1, 1)).alias("count")
        )

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(expr)
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(expr)

        # Calcul de la moyenne centrée bruitée, puis clipping, puis recentrage
        query = query.with_columns(
            (
                (pl.col("centered_sum") / pl.col("count"))
                .clip(-half_range, half_range) + center
            ).alias("mean")
        )

        return query

    def precision(self, alpha=0.05):
        return self.execute().summarize(alpha=alpha)


class quantile_dp(request_dp):
    """
    Quantile différentiellement privé - Notes de la fonction d'OpenDP

    - Sensibilité proportionnelle à la privacy_unit mais pas de formule claire

    - Nécéssaire :
        - Définir dans la marge `dp.polars.Margin(max_partition_length=?)`
        - Définir un `.fill_null(?)` dans la chaîne
        - Définir un `.fill_nan(?)` dans la chaîne si et seulement si entrée de type float
        - Définir une liste de candidats, par ordre croissant, et de valeurs entières distinctes

    - La requête se fait en (epsilon)-DP (Laplace)

    - Si marge du dataset connu, impossible privacy unit = 1 et la sensibilité proportionnelle à (privacy_unit // 2) * 4 * sensibilité de base (conjecture)

    - Si group_by :
        - L'argument keys doit être inclus dans un margin sinon requête jointe len (epsilon, delta)-DP avec seuil
        - Si entrée type float, nécessité de définir `max_num_partitions` dans le margin (must be known when the metric is not sensitive to ordering (SymmetricDistance))
        → Autre solution : jointure avec un ensemble de clés (complet ou partiel)
            → Si budget en (epsilon, delta), aucun seuil n'est appliqué (pas delta dépensé) et pas de max_num_partitions
        - Plus on augmente privacy unit, plus la sensibilité augmente rapidement (2, 8, 18, 32, 40, 48, 56) indépendamment des marges

    - Si filter :
        - Le margin n'est pas prise en compte
        → Problème potentiel avec group_by et formule maximum pour sensibilité
        → Solution (règle pas sensibilité) : jointure avec un ensemble de clés (complet ou partiel)
            → Si budget en (epsilon, delta), aucun seuil n'est appliqué (pas delta dépensé)

    - Remarques ordres des quantiles :
        - Si ordre des quantiles différents de 0, 0.25, 0.5, 0.75 ou 1:
            - Résultat extrêmement bruité
        - Sinon, dans le meilleur des cas, scale des quartiles (0, 0.25, 0.5, 0.75, 1) en (2, 6, 2, 6, 2)
    """
    def __init__(self, context, key_values, variable, nb_candidats, bounds, alpha, by=None, filtre=None):
        super().__init__(context, key_values, by, variable, bounds, filtre)

        # Attention, il y a y un problème avec OpenDP au niveau des candidats
        # Il faut qu'ils soient ordonnées de manière croissante ET de partie entière distincte
        bounds_min, bounds_max = self.bounds

        self.candidats = np.linspace(bounds_min, bounds_max, int(nb_candidats))
        # self.candidats = np.arange(candidats["min"], candidats["max"] + candidats["step"], candidats["step"]).tolist()

        if isinstance(alpha, list):
            self.list_alpha = alpha
        else:
            self.list_alpha = [alpha]

    def execute(self):
        query = self.context.query()
        aggs = [
            pl.col(self.variable)
            .fill_null(0)
            .dp.quantile(a, self.candidats)
            .alias(f"quantile_{a}")
            for a in self.list_alpha
        ]

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(*aggs)
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(*aggs)

        return query

    def precision(self, alpha=0.05):
        return self.execute().summarize(alpha=alpha)
