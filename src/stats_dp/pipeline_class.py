import polars as pl
import opendp.prelude as dp
from typing import Any
import copy
from stats_dp.process_tools import (
    calculer_toutes_les_requetes, optimisation_et_assemblage_results,
    df_comptage, df_total, df_moyenne, df_ratio, df_quantile
)
from stats_dp.fonctions import (
    optimisation_chaine,
    create_context, intervalle_confiance_quantile
)
from stats_dp.request_class import Count, Sum, Quantile, Query
import numpy as np
import time
dp.enable_features("contrib")


class Pipeline():
    def __init__(
        self,
        dict_req: dict[str, Query],
        lf: pl.LazyFrame,
        contribution_individu_max: int = 1,
        borne_max_taille_dataset: int = 1
    ):
        self.dict_req = dict_req
        self.lf = lf
        self.contribution_individu_max = contribution_individu_max
        self.borne_max_taille_dataset = borne_max_taille_dataset

        variables = {val for request in self.dict_req.values() if request.by for val in request.by}
        df = self.lf.select([pl.col(v).drop_nulls() for v in variables]).collect()
        self.key_values = {
            v: sorted(df[v].unique().to_list())
            for v in variables
        }

        data_requetes = {k: copy.deepcopy(v) for k, v in self.dict_req.items()}
        self.dict_query = {}
        i = 1

        for (key, request) in data_requetes.items():
            if not isinstance(request, (Count, Quantile)):
                tuple_request = request.transformation()
            else:
                tuple_request = (request,)

            for sous_req in tuple_request:
                if sous_req not in self.dict_query.values():
                    sous_req.id_req.append(key)
                    cle = f"query_{i}"
                    self.dict_query[cle] = sous_req
                    i += 1

                else:
                    # 🔎 Trouver la clé correspondant à la requête identique
                    id_cle = next(k for k, v in self.dict_query.items() if v == sous_req)
                    self.dict_query[id_cle].id_req.append(key)

    def execute(
        self,
        use_bounds: bool = False
    ) -> dict[str, pl.DataFrame]:
        print("==> Début execute")
        start_global = time.time()
        dict_resultat = {}

        for key, request in self.dict_req.items():

            resultat = request.execute(lf=self.lf, use_bounds=use_bounds)
            dict_resultat[key] = resultat

        total_duration = time.time() - start_global
        print(f"<== Fin execute (temps total : {total_duration:.2f} secondes)")
        return dict_resultat

    def execute_dp(
        self,
        budget_global: float,
        dict_poids: dict[str, float],
        optim_MCG: bool = True,
        progress: bool = False
    ) -> dict[str, pl.DataFrame]:
        print("==> Début execute_dp")
        start_global = time.time()

        data_query = self._query_pondere(budget_global=budget_global, dict_poids=dict_poids)
        data_lazy = self.lf

        # Extraire toutes les colonnes mentionnées dans les requêtes
        vars_by = {val for request in data_query.values() if request.by for val in request.by}
        vars_variable = {
            v for v in (getattr(req, "variable", None) for req in data_query.values())
            if v is not None
        }
        vars_variable_num = {
            v for v in (getattr(req, "variable_numerateur", None) for req in data_query.values())
            if v is not None
        }
        vars_variable_denom = {
            v for v in (getattr(req, "variable_denominateur", None) for req in data_query.values())
            if v is not None
        }
        selected_columns = set(vars_by | vars_variable | vars_variable_num | vars_variable_denom)

        # Sous-échantillon propre du LazyFrame
        if not selected_columns:
            filtered_lazy = (
                data_lazy.with_columns(pl.lit(1).alias("__dummy"))
                .select("__dummy")
                .collect()
                .lazy()
            )

        else:
            filtered_lazy = data_lazy.select(selected_columns).collect().lazy()

        context_param = {
            "data": filtered_lazy,
            "privacy_unit": dp.unit_of(contributions=self.contribution_individu_max),
            "margins": [dp.polars.Margin(max_partition_length=self.borne_max_taille_dataset)],
        }

        start_context = time.time()

        context_rho, context_eps = create_context(
            context_param, budget_global, data_query
        )
        duration = time.time() - start_context
        print(f"✅ Context terminée en {duration:.2f} secondes.")

        resultats_df = calculer_toutes_les_requetes(
            context_rho, context_eps, self.key_values, data_query, progress
        )

        resultats_df = optimisation_et_assemblage_results(resultats_df, self.dict_req, data_query, self.key_values)
        total_duration = time.time() - start_global
        print(f"<== Fin execute_dp (temps total : {total_duration:.2f} secondes)")
        return resultats_df

    def precision_dp(
        self,
        budget_global: float,
        dict_poids: dict[str, float]
    ):
        print("==> Début precision_dp")
        start_global = time.time()

        data_query = self._query_pondere(budget_global=budget_global, dict_poids=dict_poids)

        query_comptage = {k: v for k, v in data_query.items() if isinstance(v, Count)}
        query_comptage = optimisation_chaine(query_comptage, self.key_values, budget_global)

        query_total = {k: v for k, v in data_query.items() if isinstance(v, Sum)}
        query_total = optimisation_chaine(query_total, self.key_values, budget_global)

        query_quantile = {k: v for k, v in data_query.items() if isinstance(v, Quantile)}
        filtres_uniques = set(query.filtre for query in query_quantile.values())
        variables_uniques = set(query.variable for query in query_quantile.values())

        for filtre in filtres_uniques:
            for variable in variables_uniques:
                query_filtre_variable = {
                    k: v for k, v in query_quantile.items()
                    if v.variable == variable and v.filtre == filtre
                }

                for key_query, query in query_filtre_variable.items():

                    epsilon = np.sqrt(8 * budget_global * query.poids)

                    vrai_tableau = query.execute(self.lf, use_bounds=False)
                    ic = intervalle_confiance_quantile(self.lf, query, epsilon, vrai_tableau)
                    query_quantile[key_query].scale = ic

        results = {
            "Comptage": df_comptage(self.dict_req, query_comptage),
            "Total": df_total(self.lf, self.dict_req, query_comptage, query_total),
            "Moyenne": df_moyenne(self.lf, self.dict_req, query_comptage, query_total),
            "Ratio": df_ratio(self.lf, self.dict_req, query_comptage, query_total),
            "Quantile": df_quantile(query_quantile),
        }
        total_duration = time.time() - start_global
        print(f"<== Fin precision_dp (temps total : {total_duration:.2f} secondes)")
        return results

    def precision_opendp(
        self,
        budget_global: float,
        dict_poids: dict[str, float],
        alpha: float = 0.05
    ):
        dict_resultat = {}

        context_param = {
            "data": self.lf,
            "privacy_unit": dp.unit_of(contributions=self.contribution_individu_max),
            "margins": [dp.polars.Margin(max_partition_length=self.borne_max_taille_dataset)],
        }

        data_requetes = {k: copy.deepcopy(v) for k, v in self.dict_req.items()}
        poids_total = sum(poids for poids in dict_poids.values())

        for key, request in data_requetes.items():
            request.poids = dict_poids[key] / poids_total

        context_rho, context_eps = create_context(
            context_param, budget_global, data_requetes
        )

        for key, request in data_requetes.items():

            if isinstance(request, Quantile):
                context_use = context_eps
            else:
                context_use = context_rho

            resultat = request.precision_opendp(
                context=context_use, key_values=self.key_values, alpha=alpha
            )
            dict_resultat[key] = resultat

        return dict_resultat

    def _query_pondere(
        self,
        budget_global: float,
        dict_poids: dict[str, float]
    ) -> dict[str, dict[str, Any]]:
        for request in self.dict_query.values():
            request.poids = sum(dict_poids.get(r, 0) for r in request.id_req)

            if isinstance(request, (Count, Sum)):
                request.precision_dp(budget_global)
        return self.dict_query
