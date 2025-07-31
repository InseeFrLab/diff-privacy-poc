import polars as pl
import opendp.prelude as dp
from typing import Any
import copy
from src.process_tools import (
    calculer_toutes_les_requetes, optimisation_et_assemblage_results,
    df_comptage, df_total, df_moyenne, df_ratio, df_quantile
)
from src.fonctions import (
    optimisation_chaine,
    create_context, intervalle_confiance_quantile
)
import numpy as np
dp.enable_features("contrib")


class Pipeline():
    def __init__(
        self, dict_req, lf: pl.DataFrame,
        contribution_individu_max: int = 1, borne_max_taille_dataset: int = 1
    ):
        self.dict_req = dict_req
        self.lf = lf
        self.contribution_individu_max = contribution_individu_max
        self.borne_max_taille_dataset = borne_max_taille_dataset

    def execute(self, use_bounds=False, afficher=True):
        dict_resultat = {}
        for key, request in self.dict_req.items():
            resultat = request.execute(df=self.lf, use_bounds=use_bounds).to_pandas()
            dict_resultat[key] = resultat
            if afficher:
                print(resultat)
        return dict_resultat

    def execute_dp(self, budget_global, dict_poids, optim_MCG: bool = True):
        data_query = self._dict_query(budget_global=budget_global, dict_poids=dict_poids)
        data_lazy = self.lf
        keys = self._key_values()

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

        context_rho, context_eps = create_context(
            context_param, budget_global, data_query
        )

        resultats_df = {}

        calculer_toutes_les_requetes(
            context_rho, context_eps, keys, data_query, resultats_df
        )

        optimisation_et_assemblage_results(resultats_df, self.dict_req, data_query, keys)

        return resultats_df

    def precision_dp(self, budget_global, dict_poids):
        data_query = self._dict_query(budget_global=budget_global, dict_poids=dict_poids)

        query_comptage = {k: v for k, v in data_query.items() if v.__class__.__name__ == "Comptage"}
        query_comptage = optimisation_chaine(query_comptage, self._key_values(), budget_global)

        query_total = {k: v for k, v in data_query.items() if v.__class__.__name__ == "Total"}
        query_total = optimisation_chaine(query_total, self._key_values(), budget_global)

        query_quantile = {k: v for k, v in data_query.items() if v.__class__.__name__ == "Quantile"}
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

        for key, df_results in results.items():
            if not df_results.empty:
                if key != "Comptage":
                    df_results = df_results.drop(columns="label")
                # Définir l'arrondi spécifique
                cols = [
                    "cv moyen (%)", "biais relatif moyen (%)",
                    "écart type comptage", "écart type bruit comptage"
                ]

                arrondi = {col: 1 if col in cols else 0 for col in df_results.columns}
                results[key] = df_results.round(arrondi)

        return results

    def precision_opendp(self, budget_global, dict_poids, alpha=0.05, afficher=True):
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

            type_req = request.__class__.__name__

            if type_req == "Quantile":
                context_use = context_eps
            else:
                context_use = context_rho

            resultat = request.precision_opendp(
                context=context_use, key_values=self._key_values(), alpha=alpha
            )
            dict_resultat[key] = resultat

            if afficher:
                print(resultat)
        return dict_resultat

    def _key_values(self) -> dict[str, list[Any]]:
        variables = {val for request in self.dict_req.values() if request.by for val in request.by}

        df = self.lf.select([pl.col(v).drop_nulls() for v in variables]).collect()

        result = {
            v: sorted(df[v].unique().to_list())
            for v in variables
        }
        return result

    def _dict_query(self, budget_global, dict_poids) -> dict[str, dict[str, Any]]:
        print("==> Début dict_query")

        data_requetes = {k: copy.deepcopy(v) for k, v in self.dict_req.items()}
        query = {}
        i = 1

        for (key, request) in data_requetes.items():
            if request.__class__.__name__ not in ["Comptage", "Quantile"]:
                tuple_request = request.transformation()
            else:
                tuple_request = (request,)

            for sous_req in tuple_request:
                if sous_req not in query.values():
                    sous_req.id_req.append(key)
                    cle = f"query_{i}"
                    query[cle] = sous_req
                    i += 1

                else:
                    # 🔎 Trouver la clé correspondant à la requête identique
                    id_cle = next(k for k, v in query.items() if v == sous_req)
                    query[id_cle].id_req.append(key)

        for request in query.values():
            request.poids = sum(dict_poids.get(r, 0) for r in request.id_req)

            if request.__class__.__name__ in ["Comptage", "Total"]:
                request.precision_dp(budget_global)

        print("<== Fin dict_query")
        return query
