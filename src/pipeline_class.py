import polars as pl
import opendp.prelude as dp
from typing import Any
from src.fonctions import create_context
import copy

dp.enable_features("contrib")


class Pipeline():
    def __init__(
        self, dict_req, lf: pl.DataFrame, participation_individu_max=1, borne_max_taille_dataset=1
    ):
        self.dict_req = dict_req
        self.lf = lf
        self.participation_individu_max = participation_individu_max
        self.borne_max_taille_dataset = borne_max_taille_dataset

    def execute(self, use_bounds=False, afficher=True):
        dict_resultat = {}
        for key, request in self.dict_req.items():
            resultat = request.execute(df=self.lf, use_bounds=use_bounds).to_pandas()
            dict_resultat[key] = resultat
            if afficher:
                print(resultat)
        return dict_resultat

    def execute_dp(self, budget_global, dict_poids):
        pass

    def precision(self, budget_global, dict_poids):
        pass

    def precision_opendp(self, budget_global, dict_poids, alpha=0.05, afficher=True):
        dict_resultat = {}

        context_param = {
            "data": self.lf,
            "privacy_unit": dp.unit_of(contributions=self.participation_individu_max),
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
