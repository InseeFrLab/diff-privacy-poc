import re
import numpy as np
import polars as pl
import pandas as pd
from scipy.optimize import fsolve
import opendp.prelude as dp
from src.constant import (
    radio_to_weight
)
import yaml
import operator
from itertools import combinations, product
import itertools
from functools import reduce
import cvxpy as cp

# Map des opérateurs Python vers leurs fonctions correspondantes
OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


def intervalle_confiance_quantile(dataset, req, epsilon):
    variable = req.get("variable")
    bounds_min, bounds_max = req.get("bounds")
    alpha = req.get("alpha")
    nb_candidats = int(req.get("nb_candidats"))
    by = req.get("by")

    candidats = np.linspace(bounds_min, bounds_max, nb_candidats + 1)
    precisions = []

    # Si pas de group_by, on applique directement
    if by is None:
        data_variable = dataset[variable].to_numpy()

        scores, sensi = manual_quantile_score(
            data_variable,
            candidats,
            alpha=alpha,
            et_si=True
        )

        proba_non_norm = np.exp(-epsilon * scores / (2 * sensi))
        proba = proba_non_norm / np.sum(proba_non_norm)

        sorted_indices = np.argsort(proba)[::-1]
        sorted_proba = np.array(proba)[sorted_indices]
        sorted_candidats = np.array(candidats)[sorted_indices]

        cumulative = np.cumsum(sorted_proba)
        top95_mask = cumulative <= 0.95
        if not np.all(top95_mask):
            top95_mask[np.argmax(cumulative > 0.95)] = True

        candidats_top95 = sorted_candidats[top95_mask]
        precision_val = np.max(candidats_top95) - np.min(candidats_top95)
        return precision_val

    # Si group_by, on boucle sur chaque modalité
    grouped = dataset.group_by(by)

    for _, group_df in grouped:
        data_variable = group_df[variable].to_numpy()

        if len(data_variable) == 0:
            continue  # éviter les groupes vides

        scores, sensi = manual_quantile_score(
            data_variable,
            candidats,
            alpha=alpha,
            et_si=True
        )

        proba_non_norm = np.exp(-epsilon * scores / (2 * sensi))
        proba = proba_non_norm / np.sum(proba_non_norm)

        sorted_indices = np.argsort(proba)[::-1]
        sorted_proba = np.array(proba)[sorted_indices]
        sorted_candidats = np.array(candidats)[sorted_indices]

        cumulative = np.cumsum(sorted_proba)
        top95_mask = cumulative <= 0.95
        if not np.all(top95_mask):
            top95_mask[np.argmax(cumulative > 0.95)] = True

        candidats_top95 = sorted_candidats[top95_mask]
        precision_val = np.max(candidats_top95) - np.min(candidats_top95)
        precisions.append(precision_val)

    return np.mean(precisions) if precisions else None


def generate_yaml_metadata_from_lazyframe_as_string(df: pl.DataFrame, dataset_name: str = "dataset"):
    metadata = {
        'dataset_name': dataset_name,
        'n_rows': df.height,
        'n_columns': df.width,
        'columns': {}
    }

    for col in df.columns:
        series = df[col]
        dtype = series.dtype

        col_meta = {
            'type': str(dtype),
            'missing': int(series.null_count())
        }

        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64):
            col_meta.update({
                'min': round(float(series.min()), 2),
                'max': round(float(series.max()), 2)
            })

        elif dtype == pl.Utf8 or dtype == pl.Categorical:
            col_meta['unique_values'] = int(series.n_unique())

        metadata['columns'][col] = col_meta

    yaml_str = yaml.dump(metadata, sort_keys=False, allow_unicode=True)
    return yaml_str


# Calcul de p
def produit_modalites(fset, nb_modalite):
    if not fset:
        return 1
    return reduce(operator.mul, (nb_modalite[v] for v in fset), 1)


def MCG(liste_request, modalite):
    # Trouver les feuilles
    feuilles = [req for req in liste_request if not any((req < other) for other in liste_request)]

    nb_modalite = {k: len(v) for k, v in modalite.items()}

    p = sum(produit_modalites(fset, nb_modalite) for fset in feuilles)
    n = sum(produit_modalites(fset, nb_modalite) for fset in liste_request)

    # Table de correspondance beta
    beta_names = []
    df_par_requete = {}

    def beta_label(var_names, values):
        return "β[" + ", ".join(f"{v}={val}" for v, val in zip(var_names, values)) + "]"

    for req in feuilles:
        sorted_vars = sorted(req)
        domains = [range(nb_modalite[v]) for v in sorted_vars]
        combinations_vals = list(product(*domains))

        betas = [beta_label(sorted_vars, vals) for vals in combinations_vals]
        beta_names.extend(betas)

        df = pd.DataFrame(combinations_vals, columns=sorted_vars)
        df["value"] = betas
        df_par_requete[req] = df

    # Matrice X (DataFrame annoté)
    X = np.zeros((n, p), dtype=int)
    ligne_courante = 0
    request_lines = []

    for req in liste_request:
        for max_req in feuilles:
            if req <= max_req:
                df = df_par_requete[max_req].copy()
                if len(req) > 0:
                    grouped = df.groupby(sorted(req))["value"].apply(lambda x: ' + '.join(x)).reset_index()
                else:
                    grouped = pd.DataFrame({'value': [' + '.join(df["value"])]})

                for _, row in grouped.iterrows():
                    beta_sum = row["value"]
                    for b in beta_sum.split(" + "):
                        j = beta_names.index(b)
                        X[ligne_courante, j] = 1
                    if len(req) > 0:
                        dico_modalites = row[sorted(req)].to_dict()
                        dico_modalites_nom = {
                            var: modalite[var][val] if pd.notna(val) else None
                            for var, val in dico_modalites.items()
                        }
                    else:
                        dico_modalites_nom = {}

                    request_lines.append({"requête": req, **dico_modalites_nom})

                    ligne_courante += 1
                break

    X_df_infos = pd.DataFrame(request_lines)

    # Matrice R
    R_rows = []

    for req1, req2 in combinations(feuilles, 2):
        intersection = req1 & req2
        df1 = df_par_requete[req1]
        df2 = df_par_requete[req2]

        if not intersection:
            betas1 = df1["value"].tolist()
            betas2 = df2["value"].tolist()

            row = np.zeros(len(beta_names))
            for b in betas1:
                row[beta_names.index(b)] = 1
            for b in betas2:
                row[beta_names.index(b)] -= 1
            R_rows.append(row)
            continue

        grouped1 = df1.groupby(list(intersection))["value"].apply(list)
        grouped2 = df2.groupby(list(intersection))["value"].apply(list)
        common_modalities = grouped1.index.intersection(grouped2.index)

        for modality in common_modalities:
            betas1 = grouped1[modality]
            betas2 = grouped2[modality]
            row = np.zeros(len(beta_names))
            for b in betas1:
                row[beta_names.index(b)] = 1
            for b in betas2:
                row[beta_names.index(b)] -= 1
            R_rows.append(row)

    def remove_dependent_rows_qr(R, tol=1e-10):
        if R.shape[0] == 0:
            return R
        Q, R_qr = np.linalg.qr(R.T)
        independent = np.abs(R_qr).sum(axis=1) > tol
        return R[independent]

    R = np.vstack(R_rows) if R_rows else np.empty((0, len(beta_names)))
    R = remove_dependent_rows_qr(R)

    return X, R, X_df_infos


def ajouter_colonne_value(x_df_info, data_query, results_store):
    x_df_info = x_df_info.copy()
    x_df_info["value"] = np.nan
    x_df_info["sigma2"] = np.nan

    for key, df_valeurs in results_store.items():

        groupement = data_query[key]["groupement"]

        # Détecter automatiquement la colonne de valeur
        possible_value_cols = ['count', 'sum', 'value']
        valeur_col = next((col for col in possible_value_cols if col in df_valeurs.columns), None)
        if valeur_col is None:
            raise ValueError(f"Aucune colonne de valeur trouvée dans {key}")

        # Filtrage de x_df_info pour le groupement
        masque = x_df_info["requête"] == groupement

        if not any(masque):
            continue

        sous_df = x_df_info[masque].copy()

        if len(groupement) == 0:
            # Cas sans groupement : valeur unique
            if len(df_valeurs) != 1:
                raise ValueError(f"Le résultat pour '{key}' sans groupement contient plusieurs lignes.")
            valeur = df_valeurs[valeur_col].iloc[0]
            x_df_info.loc[masque, "value"] = valeur
        else:
            # Fusion classique
            jointure = pd.merge(
                sous_df,
                df_valeurs,
                how='left',
                on=list(groupement)
            )
            x_df_info.loc[masque, "value"] = jointure[valeur_col].values

        x_df_info.loc[masque, "sigma2"] = data_query[key]["sigma2"]

    return x_df_info


def mettre_a_jour_results_store(x_df_info, data_query, results_store, col_source="value_MCG", col_cible="count"):
    # Copie du results_store mis à jour
    results_store_modifié = {}

    for key, df_valeurs in results_store.items():

        groupement = data_query[key]["groupement"]

        # Filtrage de x_df_info pour le groupement
        masque = x_df_info["requête"] == groupement
        if not any(masque):
            continue

        sous_df_info = x_df_info[masque].copy()

        if len(groupement) == 0:
            # Aucun groupement : valeur unique
            if len(sous_df_info) != 1 or len(df_valeurs) != 1:
                print(sous_df_info)
                print(df_valeurs)
                raise ValueError(f"Incohérence pour '{key}' sans groupement : {len(sous_df_info)} dans x_df_info, {len(df_valeurs)} dans df_valeurs.")
            valeur = sous_df_info[col_source].iloc[0]
            df_valeurs[col_cible] = [valeur]
        else:
            # Jointure sur les colonnes du groupement
            jointure = pd.merge(
                df_valeurs,
                sous_df_info[list(groupement) + [col_source]],
                how="left",
                on=list(groupement)
            )
            df_valeurs[col_cible] = jointure[col_source]

        results_store_modifié[key] = df_valeurs

    return results_store_modifié


def calcul_MCG(results_store, modalite, dict_query, type_req, pos=True):

    liste_query = [query['groupement'] for query in dict_query.values()]

    X, R, X_df_infos = MCG(liste_query, modalite)

    n, p = X.shape

    if p > 0:

        X_df_infos = ajouter_colonne_value(X_df_infos, dict_query, results_store)

        Y = np.array(X_df_infos["value"])
        sigma2 = np.array(X_df_infos["sigma2"])
        Omega_inv = np.diag(1 / sigma2)
        Omega_inv /= np.max(np.diag(Omega_inv))
        # Variables
        beta = cp.Variable(p)

        # Objectif : moindres carrés pondérés
        objective = cp.Minimize(cp.quad_form(Y - X @ beta, Omega_inv))

        # Contraintes

        if R.shape[0] == 0:
            if pos:
                constraints = [
                    beta >= 0
                ]
            else:
                constraints = []
        else:
            if pos:
                constraints = [
                    R @ beta == 0,
                    beta >= 0
                ]
            else:
                constraints = [
                    R @ beta == 0
                ]

        # Problème
        problem = cp.Problem(objective, constraints)
        problem.solve()
        print("status:", problem.status)
        X_beta = X @ beta.value

        X_df_infos["value_MCG"] = X_beta
        print(X_df_infos)

    return mettre_a_jour_results_store(X_df_infos, dict_query, results_store, col_source="value_MCG", col_cible=type_req)


def optimization_boosted(dict_query, modalite):

    liste_query = [query['groupement'] for query in dict_query.values()]

    X, R, X_df_infos = MCG(liste_query, modalite)

    nb_modalite = {k: len(v) for k, v in modalite.items()}

    dict_request = {key: {"nb_cellule": produit_modalites(query["groupement"], nb_modalite), "sigma2": query["sigma2"]} for key, query in dict_query.items()}

    # Matrice de variance Omega (hétéroscédastique)
    sigma2 = np.array(list(itertools.chain.from_iterable(
        [v["sigma2"]] * v["nb_cellule"] for v in dict_request.values()
    )))

    Omega_inv = np.diag(1 / sigma2)

    # Matrice H = X^T Omega^{-1} X
    H = X.T @ Omega_inv @ X
    H_inv = np.linalg.inv(H)

    # Projection liée à la contrainte R beta = 0
    RHinv = R @ H_inv
    middle_term = np.linalg.inv(RHinv @ R.T)
    correction = H_inv @ R.T @ middle_term @ RHinv

    # Variance corrigée de beta_hat sous contrainte R beta = 0
    V_beta_constrained = H_inv - correction

    V_Xbeta_constrained = X @ V_beta_constrained @ X.T
    var_Xbeta_constrained = np.diag(V_Xbeta_constrained)

    index = 0
    # Création du mapping entre frozenset (clé dans dict_request) et la clé d'origine de poids

    for key in dict_request:
        nb = dict_request[key]["nb_cellule"]
        dict_query[key]["scale"] = np.sqrt(var_Xbeta_constrained[index])
        index += nb
    return dict_query


def update_context(CONTEXT_PARAM, budget, requete):

    poids_req_rho = [req["poids"] for req in requete.values() if req["type"].lower() != "quantile"]
    poids_req_eps = [req["poids"] for req in requete.values() if req["type"].lower() == "quantile"]
    budget_rho = budget * sum(poids_req_rho)
    budget_eps = np.sqrt(8 * budget * sum(poids_req_eps))

    if budget_rho == 0:
        context_rho = None
    else:
        context_rho = dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(rho=budget_rho),
            split_by_weights=poids_req_rho
        )

    if budget_eps == 0:
        context_eps = None
    else:
        context_eps = dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(epsilon=budget_eps),
            split_by_weights=poids_req_eps
        )

    return context_rho, context_eps


def rho_from_eps_delta(epsilon, delta):
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    log_term = np.log(1 / delta)
    sqrt_term = np.sqrt(log_term * (epsilon + log_term))
    rho = 2 * log_term + epsilon - 2 * sqrt_term
    return rho


def eps_from_rho_delta(rho, delta):
    if rho <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("rho must be positive and delta in (0, 1)")

    def equation(y, rho, delta):
        denom = delta * (1 + (y - rho) / (2 * rho))
        if denom <= 0:
            return np.inf  # force fsolve à éviter cette zone
        if rho * np.log(1 / denom) <= 0:
            return np.inf  # force fsolve à éviter cette zone
        sqrt_term = np.sqrt(rho * np.log(1 / denom))
        return y - (rho + 2 * sqrt_term)

    epsilon_base = rho + 2 * np.sqrt(rho * np.log(1 / delta))
    y0 = rho + 1

    try:
        result, info, ier, _ = fsolve(equation, y0, args=(rho, delta), full_output=True)
        if ier == 1 and result[0] >= 0:
            return result[0]
        else:
            return epsilon_base
    except Exception:
        return epsilon_base


def parse_single_condition(condition: str) -> pl.Expr:
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


def parse_filter_string(filter_str: str) -> pl.Expr:
    """Transforme une chaîne de filtres combinés en une unique pl.Expr."""
    # Séparation sécurisée via regex avec maintien des opérateurs binaires
    tokens = re.split(r'(\s+\&\s+|\s+\|\s+)', filter_str)
    exprs = []
    ops = []

    for token in tokens:
        token = token.strip()
        if token == "&":
            ops.append("&")
        elif token == "|":
            ops.append("|")
        elif token:  # une condition
            exprs.append(parse_single_condition(token))

    # Combine les expressions avec les bons opérateurs
    expr = exprs[0]
    for op, next_expr in zip(ops, exprs[1:]):
        if op == "&":
            expr = expr & next_expr
        elif op == "|":
            expr = expr | next_expr

    return expr


def manual_quantile_score(data, candidats, alpha, et_si=False):
    if alpha == 0:
        alpha_num, alpha_denum = 0, 1
    elif alpha == 0.25:
        alpha_num, alpha_denum = 1, 4
    elif alpha == 0.5:
        alpha_num, alpha_denum = 1, 2
    elif alpha == 0.75:
        alpha_num, alpha_denum = 3, 4
    elif alpha == 1:
        alpha_num, alpha_denum = 1, 1
    else:
        alpha_num = int(np.floor(alpha * 10_000))
        alpha_denum = 10_000

    if et_si:
        alpha_num = int(np.floor(alpha * 10_000))
        alpha_denum = 10_000

    scores = []
    for c in candidats:
        n_less = np.sum(data < c)
        n_equal = np.sum(data == c)
        score = alpha_denum * n_less - alpha_num * (len(data) - n_equal)
        scores.append(abs(score))

    return np.array(scores), max(alpha_num, alpha_denum - alpha_num)


def get_weights(request, input) -> dict:
    # Étape 1 : récupération des poids bruts
    raw_weights = {
        key: radio_to_weight.get(float(getattr(input, key)()), 0)
        for key in request
    }

    # Étape 2 : normalisation initiale
    total = sum(raw_weights.values())
    weights = {k: v / total for k, v in raw_weights.items()} if total > 0 else {k: 0 for k in raw_weights}

    # Étape 3 : division par 2 des poids de type "Moyenne"
    for k in weights:
        if request[k].get("type") in ["Moyenne", "Total"]:
            weights[k] /= 2

    return weights


def load_data(path: str, storage_options):
    if path.startswith("s3://"):
        df = pl.read_parquet(path, storage_options=storage_options)
    else:
        df = pl.read_parquet(path)

    if "geometry" in df.columns:
        df = df.drop("geometry")

    return df
