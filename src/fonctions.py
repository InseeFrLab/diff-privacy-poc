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
import os
from itertools import combinations, product
import itertools
from functools import reduce
import cvxpy as cp
from typing import Optional, Any

# Map des opérateurs Python vers leurs fonctions correspondantes
OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


def optimisation_chaine(dict_query, modalite) :
    filtres_uniques = set(v.get("filtre") for v in dict_query.values())
    variables_uniques = set(v.get("variable") for v in dict_query.values())

    requetes_finales = {}

    for filtre in filtres_uniques:
        for variable in variables_uniques:
            query_filtre_variable = {
                k: v for k, v in dict_query.items()
                if v.get("variable") == variable and v.get("filtre") == filtre
            }
            query_filtre_variable_opt = optimization_boosted(
                dict_query=query_filtre_variable, modalite=modalite
            )
            requetes_finales.update(query_filtre_variable_opt)

    return requetes_finales


def save_yaml_metadata_from_dataframe(lf: pl.DataFrame, dataset_name: str = "dataset") -> None:
    # Résolution anticipée du schéma et des noms de colonnes
    schema = lf.collect_schema()
    colnames = list(schema.keys())

    # Génération des expressions
    exprs = []
    for col in colnames:
        exprs.extend([
            pl.col(col).null_count().alias(f"{col}__nulls"),
            pl.col(col).n_unique().alias(f"{col}__unique"),
            pl.col(col).min().alias(f"{col}__min"),
            pl.col(col).max().alias(f"{col}__max"),
        ])

    collected = lf.select(exprs).collect()
    n_rows = lf.select(pl.len()).collect().item()

    # Construction du dictionnaire de métadonnées
    metadata = {
        'dataset_name': dataset_name,
        'n_rows': n_rows,
        'n_columns': len(colnames),
        'columns': {}
    }

    for col in colnames:
        dtype = schema[col]
        col_meta = {
            'type': str(dtype),
            'missing': int(collected[f"{col}__nulls"][0])
        }

        if dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64}:
            col_meta['min'] = (
                round(float(collected[f"{col}__min"][0]), 2)
                if collected[f"{col}__min"][0] is not None else None
            )
            col_meta['max'] = (
                round(float(collected[f"{col}__max"][0]), 2)
                if collected[f"{col}__max"][0] is not None else None
            )

        col_meta['unique_values'] = int(collected[f"{col}__unique"][0])

        metadata['columns'][col] = col_meta

    # Sauvegarde du YAML dans le dossier "data"
    os.makedirs("data", exist_ok=True)
    yaml_path = os.path.join("data", f"{dataset_name}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, sort_keys=False, allow_unicode=True)


def load_yaml_metadata(dataset_name: str = "dataset") -> dict:
    yaml_path = os.path.join("data", f"{dataset_name}.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    return metadata


def intervalle_confiance_quantile(dataset: pl.LazyFrame, req: dict, epsilon: float, vrai_tableau: pl.DataFrame):
    variable = req.get("variable")
    bounds_min, bounds_max = req.get("bounds")
    alphas = [float(a) for a in req.get("alpha")]
    nb_candidats = int(req.get("nb_candidats"))
    by = req.get("by")

    candidats = np.linspace(bounds_min, bounds_max, nb_candidats + 1)
    precisions_by_alpha = {alpha: [] for alpha in alphas}

    def process_data(data_variable: np.ndarray, vraie_value: float, alpha: float):
        if data_variable.size == 0:
            return None

        scores, sensi = manual_quantile_score(
            data_variable,
            candidats,
            alpha=alpha,
            et_si=True
        )

        scaled_scores = -epsilon * scores / (2 * sensi)
        scaled_scores -= scaled_scores.max()  # stabilisation

        proba_non_norm = np.exp(scaled_scores)
        proba = proba_non_norm / proba_non_norm.sum()

        sorted_indices = np.argsort(proba)[::-1]
        sorted_candidats = candidats[sorted_indices]

        cumulative = np.cumsum(proba[sorted_indices])
        top95_mask = cumulative <= 0.95
        if not np.all(top95_mask):
            top95_mask[np.argmax(cumulative > 0.95)] = True

        candidats_top95 = sorted_candidats[top95_mask]
        ic_sup = np.max(candidats_top95)
        ic_inf = np.min(candidats_top95)
        return max(ic_sup - ic_inf, ic_sup - vraie_value, vraie_value - ic_inf)

    if by is None:
        # Cas sans group_by
        data_variable = dataset.select(variable).collect()[variable].to_numpy()
        for alpha in alphas:
            col_quantile = f"quantile_{alpha}"
            vraie_value = vrai_tableau[col_quantile][0]
            precision_val = process_data(data_variable, vraie_value, alpha)
            if precision_val is not None:
                precisions_by_alpha[alpha].append(precision_val)
    else:
        # Cas avec group_by
        df = dataset.select([*by, variable]).collect()
        grouped = df.group_by(by, maintain_order=True)

        for group_key, group_df in grouped:
            filtre_expr = None
            for col, val in zip(by, group_key):
                condition = pl.col(col) == val
                filtre_expr = condition if filtre_expr is None else (filtre_expr & condition)
            ligne = vrai_tableau.filter(filtre_expr)

            if ligne.is_empty():
                continue

            data_variable = group_df[variable].to_numpy()

            for alpha in alphas:
                col_quantile = f"quantile_{alpha}"
                if col_quantile not in ligne.columns:
                    continue
                vraie_value = ligne[col_quantile][0]
                precision_val = process_data(data_variable, vraie_value, alpha)
                if precision_val is not None:
                    precisions_by_alpha[alpha].append(precision_val)

    # Moyenne des précisions par quantile
    return {
        f"quantile_{alpha}": (
            np.mean(precisions_by_alpha[alpha]) if precisions_by_alpha[alpha] else None
        )
        for alpha in alphas
    }


def generate_yaml_metadata_from_dataframe(lf: pl.DataFrame, dataset_name: str = "dataset") -> str:

    # Résolution anticipée du schéma et des noms de colonnes
    schema = lf.collect_schema()
    colnames = list(schema.keys())

    # Génération des expressions
    exprs = []
    for col in colnames:
        exprs.extend([
            pl.col(col).null_count().alias(f"{col}__nulls"),
            pl.col(col).n_unique().alias(f"{col}__unique"),
            pl.col(col).min().alias(f"{col}__min"),
            pl.col(col).max().alias(f"{col}__max"),
        ])

    collected = lf.select(exprs).collect()
    n_rows = lf.select(pl.len()).collect().item()

    metadata = {
        'dataset_name': dataset_name,
        'n_rows': n_rows,
        'n_columns': len(colnames),
        'columns': {}
    }

    for col in colnames:
        dtype = schema[col]
        col_meta = {
            'type': str(dtype),
            'missing': int(collected[f"{col}__nulls"][0])
        }

        if dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64}:
            col_meta['min'] = (
                round(float(collected[f"{col}__min"][0]), 2)
                if collected[f"{col}__min"][0] is not None else None
            )
            col_meta['max'] = (
                round(float(collected[f"{col}__max"][0]), 2)
                if collected[f"{col}__max"][0] is not None else None
            )

        elif dtype in {pl.Utf8, pl.Categorical}:
            col_meta['unique_values'] = int(collected[f"{col}__unique"][0])

        metadata['columns'][col] = col_meta

    yaml_str = yaml.dump(metadata, sort_keys=False, allow_unicode=True)
    return yaml_str


# ------------------------------------------
# Utils: produit des modalités pour un frozenset
def produit_modalites(fset, nb_modalite):
    if not fset:
        return 1
    return reduce(operator.mul, (nb_modalite[v] for v in fset), 1)


# ------------------------------------------
# Construction des matrices X (design) et R (contraintes)
def MCG(liste_requests, modalite):
    """
    Construit la matrice de design X, la matrice de contraintes R,
    et un DataFrame annoté des requêtes (X_df_infos).

    Args:
        liste_requests : liste de frozensets représentant les groupements.
        modalite : dict {variable: [modalités]}.

    Returns:
        X : matrice numpy (n x p)
        R : matrice numpy (r x p) des contraintes linéaires
        X_df_infos : DataFrame avec informations sur chaque ligne de X
    """
    nb_modalite = {k: len(v) for k, v in modalite.items()}

    # Les feuilles sont les groupements qui ne sont inclus dans aucun autre
    feuilles = [req for req in liste_requests if not any(req < other for other in liste_requests)]

    p = sum(produit_modalites(fset, nb_modalite) for fset in feuilles)
    n = sum(produit_modalites(fset, nb_modalite) for fset in liste_requests)

    # Construction des noms des coefficients beta (liste)
    beta_names = []
    df_par_requete = {}

    def beta_label(vars_names, values):
        return "β[" + ", ".join(f"{v}={val}" for v, val in zip(vars_names, values)) + "]"

    # Pré-calcul des DataFrames pour chaque feuille (modalités et noms beta)
    for req in feuilles:
        sorted_vars = sorted(req)
        domains = [range(nb_modalite[v]) for v in sorted_vars]
        combos = list(product(*domains))
        betas = [beta_label(sorted_vars, vals) for vals in combos]
        beta_names.extend(betas)
        df_par_requete[req] = pd.DataFrame(combos, columns=sorted_vars).assign(value=betas)

    # Construction de X et X_df_infos
    X = np.zeros((n, p), dtype=int)
    ligne_courante = 0
    request_lines = []

    for req in liste_requests:
        for max_req in feuilles:
            if req <= max_req:
                df = df_par_requete[max_req].copy()
                if len(req) > 0:
                    grouped = df.groupby(sorted(req))["value"].apply(lambda x: ' + '.join(x)).reset_index()
                else:
                    grouped = pd.DataFrame({'value': [' + '.join(df["value"])]})

                for _, row in grouped.iterrows():
                    for b in row["value"].split(" + "):
                        X[ligne_courante, beta_names.index(b)] = 1

                    if len(req) > 0:
                        dico_modalites = row[sorted(req)].to_dict()
                        dico_modalites_nom = {var: modalite[var][val] for var, val in dico_modalites.items()}
                    else:
                        dico_modalites_nom = {}

                    request_lines.append({"requête": req, **dico_modalites_nom})
                    ligne_courante += 1
                break

    X_df_infos = pd.DataFrame(request_lines)

    # Construction matrice R (contraintes) : égalité entre coefficients beta sur intersections
    R_rows = []

    for req1, req2 in combinations(feuilles, 2):
        inter = req1 & req2
        df1, df2 = df_par_requete[req1], df_par_requete[req2]

        if not inter:
            row = np.zeros(len(beta_names))
            for b in df1["value"]:
                row[beta_names.index(b)] = 1
            for b in df2["value"]:
                row[beta_names.index(b)] = -1
            R_rows.append(row)
            continue

        grouped1 = df1.groupby(list(inter))["value"].apply(list)
        grouped2 = df2.groupby(list(inter))["value"].apply(list)
        common_modalities = grouped1.index.intersection(grouped2.index)

        for modality in common_modalities:
            row = np.zeros(len(beta_names))
            for b in grouped1[modality]:
                row[beta_names.index(b)] = 1
            for b in grouped2[modality]:
                row[beta_names.index(b)] = -1
            R_rows.append(row)

    R = np.vstack(R_rows) if R_rows else np.empty((0, len(beta_names)))

    # Supprime les contraintes dépendantes (ligne linéairement dépendante)
    def remove_dependent_rows_qr(R, tol=1e-10):
        if R.shape[0] == 0:
            return R
        Q, R_qr = np.linalg.qr(R.T)
        independent = np.abs(R_qr).sum(axis=1) > tol
        return R[independent]

    R = remove_dependent_rows_qr(R)

    return X, R, X_df_infos


# ------------------------------------------
# Intégration des valeurs observées et variances dans le DataFrame info
def ajouter_colonne_value(x_df_info, data_query, results_store):
    """
    Ajoute les colonnes "value" et "sigma2" dans X_df_infos à partir des résultats
    et des requêtes.
    """
    X_df_infos = x_df_info.copy()
    X_df_infos["value"] = np.nan
    X_df_infos["sigma2"] = np.nan

    for key, df_valeurs in results_store.items():
        groupement = data_query[key]["groupement"]

        # Identifier la colonne valeur dans le DataFrame
        colonnes_valeur = ['count', 'sum', 'value']
        valeur_col = next((col for col in colonnes_valeur if col in df_valeurs.columns), None)
        if valeur_col is None:
            raise ValueError(f"Aucune colonne de valeur trouvée dans le résultat '{key}'.")

        masque = X_df_infos["requête"] == groupement
        if not masque.any():
            continue

        sous_df = X_df_infos[masque]

        if len(groupement) == 0:
            if len(df_valeurs) != 1:
                raise ValueError(f"Résultat sans groupement '{key}' contient plusieurs lignes.")
            valeur = df_valeurs[valeur_col].iloc[0]
            X_df_infos.loc[masque, "value"] = valeur
        else:
            jointure = pd.merge(sous_df, df_valeurs, how='left', on=list(groupement))
            X_df_infos.loc[masque, "value"] = jointure[valeur_col].values

        X_df_infos.loc[masque, "sigma2"] = data_query[key]["sigma2"]

    return X_df_infos


# ------------------------------------------
# Mise à jour du dictionnaire results_store avec les valeurs recalculées
def mettre_a_jour_results_store(x_df_info, data_query, results_store, col_source="value_MCG", col_cible="count"):
    """
    Met à jour les DataFrames dans results_store avec les valeurs calculées.
    """
    results_modif = {}

    for key, df_valeurs in results_store.items():
        groupement = data_query[key]["groupement"]
        masque = x_df_info["requête"] == groupement
        if not masque.any():
            continue

        sous_df_info = x_df_info[masque]

        if len(groupement) == 0:
            if len(sous_df_info) != 1 or len(df_valeurs) != 1:
                raise ValueError(f"Incohérence dans '{key}': {len(sous_df_info)} lignes dans X_df_infos, {len(df_valeurs)} dans df_valeurs.")
            valeur = sous_df_info[col_source].iloc[0]
            df_valeurs[col_cible] = [valeur]
        else:
            jointure = pd.merge(df_valeurs, sous_df_info[list(groupement) + [col_source]], how="left", on=list(groupement))
            df_valeurs[col_cible] = jointure[col_source]

        results_modif[key] = df_valeurs

    return results_modif


# ------------------------------------------
# Calcul des coefficients beta via moindres carrés pondérés sous contraintes
def calcul_MCG(results_store, modalite, dict_query, type_req, pos=True):
    """
    Calcule les coefficients beta via MCG et met à jour results_store.

    Args:
        results_store : dict de DataFrames des résultats initiaux.
        modalite : dict des modalités.
        dict_query : dict des requêtes avec infos.
        type_req : nom de la colonne cible à mettre à jour dans results_store.
        pos : bool, impose beta >= 0 si True.

    Returns:
        dict de DataFrames mis à jour.
    """
    liste_requests = [d["groupement"] for d in dict_query.values()]

    if len(liste_requests) == 0:
        return None

    X, R, X_df_infos = MCG(liste_requests, modalite)
    X_df_infos = ajouter_colonne_value(X_df_infos, dict_query, results_store)

    if len(liste_requests) == 1:
        X_df_infos[type_req + "_MCG"] = X_df_infos["value"].values
        results_modif = mettre_a_jour_results_store(X_df_infos, dict_query, results_store, col_source=type_req + "_MCG", col_cible=type_req)
        return results_modif

    # Pondération par sigma2
    sigma = X_df_infos["sigma2"].values
    W = np.diag(1 / sigma)
    y = X_df_infos["value"].values
    Omega_inv = W.T @ W
    Omega_inv = Omega_inv / Omega_inv.diagonal().max()  # Normalisation

    # Résolution avec cvxpy
    beta = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.quad_form(X @ beta - y, Omega_inv))
    contraintes = []

    if pos:
        contraintes.append(beta >= 0)

    if R.size > 0:
        contraintes.append(R @ beta == 0)

    prob = cp.Problem(objective, contraintes)
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("Problème d'optimisation non résolu")

    beta_val = beta.value

    # Ajout dans X_df_infos des valeurs prédites par MCG
    X_df_infos[type_req + "_MCG"] = X @ beta_val

    # Mise à jour results_store
    print(X_df_infos)
    results_modif = mettre_a_jour_results_store(X_df_infos, dict_query, results_store, col_source=type_req + "_MCG", col_cible=type_req)
    return results_modif


# ------------------------------------------
# Estimation de l'incertitude (variance corrigée) via méthode boostée
def optimization_boosted(modalite, dict_query):
    """
    Calcule la variance corrigée de beta sous contraintes, met à jour dict_query avec 'scale'.
    """
    liste_requests = [d["groupement"] for d in dict_query.values()]
    nb_modalite = {k: len(v) for k, v in modalite.items()}
    X, R, X_df_infos = MCG(liste_requests, modalite)

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

    # Calcul scale par requête
    for key in dict_request:
        nb = dict_request[key]["nb_cellule"]
        dict_query[key]["scale"] = np.sqrt(var_Xbeta_constrained[index])
        index += nb
    return dict_query


def update_context(CONTEXT_PARAM, budget, requete):
    # Séparer les poids selon le type de requête
    poids_rho = [req["poids"] for req in requete.values() if req["type"].lower() != "quantile"]
    poids_eps = [req["poids"] for req in requete.values() if req["type"].lower() == "quantile"]

    somme_rho = sum(poids_rho)
    somme_eps = sum(poids_eps)

    budget_rho = budget * somme_rho
    budget_eps = np.sqrt(8 * budget * somme_eps)

    def create_context(budget_val, poids, is_rho):
        if budget_val == 0:
            return None
        return dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(rho=budget_val) if is_rho else dp.loss_of(epsilon=budget_val),
            split_by_weights=poids
        )

    context_rho = create_context(budget_rho, poids_rho, is_rho=True)
    context_eps = create_context(budget_eps, poids_eps, is_rho=False)

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


def parse_filter_string(filter_str: str, columns: Optional[list[str]] = None) -> pl.Expr:
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
        elif token:  # une condition
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


def manual_quantile_score(data, candidats, alpha, et_si=False):
    def get_fractional_alpha(alpha_val):
        known_alphas = {0: (0, 1), 0.25: (1, 4), 0.5: (1, 2), 0.75: (3, 4), 1: (1, 1)}
        return known_alphas.get(alpha_val, (int(np.floor(alpha_val * 10_000)), 10_000))

    alpha_num, alpha_denum = get_fractional_alpha(alpha)
    if et_si:
        alpha_num, alpha_denum = int(np.floor(alpha * 10_000)), 10_000

    data_len = len(data)
    if data_len == 0:
        return np.array([]), max(alpha_num, alpha_denum - alpha_num)

    sorted_data = np.sort(data)

    scores = []
    for c in candidats:
        # nombre d'éléments < c : recherche d'indice d'insertion à gauche
        n_less = np.searchsorted(sorted_data, c, side='left')
        # nombre d'éléments == c : différence d'indices d'insertion droite et gauche
        n_equal = np.searchsorted(sorted_data, c, side='right') - n_less

        score = alpha_denum * n_less - alpha_num * (data_len - n_equal)
        scores.append(abs(score))

    max_alpha = max(alpha_num, alpha_denum - alpha_num)
    return np.array(scores), max_alpha


def get_weights(request: dict[str, dict[str, Any]], dict_values: dict[str, str]) -> dict:
    # Étape 1 : récupération des poids bruts
    raw_weights = {
        key: radio_to_weight.get(float(dict_values[key]), 0)
        for key in request.keys()
    }

    # Étape 2 : normalisation initiale
    total = sum(raw_weights.values())
    if total > 0:
        weights = {k: v / total for k, v in raw_weights.items()}
    else:
        weights = {k: 0 for k in raw_weights}

    # Étape 3 : ajustement selon le type de requête
    adjustment_factors = {
        "Moyenne": 2,
        "Total": 2,
        "Ratio": 3,
    }

    for k, v in weights.items():
        req_type = request[k].get("type")
        factor = adjustment_factors.get(req_type, 1)
        weights[k] = v / factor

    return weights


def load_data(path: str, storage_options=None) -> pl.LazyFrame:
    read_kwargs = {"storage_options": storage_options} if path.startswith("s3://") else {}
    lf = pl.read_parquet(path, **read_kwargs).lazy()

    # On vérifie les colonnes pour éviter d'inclure "geometry"
    # Astuce : on supprime la colonne sans collecter en amont
    return lf.drop("geometry") if "geometry" in lf.schema else lf


def extract_column_names_from_choices(choices: dict) -> list[str]:
    """À partir du dict retourné par `variable_choices`, extrait la liste plate des noms de colonnes."""
    columns = []
    for key, val in choices.items():
        if isinstance(val, dict):  # sections comme "Qualitatives" ou "Quantitatives"
            columns.extend(val.values())
    return columns


def extract_bounds(metadata: dict, var_name: str) -> list[float] | None:
    if 'columns' not in metadata or var_name not in metadata['columns']:
        return None
    col_meta = metadata['columns'][var_name]
    min_val = col_meta.get('min')
    max_val = col_meta.get('max')
    if min_val is not None and max_val is not None:
        return [float(min_val), float(max_val)]
    return None


def same_base_request(a: dict, b: dict) -> bool:
    return (
        a.get("type") == b.get("type") and
        a.get("variable") == b.get("variable") and
        a.get("bounds") == b.get("bounds") and
        a.get("by", []) == b.get("by", []) and
        a.get("filtre") == b.get("filtre")
    )


def same_quantile_params(a: dict, b: dict) -> bool:
    return (
        a.get("alpha") == b.get("alpha") and
        a.get("nb_candidats") == b.get("nb_candidats")
    )


def same_ratio_params(a: dict, b: dict) -> bool:
    return (
        a.get("variable_denominateur") == b.get("variable_denominateur") and
        a.get("bounds_denominateur") == b.get("bounds_denominateur")
    )


def assert_or_notify(condition: bool, message: str) -> bool:
    if not condition:
        ui.notification_show(f"❌ {message}", type="error")
        return False
    return True
