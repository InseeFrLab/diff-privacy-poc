import re
import numpy as np
import polars as pl
import pandas as pd
from scipy.optimize import fsolve
import opendp.prelude as dp
from collections import defaultdict
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


def MCG(liste_request, nb_modalite):
    # Trouver les feuilles (non inclus dans d'autres strictement plus grands)
    feuilles = [
        req for req in liste_request
        if not any((req < other) for other in liste_request)
    ]

    p = sum(produit_modalites(fset, nb_modalite) for fset in feuilles)
    n = sum(produit_modalites(fset, nb_modalite) for fset in liste_request)

    # Affichage
    print("Tous les frozensets :", liste_request)
    print("Feuilles :", feuilles)

    # 6. Génération des noms des bêtas
    beta_names = [f"beta_{i+1}" for i in range(p)]

    # 7. Construction des DataFrames pour chaque requête maximale
    df_par_requete = {}
    compteur = 0

    for req in feuilles:
        sorted_vars = sorted(req)
        domains = [range(nb_modalite[v]) for v in sorted_vars]
        combinations_vals = list(product(*domains))  # toutes les combinaisons pour cette requête
        nb = len(combinations_vals)

        # Associer les bonnes bêtas
        betas = beta_names[compteur:compteur + nb]
        compteur += nb

        # Construire le DataFrame
        df = pd.DataFrame(combinations_vals, columns=sorted_vars)
        df["value"] = betas
        df_par_requete[req] = df

    # 8. Construction de la matrice X (n x p)
    X = np.zeros((n, p), dtype=int)

    # 8.2 Reste : combinaisons/group_by
    ligne_courante = 0
    for req in liste_request:
        # Chercher un maximal qui contient la requête
        for max_req in feuilles:
            if req <= max_req:
                df = df_par_requete[max_req].copy()
                # groupby + concaténation des strings de valeurs
                if len(req) > 0:
                    grouped = df.groupby(sorted(req))["value"].apply(lambda x: ' + '.join(x)).reset_index()
                else:
                    # cas du total : groupby sur rien → tout sommer
                    grouped = pd.DataFrame({'value': [' + '.join(df["value"])]})

                for _, row in grouped.iterrows():
                    beta_sum = row["value"]
                    for b in beta_sum.split(" + "):
                        j = beta_names.index(b)
                        X[ligne_courante, j] = 1
                    ligne_courante += 1
                break

    # 10. R
    R_rows = []

    # On parcourt toutes les paires de requêtes maximales
    for req1, req2 in combinations(feuilles, 2):
        intersection = req1 & req2
        df1 = df_par_requete[req1]
        df2 = df_par_requete[req2]

        if not intersection:
            # Cas particulier : contrainte sur le total global (somme de toutes les cellules)
            betas1 = df1["value"].tolist()
            betas2 = df2["value"].tolist()

            row = np.zeros(len(beta_names))
            for b in betas1:
                row[beta_names.index(b)] = 1
            for b in betas2:
                row[beta_names.index(b)] -= 1
            R_rows.append(row)
            continue

        # Regrouper par les variables communes (intersection)
        grouped1 = df1.groupby(list(intersection))["value"].apply(list)
        grouped2 = df2.groupby(list(intersection))["value"].apply(list)

        # On parcourt les modalités communes aux deux tables
        common_modalities = grouped1.index.intersection(grouped2.index)

        modalities = list(common_modalities)

        for modality in modalities:
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
        Q, R_qr = np.linalg.qr(R.T)  # QR sur les colonnes revient à détecter les lignes dépendantes
        independent = np.abs(R_qr).sum(axis=1) > tol
        return R[independent]

    # Finalisation
    R = np.vstack(R_rows) if R_rows else np.empty((0, len(beta_names)))
    R = remove_dependent_rows_qr(R)
    return X, R

def MCG(liste_request, nb_modalite):
    # Trouver les feuilles
    feuilles = [req for req in liste_request if not any((req < other) for other in liste_request)]

    p = sum(produit_modalites(fset, nb_modalite) for fset in feuilles)
    n = sum(produit_modalites(fset, nb_modalite) for fset in liste_request)

    print("Tous les frozensets :", liste_request)
    print("Feuilles :", feuilles)

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
                    else:
                        dico_modalites = {}
                    request_lines.append({"requête": req, **dico_modalites})
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



def calcul_MCG(results_store, nb_modalite, poids):

    poids_set = {
        frozenset() if v['groupement'] == 'Aucun' else frozenset([v['groupement']]) if isinstance(v['groupement'], str) else frozenset(v['groupement']): v["poids_normalise"]
        for v in poids.values()
    }

    liste_request = [s for s in poids_set.keys()]

    X, R, X_df_infos = MCG(liste_request, nb_modalite)

    n, p = X.shape

    dict_request = {fset: {"nb_cellule": produit_modalites(fset, nb_modalite), "sigma2": 1/(2*budget_rho*poids_set[fset])} for fset in liste_request}

    # Matrice de variance Omega (hétéroscédastique)
    sigma2 = np.array(list(itertools.chain.from_iterable(
        [v["sigma2"]] * v["nb_cellule"] for v in dict_request.values()
    )))

    Y = np.array([20, 12, 23, 10, 7, 9, 13, 21, 23, 24, 25, 20, 20, 27, 59])
    sigma2 = np.linspace(5, 20, n)
    Omega_inv = np.diag(1 / sigma2)

    # Variables
    beta = cp.Variable(p)

    # Objectif : moindres carrés pondérés
    objective = cp.Minimize(cp.quad_form(Y - X @ beta, Omega_inv))

    # Contraintes
    constraints = [
        R @ beta == 0,
        beta >= 0
    ]

    # Problème
    problem = cp.Problem(objective, constraints)
    problem.solve()
    print("status:", problem.status)
    X_beta = X @ beta.value
    X_beta_rounded = np.round(X_beta).astype(int)
    print("X*beta estimé :", X_beta)
    print("X*beta estimé rounded :", X_beta_rounded)

    return X_beta_rounded


def optimization_boosted(budget_rho, nb_modalite, poids):

    poids_set = {
        frozenset() if v['groupement'] == 'Aucun' else frozenset([v['groupement']]) if isinstance(v['groupement'], str) else frozenset(v['groupement']): v["poids_normalise"]
        for v in poids.values()
    }
    liste_request = [s for s in poids_set.keys()]

    X, R, X_df_infos = MCG(liste_request, nb_modalite)
    print(X)

    print(X_df_infos)

    dict_request = {fset: {"nb_cellule": produit_modalites(fset, nb_modalite), "sigma2": 1/(2*budget_rho*poids_set[fset])} for fset in liste_request}
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
    fs_to_key = {
        frozenset() if v['groupement'] == 'Aucun' else frozenset([v['groupement']]) if isinstance(v['groupement'], str) else frozenset(v['groupement']): k
        for k, v in poids.items()
    }
    for fset in dict_request:
        nb = dict_request[fset]["nb_cellule"]
        if fset in fs_to_key:
            original_key = fs_to_key[fset]
            poids[original_key]["scale"] = np.sqrt(var_Xbeta_constrained[index])
        index += nb
    return poids


def ameliorer_total(key, req, df_result, poids_estimateur_tot, results_store, lien_total_req_dict):

    variable = req["variable"]
    vars_key = next((k for k, v in lien_total_req_dict[variable].items() if v == key), None)
    poids_dict = poids_estimateur_tot[variable][vars_key]
    df_result_base = df_result.copy() if df_result is not None else None
    vars_key_list = [vars_key] if isinstance(vars_key, str) else list(vars_key)
    if df_result_base is not None:
        df_result_base['sum_amelioree'] = 0.0

    for ref_key, poids in poids_dict.items():
        if poids == 0:
            continue

        df_ref = results_store.get(lien_total_req_dict[variable][ref_key])
        df_ref = df_ref.copy()

        if ref_key == vars_key:
            df_result_base['sum_amelioree'] += poids * df_result_base['sum']

        else:
            group_vars = [ref_key] if isinstance(ref_key, str) else list(ref_key)
            common_vars = list(set(group_vars) & set(vars_key_list))

            if not common_vars:
                total_ref = df_ref["sum"].sum()
                if df_result_base is None:
                    # Créer un DataFrame d'une seule ligne
                    df_result_base = pd.DataFrame([{'sum_amelioree': poids * total_ref}])
                else:
                    df_result_base['sum_amelioree'] += poids * total_ref
            else:
                df_proj = (
                    df_ref
                    .groupby(common_vars, as_index=False)
                    .agg({'sum': 'sum'})
                    .rename(columns={'sum': 'sum_ref'})
                )

                if df_result_base is None:
                    df_result_base = df_proj.copy()
                    df_result_base['sum_amelioree'] = poids * df_result_base['sum_ref']
                    df_result_base = df_result_base.drop(columns=['sum_ref'])
                else:
                    merged = df_result_base.merge(df_proj, on=common_vars, how='left')
                    df_result_base['sum_amelioree'] += poids * merged['sum_ref']

    # ✅ Remplacer la colonne sum
    df_result_base['sum'] = df_result_base['sum_amelioree'].round(0).clip(lower=0).astype(int)
    df_result_base = df_result_base.drop(columns=['sum_amelioree'])
    return df_result_base


def ameliorer_comptage(key, df_result, poids_estimateur, results_store, lien_comptage_req):
    vars_key = next((k for k, v in lien_comptage_req.items() if v == key), None)
    poids_dict = poids_estimateur[vars_key]
    df_result_base = df_result.copy() if df_result is not None else None
    vars_key_list = [vars_key] if isinstance(vars_key, str) else list(vars_key)
    if df_result_base is not None:
        df_result_base['count_amelioree'] = 0.0

    for ref_key, poids in poids_dict.items():
        if poids == 0:
            continue

        df_ref = results_store.get(lien_comptage_req[ref_key])
        df_ref = df_ref.copy()

        if ref_key == vars_key:
            df_result_base['count_amelioree'] += poids * df_result_base['count']

        else:
            group_vars = [ref_key] if isinstance(ref_key, str) else list(ref_key)
            common_vars = list(set(group_vars) & set(vars_key_list))

            if not common_vars:
                total_ref = df_ref["count"].sum()
                if df_result_base is None:
                    # Créer un DataFrame d'une seule ligne
                    df_result_base = pd.DataFrame([{'count_amelioree': poids * total_ref}])
                else:
                    df_result_base['count_amelioree'] += poids * total_ref
            else:
                df_proj = (
                    df_ref
                    .groupby(common_vars, as_index=False)
                    .agg({'count': 'sum'})
                    .rename(columns={'count': 'count_ref'})
                )

                if df_result_base is None:
                    df_result_base = df_proj.copy()
                    df_result_base['count_amelioree'] = poids * df_result_base['count_ref']
                    df_result_base = df_result_base.drop(columns=['count_ref'])
                else:
                    merged = df_result_base.merge(df_proj, on=common_vars, how='left')
                    df_result_base['count_amelioree'] += poids * merged['count_ref']

    # ✅ Remplacer la colonne count
    df_result_base['count'] = df_result_base['count_amelioree'].round(0).clip(lower=0).astype(int)
    df_result_base = df_result_base.drop(columns=['count_amelioree'])
    return df_result_base


def update_context(CONTEXT_PARAM, budget, requete):

    poids_req_rho = [req["poids"] for req in requete.values() if req["type"].lower() != "quantile"]
    poids_req_eps = 1 - poids_req_rho
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


# Fonction pour forcer une variable à 0 en modifiant A et b
def impose_zero(A, b, index):
    A[index, :] = 0
    A[index, index] = 1
    b[index] = 0
    return A, b


# Résolution itérative avec contraintes positives
def solve_projected(A, b):
    for _ in range(10):  # max 10 itérations
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("❌ Système non inversible.")
            return None

        if np.all(x >= -1e-10):
            return np.clip(x, 0, None)

        # On impose à 0 les variables négatives
        for i, val in enumerate(x):
            if val < 0:
                A, b = impose_zero(A, b, i)

    print("❌ Convergence non atteinte.")
    return None


def sys_budget_dp(budget_rho, nb_modalite, poids):
    total = sum(poids.values())
    poids = {k: v / total for k, v in poids.items()}
    poids_set = {
        frozenset() if k == 'Aucun' else frozenset([k]) if isinstance(k, str) else frozenset(k): v
        for k, v in poids.items()
    }
    subsets = [s for s in poids_set.keys()]
    N = len(poids_set)

    # Matrices initialisées
    Q = np.zeros((N, N))
    # Construction de Q
    for i, Ei in enumerate(subsets):
        for j, Ej in enumerate(subsets):
            if Ei == Ej:
                Q[i, j] = 1
            elif Ei.issubset(Ej):
                diff = Ej - Ei
                prod = 1
                for k in diff:
                    prod *= nb_modalite[k]
                Q[i, j] = 1 / prod

    P = np.zeros((N, 1))
    for i, subset in enumerate(subsets):
        P[i, 0] = - poids_set.get(frozenset(subset), 0)

    b = np.zeros((N + 1, 1))
    b[-1, 0] = budget_rho

    A = np.zeros((N + 1, N + 1))
    A[:N, N] = P.flatten()
    A[:N, :N] = Q
    A[N, :N] = 1

    x_sol = solve_projected(A.copy(), b.copy())
    rho_req = {}
    rho_atteint = {}
    poids_estimateur = {}
    if x_sol is not None:
        print(f"ρ optimal = {x_sol[N].item():.3f} vs ρ budget = {budget_rho:.3f} (Gain net de {(x_sol[N].item()-budget_rho):.3f})")
        for i, ((nom, p), x) in enumerate(zip(poids.items(), x_sol[:N])):
            if x.item() != 0:
                var_estim = 1/(2*p*x_sol[N].item())
                rho_req[nom] = x.item()
                print(f"Ecart type de la requête pour {nom} = {np.sqrt(1/(2*x.item())):.2f}Δ et écart type de l'estimation = {np.sqrt(var_estim):.2f}Δ")
            else:
                var_estim = 1/(2*np.dot(Q[i], x_sol[:N].flatten()))
                print(f"Pas de requête pour {nom} et écart type de l'estimation = {np.sqrt(var_estim):.2f}Δ")

            rho_atteint[nom] = 1/(2*var_estim)
            poids_estimateur[nom] = {}
            for j, ((nom_2, _), x_2) in enumerate(zip(poids.items(), x_sol[:N])):
                poids_estim = var_estim * Q[i][j] * 2 * x_sol[j].item()
                if poids_estim > 0:
                    poids_estimateur[nom][nom_2] = poids_estim
                    print(f"    - Poids de l'estimation par la requête {nom_2} = {poids_estim:.2f}")
    else:
        print("❌ Aucune solution admissible.")
    print(f"---------------------------------------------------------------------------------------------------")
    return rho_atteint, rho_req, poids_estimateur


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


def organiser_par_by(dico_requetes, dico_poids):
    lien_croisement_req = defaultdict(dict)

    # Première passe : création des entrées avec poids brut
    for query, params in dico_requetes.items():
        if 'by' not in params:
            groupement = 'Aucun'
        else:
            by = params['by']
            if isinstance(by, str):
                groupement = by
            elif isinstance(by, list):
                groupement = by[0] if len(by) == 1 else tuple(by)
            else:
                groupement = 'Aucun'

        # Récupération des noms de requêtes associés
        reqs = params.get('req', [])

        # Calcul du poids total
        poids_total = sum(dico_poids.get(r, 0) for r in reqs)

        # Ajout à la structure finale
        lien_croisement_req[query] = {
            "groupement": groupement,
            "req": reqs,
            "poids": poids_total
        }

    # Deuxième passe : calcul des poids normalisés
    total_poids = sum(v["poids"] for v in lien_croisement_req.values())
    if total_poids > 0:
        for v in lien_croisement_req.values():
            v["poids_normalise"] = v["poids"] / total_poids
    else:
        for v in lien_croisement_req.values():
            v["poids_normalise"] = 0.0  # ou None si tu préfères

    return dict(lien_croisement_req)


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
        if request[k].get("type") == "Moyenne":
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
