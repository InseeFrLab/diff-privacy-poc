import re
import numpy as np
import polars as pl
import pandas as pd
from scipy.optimize import fsolve
import opendp.prelude as dp
from collections import defaultdict
from src.constant import (
    OPS, radio_to_weight
)


def optimization_boosted(budget_rho, nb_modalite, poids):
    from functools import reduce
    import operator
    import pandas as pd
    import numpy as np
    from itertools import product
    from collections import defaultdict

    total = sum(poids.values())
    poids = {k: v / total for k, v in poids.items()}
    poids_set = {
        frozenset() if k == 'Total' else frozenset([k]) if isinstance(k, str) else frozenset(k): v
        for k, v in poids.items()
    }
    liste_request = [s for s in poids_set.keys()]

    # Trouver les maximaux (non inclus dans d'autres strictement plus grands)
    maximaux = [
        req for req in liste_request
        if not any((req < other) for other in liste_request)
    ]

    non_maximaux = [req for req in liste_request if req not in maximaux]
    liste_request = (
        sorted(maximaux, key=lambda x: -len(x)) +
        sorted(non_maximaux, key=lambda x: -len(x))
    )

    # Calcul de p
    def produit_modalites(fset):
        if not fset:
            return 1
        return reduce(operator.mul, (nb_modalite[v] for v in fset), 1)

    p = sum(produit_modalites(fset) for fset in maximaux)
    n = sum(produit_modalites(fset) for fset in liste_request)

    # Affichage
    print("Tous les frozensets :", liste_request)
    print("Requêtes maximales :", maximaux)
    print("Nombre de colonnes de X: p =", p)
    print("Nombre de lignes de X: n =", n)

    # 6. Génération des noms des bêtas
    beta_names = [f"beta_{i+1}" for i in range(p)]

    # 7. Construction des DataFrames pour chaque requête maximale
    df_par_requete = {}
    compteur = 0

    for req in maximaux:
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

    # 8.1 Première partie : identité pour les lignes correspondant aux β
    for i in range(p):
        X[i, i] = 1

    # 8.2 Reste : combinaisons/group_by
    ligne_courante = p  # on commence après les p premières lignes
    for req in liste_request:
        if req in maximaux:
            continue  # déjà traité via identité

        # Chercher un maximal qui contient la requête
        for max_req in maximaux:
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

    # Association des variables aux requêtes maximales
    var_to_requetes = defaultdict(list)
    for req in maximaux:
        for var in req:
            var_to_requetes[var].append(req)

    # --- Contraintes liées aux variables partagées dans plusieurs requêtes maximales ---
    for var, requetes in var_to_requetes.items():
        if len(requetes) <= 1:
            continue

        ref_req = requetes[0]
        df_ref = df_par_requete[ref_req]
        grouped_ref = df_ref.groupby(var)['value'].apply(list)

        for other_req in requetes[1:]:
            df_other = df_par_requete[other_req]
            grouped_other = df_other.groupby(var)['value'].apply(list)

            modalities = list(grouped_ref.index)
            # Enlever une modalité arbitraire (ex: la dernière) pour éviter redondance
            if len(modalities) > 1:
                modalities = modalities[:-1]

            for modality in modalities:
                betas_ref = grouped_ref[modality]
                betas_other = grouped_other[modality]

                row = np.zeros(len(beta_names))
                for b in betas_ref:
                    row[beta_names.index(b)] = 1
                for b in betas_other:
                    row[beta_names.index(b)] -= 1
                R_rows.append(row)

    # --- Contraintes liées au total (requête vide) : égalité entre les totaux de chaque requête maximale ---
    # On suppose que df_par_requete[frozenset()] correspond à la requête vide
    total_betas_by_req = {
        req: df_par_requete[req]["value"].tolist()
        for req in maximaux
    }

    max_reqs = list(total_betas_by_req.keys())
    ref_req = max_reqs[0]

    for other_req in max_reqs[1:]:
        row = np.zeros(len(beta_names))
        for b in total_betas_by_req[ref_req]:
            row[beta_names.index(b)] = 1
        for b in total_betas_by_req[other_req]:
            row[beta_names.index(b)] -= 1
        R_rows.append(row)

    # --- Construction finale de la matrice R ---
    if len(R_rows) > 0:
        R = np.vstack(R_rows)
    else:
        R = np.empty((0, p))
    print("R.shape =", R.shape)
    print("R (matrice des contraintes):")
    print(R)

    import numpy as np
    import itertools

    # Hypothèse : X et R sont déjà définies
    # Dimensions
    n, p = X.shape

    dict_request = {fset: {"nb_cellule": produit_modalites(fset), "sigma2": 1/(2*budget_rho*poids_set[fset])} for fset in liste_request}
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

    # Facultatif : Variance de X beta (si tu veux l'effet de la contrainte sur les sorties)
    V_Xbeta_constrained = X @ V_beta_constrained @ X.T
    var_Xbeta_constrained = np.diag(V_Xbeta_constrained)

    var_atteint = {}
    index = 0
    # Création du mapping entre frozenset (clé dans dict_request) et la clé d'origine de poids
    fs_to_key = {
        frozenset() if k == 'Total' else frozenset([k]) if isinstance(k, str) else frozenset(k): k
        for k in poids
    }

    for fset in dict_request:
        nb = dict_request[fset]["nb_cellule"]
        if fset in fs_to_key:
            original_key = fs_to_key[fset]
            var_atteint[original_key] = var_Xbeta_constrained[index]
        index += nb
    return var_atteint


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


def update_context(CONTEXT_PARAM, budget, budget_comptage, budget_totaux, poids_requetes_comptage, poids_requetes_total, poids_requetes_moyenne, poids_requetes_quantile):

    budget_moyenne = budget * sum(poids_requetes_moyenne)
    budget_quantile = np.sqrt(8 * budget * sum(poids_requetes_quantile))

    if budget_comptage == 0:
        context_comptage = None
    else:
        context_comptage = dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(rho=budget_comptage),
            split_by_weights=poids_requetes_comptage
        )

    if budget_totaux == 0:
        context_total = None
    else:
        context_total = dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(rho=budget_totaux),
            split_by_weights=poids_requetes_total
        )

    if budget_moyenne == 0:
        context_moyenne = None
    else:
        context_moyenne = dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(rho=budget_moyenne),
            split_by_weights=poids_requetes_moyenne
        )
    if budget_quantile == 0:
        context_quantile = None
    else:
        context_quantile = dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(epsilon=budget_quantile),
            split_by_weights=poids_requetes_quantile
        )

    return context_comptage, context_total, context_moyenne, context_quantile


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
        frozenset() if k == 'Total' else frozenset([k]) if isinstance(k, str) else frozenset(k): v
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


def organiser_par_by(dico_requetes, dico_valeurs):
    result = defaultdict(dict)
    result_bis = defaultdict(dict)
    for req, params in dico_requetes.items():
        # clé selon existence et contenu de 'by'
        if 'by' not in params:
            key = 'Total'
        else:
            by = params['by']
            # gérer le cas où by est une string au lieu d'une liste
            if isinstance(by, str):
                key = by
            elif isinstance(by, list):
                if len(by) == 1:
                    key = by[0]
                else:
                    key = tuple(by)
            else:
                # cas imprévu, on met en tot par défaut
                key = 'Total'

        # ajouter la valeur correspondante si elle existe dans dico_valeurs
        if req in dico_valeurs:
            result[key] = dico_valeurs[req]
            result_bis[key] = req
    return dict(result), dict(result_bis)


def normalize_weights(request, input) -> dict:
    raw_weights = {
        key: radio_to_weight.get(float(getattr(input, key)()), 0)
        for key, requete in request.items()
    }
    total = sum(raw_weights.values())
    return {k: v / total for k, v in raw_weights.items()}


def load_data(path: str, storage_options):
    if path.startswith("s3://"):
        df = pl.read_parquet(path, storage_options=storage_options)
    else:
        df = pl.read_parquet(path)

    if "geometry" in df.columns:
        df = df.drop("geometry")

    return df
