import re
import numpy as np
import polars as pl
from scipy.optimize import fsolve
import opendp.prelude as dp
from collections import defaultdict
from src.constant import (
    OPS, radio_to_weight
)


def update_context(CONTEXT_PARAM, budget_total, budget_comptage, poids_requetes_comptage, poids_requetes_moyenne_total, poids_requetes_quantile):

    budget_moyenne_total = budget_total * sum(poids_requetes_moyenne_total)
    budget_quantile = np.sqrt(8 * budget_total * sum(poids_requetes_quantile))

    if budget_comptage == 0:
        context_comptage = None
    else:
        context_comptage = dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(rho=budget_comptage),
            split_by_weights=poids_requetes_comptage
        )

    if budget_moyenne_total == 0:
        context_moyenne_total = None
    else:
        context_moyenne_total = dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(rho=budget_moyenne_total),
            split_by_weights=poids_requetes_moyenne_total
        )
    if budget_quantile == 0:
        context_quantile = None
    else:
        context_quantile = dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(epsilon=budget_quantile),
            split_by_weights=poids_requetes_quantile
        )

    return context_comptage, context_moyenne_total, context_quantile


# Fonction pour forcer une variable à 0 en modifiant A et b
def impose_zero(A, b, index):
    A[index - 1, :] = 0
    A[index - 1, index] = -1
    b[index - 1] = 0
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
                Q[i, j] = -1
            elif Ei.issubset(Ej):
                diff = Ej - Ei
                prod = 1
                for k in diff:
                    prod *= nb_modalite[k]
                Q[i, j] = -1 / prod

    P = np.zeros((N, 1))
    for i, subset in enumerate(subsets):
        P[i, 0] = 2 * poids_set.get(frozenset(subset), 0)

    b = np.zeros((N + 1, 1))
    b[-1, 0] = budget_rho

    A = np.zeros((N + 1, N + 1))
    A[:N, 0] = P.flatten()
    A[:N, 1:] = Q
    A[N, 1:] = 0.5

    x_sol = solve_projected(A.copy(), b.copy())
    print(x_sol)
    variance_req = {}
    variance_atteinte = {}
    poids_estimateur = {}
    if x_sol is not None:
        print(f"---------------------------------------------------------------------------------------------------")
        print(f"ρ optimal = {x_sol[0].item():.3f}")
        for i, ((nom, p), x) in enumerate(zip(poids.items(), x_sol[1:])):
            if x.item() != 0:
                var_estim = 1/(2*p*x_sol[0].item())
                variance_req[nom] = 1/x.item()
                print(f"Variance de la requête pour {nom} = {1/x.item():.2f} et variance d'estimation = {var_estim:.2f}")
            else:
                var_estim = -1/np.dot(Q[i], x_sol[1:].flatten())
                print(f"Pas de requête pour {nom} et de variance d'estimation = {var_estim:.2f}")

            variance_atteinte[nom] = var_estim
            poids_estimateur[nom] = {}
            for j, ((nom_2, _), x_2) in enumerate(zip(poids.items(), x_sol[1:])):
                poids_estim = -var_estim * Q[i][j]*x_sol[1+j].item()
                if poids_estim > 0:
                    poids_estimateur[nom][nom_2] = poids_estim
                    print(f"- Poids de l'estimation par {nom_2} = {poids_estim:.2f}")
    else:
        print("❌ Aucune solution admissible.")
    print(f"---------------------------------------------------------------------------------------------------")
    return variance_atteinte, variance_req, poids_estimateur


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
    if rho < 0:
        raise ValueError("rho must be positive")

    def equation(y, rho, delta):
        return y - (rho + 2 * np.sqrt(rho * np.log(1 / (delta * (1 + (y - rho) / (2 * rho))))))

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
