from src.fonctions import manual_quantile_score
from scipy.stats import gumbel_r
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import networkx as nx
from itertools import combinations
import pandas as pd


def create_barplot(df: pd.DataFrame, x_col: str, y_col: str, hoover: str = None, color: str = None) -> go.Figure:
    """
    Génère un histogramme horizontal avec Plotly, en personnalisant la couleur des barres
    et les infobulles en fonction des colonnes fournies.

    Paramètres
    ----------
    df : pd.DataFrame
        Le tableau de données à représenter.
    x_col : str
        Le nom de la colonne contenant les catégories (axe des ordonnées).
    y_col : str
        Le nom de la colonne contenant les valeurs numériques (axe des abscisses).
    hoover : str, optionnel
        Le nom de la colonne à afficher dans les infobulles (texte survolé).
    color : str, optionnel
        Le nom de la colonne ou une valeur utilisée pour définir la couleur des barres.

    Retourne
    --------
    fig : go.Figure
        Une figure Plotly contenant l'histogramme horizontal.
    """
    fig = go.Figure()

    if not df.empty:
        if hoover is not None and hoover in df.columns:
            hover_texts = []
            colors = []
            for cross, colval, val in zip(df[hoover], df[color], df[y_col]):
                val_text = f"{val:.1f}" if isinstance(val, (int, float)) else str(val)
                hover_texts.append(f"{cross}<br>{y_col}: {val_text}")

                # Définition des couleurs
                if colval == "Aucun":
                    colors.append("darkgreen")
                elif isinstance(colval, str):
                    colors.append("steelblue")
                elif isinstance(colval, tuple) and len(colval) == 2:
                    colors.append("darkorange")
                elif isinstance(colval, tuple) and len(colval) == 3:
                    colors.append("firebrick")
                else:
                    colors.append("gray")
        else:
            hover_texts = [
                f"{y_col}: {val:.2f}" if isinstance(val, (int, float)) else f"{y_col}: {val}"
                for val in df[y_col]
            ]
            colors = ["steelblue"] * len(df)

        fig.add_trace(go.Bar(
            x=df[y_col],
            y=df[x_col],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=1, color="black"),
                opacity=0.85,
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            textposition="auto",
            textfont=dict(color="white", size=18),
        ))
    else:
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                mode="text",
                text=["Aucune donnée"],
                textposition="middle center",
                showlegend=False
            )
        )

    fig.update_layout(
        xaxis_title=y_col,
        yaxis_title=x_col,
        plot_bgcolor='white',
        margin=dict(t=40, r=30, l=60, b=60),
        xaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='black', linewidth=1, mirror=True),
        yaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='black', linewidth=1, mirror=True),
    )

    return fig


def create_histo_plot(df: pd.DataFrame, quantile_alpha: float):
    """
    Affiche un histogramme des valeurs de la variable 'body_mass_g', accompagné
    d'une ligne verticale représentant un quantile donné.

    Paramètres
    ----------
    df : pd.DataFrame
        Le tableau de données contenant une colonne 'body_mass_g'.
    quantile_alpha : float
        Le niveau de quantile à afficher (par exemple, 0.5 pour la médiane).

    Retourne
    --------
    fig : matplotlib.figure.Figure
        La figure contenant l'histogramme avec la ligne de quantile.
    """

    quantile_val = np.quantile(df['body_mass_g'].dropna(), quantile_alpha)
    fig, ax = plt.subplots()
    sns.histplot(df['body_mass_g'], stat="percent", bins=40, color='lightcoral', ax=ax)
    ax.axvline(quantile_val, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel("Valeur")
    ax.set_ylabel("Pourcentage")
    ax.grid(True)
    return fig


def create_fc_emp_plot(df: pd.DataFrame, alpha: float):
    """
    Trace la fonction de répartition empirique (CDF) de la variable 'body_mass_g',
    avec une annotation visuelle du quantile à un niveau donné.

    Paramètres
    ----------
    df : pd.DataFrame
        Le tableau de données contenant une colonne 'body_mass_g'.
    alpha : float
        Le niveau du quantile à représenter sur la CDF (entre 0 et 1).

    Retourne
    --------
    fig : matplotlib.figure.Figure
        La figure contenant la fonction de répartition empirique avec une annotation du quantile.
    """
    sorted_df = np.sort(df['body_mass_g'])
    cdf = np.arange(1, len(df) + 1) / len(df)
    quantile_val = np.quantile(df['body_mass_g'].dropna(), alpha)
    fig, ax = plt.subplots()
    ax.plot(sorted_df, cdf, color="lightcoral")
    ax.plot([quantile_val, quantile_val], [0, alpha], color='black', linestyle='--', linewidth=2)
    ax.plot([min(df['body_mass_g']), quantile_val], [alpha, alpha], color='black', linestyle='--', linewidth=2)
    ax.text(quantile_val, alpha + 0.1, rf"$q_{{{alpha:.2f}}} = {quantile_val:.0f}$", color='black', ha='right', fontsize=11)
    ax.set_xlabel("Valeur")
    ax.set_ylabel("Probabilité cumulative")
    ax.grid(True)
    return fig


def create_score_plot(df: pd.DataFrame, alpha: float, epsilon: float, cmin: float, cmax: float, cstep: int) -> go.Figure:
    """
    Affiche les scores bruités de candidats quantiles selon un mécanisme de Gumbel 
    (basé sur la confidentialité différentielle), avec leurs intervalles de confiance 
    à 99 %, en distinguant les candidats qui chevauchent ou non celui du score minimum.

    Paramètres
    ----------
    df : pd.DataFrame
        Le tableau de données contenant une colonne 'body_mass_g'.
    alpha : float
        Le niveau de quantile cible (entre 0 et 1).
    epsilon : float
        Le paramètre de confidentialité différentielle (plus il est grand, moins il y a de bruit).
    cmin : float
        Valeur minimale parmi les candidats quantiles.
    cmax : float
        Valeur maximale parmi les candidats quantiles.
    cstep : int
        Nombre de pas (discrétisation) entre `cmin` et `cmax`.

    Retourne
    --------
    fig : go.Figure
        Une figure Plotly représentant les scores bruités des candidats avec
        leurs intervalles de confiance, en distinguant les chevauchements.
    """
    candidats = np.linspace(cmin, cmax, cstep + 1).tolist()
    scores, sensi = manual_quantile_score(df['body_mass_g'], candidats, alpha=alpha, et_si=True)
    low_q, high_q = gumbel_r.ppf([0.005, 0.995], loc=0, scale=2 * sensi / epsilon)
    lower = scores + low_q
    upper = scores + high_q
    min_idx = np.argmin(scores)
    min_lower, min_upper = lower[min_idx], upper[min_idx]

    rouges_x, rouges_y, rouges_err = [], [], []
    bleus_x, bleus_y, bleus_err = [], [], []
    for c, s, l, u in zip(candidats, scores, lower, upper):
        if not (u < min_lower or l > min_upper):
            rouges_x.append(c)
            rouges_y.append(s)
            rouges_err.append([s - l, u - s])
        else:
            bleus_x.append(c)
            bleus_y.append(s)
            bleus_err.append([s - l, u - s])

    trace_bleu = go.Scatter(x=bleus_x, y=bleus_y, mode='markers',
        marker=dict(color='blue', size=10, opacity=0.7), name='Non chevauchement')
    trace_rouge = go.Scatter(x=rouges_x, y=rouges_y, mode='markers',
        marker=dict(color='red', size=10, opacity=0.7), name='Chevauchement')
    trace_erreur = go.Scatter(x=candidats, y=scores, mode='markers',
        marker=dict(color='black', size=2), error_y=dict(
            type='data', symmetric=False,
            array=[u - s for s, u in zip(scores, upper)],
            arrayminus=[s - l for s, l in zip(scores, lower)],
            color='black', thickness=1, width=4),
        name='IC à 99% (Gumbel)')

    layout = go.Layout(
        xaxis=dict(title="Candidat", showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title="Score", showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=0.5, y=1.2, xanchor='center', yanchor='top', orientation='h'),
        margin=dict(t=50, r=30, l=60, b=60),
    )
    return go.Figure(data=[trace_erreur, trace_bleu, trace_rouge], layout=layout)


def create_proba_plot(df, alpha, epsilon, cmin, cmax, cstep):

    candidats = np.linspace(cmin, cmax, cstep + 1).tolist()
    scores, sensi = manual_quantile_score(df['body_mass_g'], candidats, alpha=alpha, et_si=True)
    proba_non_norm = np.exp(-epsilon * scores / (2 * sensi))
    proba = proba_non_norm / np.sum(proba_non_norm)

    # Tri décroissant des probabilités
    sorted_indices = np.argsort(proba)[::-1]
    sorted_proba = np.array(proba)[sorted_indices]

    # Sélection des indices jusqu'à ce que la somme atteigne 95%
    cumulative = np.cumsum(sorted_proba)
    top95_mask = cumulative <= 0.95
    if not np.all(top95_mask):  # Inclure le premier élément qui fait dépasser 95%
        top95_mask[np.argmax(cumulative > 0.95)] = True

    red_indices = sorted_indices[top95_mask]
    blue_indices = sorted_indices[~top95_mask]

    trace_red = go.Scatter(
        x=[candidats[i] for i in red_indices],
        y=[proba[i] for i in red_indices],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Top 95% cumulés'
    )
    trace_blue = go.Scatter(
        x=[candidats[i] for i in blue_indices],
        y=[proba[i] for i in blue_indices],
        mode='markers',
        marker=dict(color='blue', size=10),
        name='Autres'
    )

    layout = go.Layout(
        xaxis=dict(title="Candidat", showgrid=True),
        yaxis=dict(title="Probabilité", showgrid=True),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=0.5, y=1.2, xanchor='center', yanchor='top', orientation='h'),
        margin=dict(t=50, r=30, l=60, b=60),
    )
    return go.Figure(data=[trace_red, trace_blue], layout=layout)


def plot_subset_tree(liste_request, taille_noeuds=2000, keep_gris=False):
    dict_nb_modalite_variables = {}
    for fs in liste_request:
        for var in fs:
            dict_nb_modalite_variables[var] = 2  # hypothèse : 2 modalités par défaut

    def powerset_frozensets(iterable):
        return [frozenset(s) for r in range(len(iterable)+1) for s in combinations(iterable, r)]

    all_vars = list(dict_nb_modalite_variables.keys())
    all_nodes = powerset_frozensets(all_vars)

    full_graphe = {fs: [] for fs in all_nodes}
    for parent in all_nodes:
        for enfant in all_nodes:
            if len(enfant) == len(parent) + 1 and parent.issubset(enfant):
                full_graphe[parent].append(enfant)

    def compute_levels(graphe_nodes):
        return {node: len(node) for node in graphe_nodes}

    def manual_tree_layout(graphe):
        levels = compute_levels(graphe.nodes)
        pos = {}
        nodes_by_level = {}
        for node, lvl in levels.items():
            nodes_by_level.setdefault(lvl, []).append(node)
        for lvl, nodes in nodes_by_level.items():
            for i, node in enumerate(sorted(nodes, key=lambda x: sorted(x))):
                pos[node] = (i, -lvl)
        return pos

    G = nx.DiGraph()
    for parent, enfants in full_graphe.items():
        for enfant in enfants:
            G.add_edge(enfant, parent)

    noeuds_request = set(liste_request)
    feuilles = [
        n for n in noeuds_request
        if all(child not in noeuds_request for child in G.predecessors(n))
    ]

    # Couleurs initiales
    node_colors = []
    for n in G.nodes:
        if n in feuilles:
            node_colors.append("orange")
        elif n in noeuds_request:
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightgray")

    # Calculer les niveaux de tous les nœuds
    levels = compute_levels(G.nodes)

    if not keep_gris:

        niveau_max_feuilles = max(levels[n] for n in feuilles) - 1

        nodes_to_remove_gris = [n for n in G.nodes if n not in noeuds_request and levels[n] > niveau_max_feuilles]
        G.remove_nodes_from(nodes_to_remove_gris)

        nodes_to_remove = [n for n in G.nodes if n not in noeuds_request and G.in_degree(n) == 0]
        G.remove_nodes_from(nodes_to_remove)

    # Recalculer feuilles et couleurs après suppression
    noeuds_request = set(n for n in noeuds_request if n in G.nodes)
    feuilles = [
        n for n in noeuds_request
        if all(child not in noeuds_request for child in G.predecessors(n))
    ]

    # --- Trouver les nœuds intersection entre deux feuilles ---
    intersections = set()
    feuilles_list = list(feuilles)
    for i in range(len(feuilles_list)):
        for j in range(i+1, len(feuilles_list)):
            inter = feuilles_list[i].intersection(feuilles_list[j])
            # Ici, on accepte aussi l'intersection vide (frozenset())
            if (inter in G.nodes) and (inter is not None):
                intersections.add(inter)

    # Préparer les listes par catégorie
    noeuds_carre = intersections
    noeuds_feuilles = set(feuilles) - noeuds_carre
    noeuds_autres = set(G.nodes) - noeuds_feuilles - noeuds_carre

    # Couleurs
    color_feuilles = 'orange'
    # Couleurs pour les intersections (carrés) : même logique que les autres nœuds
    color_carre = []
    for n in noeuds_carre:
        if n in noeuds_request:
            color_carre.append("lightgreen")
        else:
            color_carre.append("lightgray")
    color_autres = []
    for n in noeuds_autres:
        if n in noeuds_request:
            color_autres.append("lightgreen")
        else:
            color_autres.append("lightgray")

    pos = manual_tree_layout(G)

    # feuilles est la liste des nœuds feuilles (orange)
    feuilles_set = set(feuilles)

    labels = {}
    G_inv = G.reverse(copy=False)
    for n in G.nodes:
        descendants = nx.descendants( G_inv, n)  # tous les descendants de n (parcours orienté)
        nb_feuilles_accessibles = len(descendants.intersection(feuilles_set))
        if nb_feuilles_accessibles > 0:
            labels[n] = f"{','.join(sorted(n)) if n else '∅'}\n({nb_feuilles_accessibles})"
        else:
            labels[n] = f"{','.join(sorted(n)) if n else '∅'}"

    plt.figure(figsize=(14, 10))

    # Dessiner les autres nœuds (ronds)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=noeuds_autres,
        node_color=color_autres,
        node_size=taille_noeuds,
        edgecolors="black",
        linewidths=1.5,
        node_shape='o'
    )

    # Dessiner les feuilles (ronds orange)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=noeuds_feuilles,
        node_color=color_feuilles,
        node_size=taille_noeuds,
        edgecolors="black",
        linewidths=1.5,
        node_shape='o'
    )

    # Dessiner les intersections (carrés cyan)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=noeuds_carre,
        node_color=color_carre,
        node_size=taille_noeuds,
        edgecolors="black",
        linewidths=1.5,
        node_shape='s'
    )

    # Dessiner les arêtes et labels
    nx.draw_networkx_edges(
        G, pos
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    plt.axis("off")
    plt.show()
