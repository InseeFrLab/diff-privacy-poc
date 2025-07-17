import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
sns.set_theme(style="whitegrid")


def get_custom_color(val_color):
    if val_color == "Aucun":
        return "darkgreen"
    if isinstance(val_color, str):
        return "steelblue"
    if isinstance(val_color, tuple):
        return {2: "darkorange", 3: "firebrick"}.get(len(val_color), "gray")
    return "gray"


def create_barplot(df: pd.DataFrame, x_col: str, y_col: str, text: str, color: str) -> go.Figure:
    """
    Affiche un diagramme en barre horizontale, accompagnée d'un code couleur et de texte sur
    les barres si la place le permet.

    Args:
        df (pd.DataFrame): Données contenant les variables 'x_col', 'y_col', 'text' et 'color'.
        x_col (str): Nom de la colonne en axe des x (clé de la requête).
        y_col (str): Nom de la colonne en axe des y (mesure de la qualité de l'estimation).
        text (str): Nom de la colonne contenant le texte affiché sur les barres.
        color (str): Nome de la colonne pour la couleur (basé sur le nombre de variables groupées)

    Returns:
        plt.Figure: Diagramme en barre.
    """
    fig = go.Figure()

    displayed_texts = []
    hover_texts = []
    colors = []

    for _, row in df.iterrows():
        # Texte affiché sur la barre : court
        val = row[y_col]
        val_text = f"{val:.1f}" if isinstance(val, (int, float)) else str(val)
        text_displayed = f"{row[text]}<br>{y_col}: {val_text}"
        displayed_texts.append(f"{text_displayed}")

        # Texte survolé (hover) : toutes les colonnes
        hover_info = "<br>".join(f"{col}: {row[col]}" for col in df.columns if col != "label")
        hover_texts.append(f"{hover_info}")

        # Couleur personnalisée
        colors.append(get_custom_color(row[color]))

    fig.add_trace(go.Bar(
        x=df[y_col],
        y=df[x_col],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=1, color="black"),
            opacity=0.85,
        ),
        text=displayed_texts,  # Texte affiché sur les barres
        hovertext=hover_texts,  # Toutes les infos en hover
        hovertemplate="%{hovertext}<extra></extra>",
        textposition="auto",
        textfont=dict(color="white", size=18),
    ))

    fig.update_layout(
        xaxis_title=y_col,
        yaxis_title=x_col,
        plot_bgcolor='white',
        margin=dict(t=40, r=30, l=60, b=60),
        xaxis=dict(
            showgrid=True, gridcolor='lightgray', linecolor='lightgray', linewidth=1, mirror=True),
        yaxis=dict(
            showgrid=False, gridcolor='lightgray', linecolor='lightgray', linewidth=1, mirror=True),
    )

    return fig


def create_histo_plot(df: pd.DataFrame, quantile_alpha: float) -> plt.Figure:
    """
    Affiche un histogramme des valeurs de la variable 'body_mass_g', accompagné
    d'une ligne verticale représentant un quantile donné.

    Args:
        df (pd.DataFrame): Données contenant la variable 'body_mass_g'.
        quantile_alpha (float): Ordre du quantile.

    Returns:
        plt.Figure: Histogramme.
    """
    quantile_val = np.quantile(df['body_mass_g'], quantile_alpha)
    fig, ax = plt.subplots()
    sns.histplot(df['body_mass_g'], stat="percent", bins=40, color='lightcoral', ax=ax)
    ax.axvline(quantile_val, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel("Masse des manchots en grammes")
    ax.set_ylabel("Pourcentage")
    ax.grid(True)
    return fig


def create_fc_emp_plot(df: pd.DataFrame, quantile_alpha: float) -> plt.Figure:
    """
    Affiche la fonction de répartition empirique (CDF) de la variable 'body_mass_g',
    avec une annotation visuelle du quantile à un niveau donné.

    Args:
        df (pd.DataFrame): Données contenant la variable 'body_mass_g'.
        quantile_alpha (float): Ordre du quantile.

    Returns:
        plt.Figure: Fonction de répartition empirique.
    """
    sorted_vals = np.sort(df['body_mass_g'])
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    quantile_val = np.quantile(sorted_vals, quantile_alpha)

    fig, ax = plt.subplots()
    ax.plot(sorted_vals, cdf, color="lightcoral", label="CDF empirique")
    ax.plot(
        [quantile_val, quantile_val], [0, quantile_alpha],
        color='black', linestyle='--'
    )
    ax.plot(
        [sorted_vals[0], quantile_val], [quantile_alpha, quantile_alpha],
        color='black', linestyle='--'
    )
    ax.text(
        quantile_val, quantile_alpha + 0.05, rf"$q_{{{quantile_alpha:.2f}}} = {quantile_val:.0f}$",
        color='black', ha='right', fontsize=11
    )
    ax.set_xlabel("Masse des manchots en grammes")
    ax.set_ylabel("Probabilité cumulée")
    ax.grid(True)
    return fig


def create_score_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Affiche un nuage de points des scores des valeurs candidates,
    en distinguant les plus probables d'être tirées jusqu'à atteintdre 95%
    cumulés (rouge) des autres (bleu).

    Args:
        df (pd.DataFrame): Données contenant les valeurs candidates et les scores

    Returns:
        plt.Figure: Nuage de points.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="Candidat", y="Score", hue="Top95", palette={True: "red", False: "blue"},
        ax=ax, s=100, alpha=0.7, legend="full"
    )
    ax.set_xlabel("Valeur candidate")
    ax.set_ylabel("Score")
    ax.legend(title="Top 95% cumulés")
    ax.grid(True)
    return fig


def create_proba_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Affiche un nuage de points des probabilités des candidats,
    en distinguant les plus probables d'être tirées jusqu'à atteintdre 95%
    cumulés (rouge) des autres (bleu).

    Args:
        df (pd.DataFrame): Données contenant les valeurs candidates et les probabilités de sélection

    Returns:
        plt.Figure: Nuage de points.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="Candidat", y="Probabilité", hue="Top95", palette={True: "red", False: "blue"},
        ax=ax, s=100, alpha=0.7, legend="full"
    )
    ax.set_xlabel("Valeur candidate")
    ax.set_ylabel("Probabilité")
    ax.legend(title="Top 95% cumulés")
    ax.grid(True)
    return fig
