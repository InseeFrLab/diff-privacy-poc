import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd


def create_barplot(df: pd.DataFrame, x_col: str, y_col: str, hoover: str = None, color: str = None) -> go.Figure:
    fig = go.Figure()

    if not df.empty:
        displayed_texts = []
        hover_texts = []
        colors = []

        for i, row in df.iterrows():
            # Texte affiché sur la barre : court
            if hoover and hoover in df.columns:
                val = row[y_col]
                val_text = f"{val:.1f}" if isinstance(val, (int, float)) else str(val)
                text_displayed = f"{row[hoover]}<br>{y_col}: {val_text}"
            else:
                text_displayed = f"{y_col}: {row[y_col]:.2f}" if isinstance(row[y_col], (int, float)) else f"{y_col}: {row[y_col]}"

            displayed_texts.append(f"{text_displayed}")
            # Texte survolé (hover) : toutes les colonnes
            hover_info = "<br>".join(f"{col}: {row[col]}" for col in df.columns)
            hover_texts.append(f"{hover_info}")

            # Couleur personnalisée
            colval = row[color] if color and color in df.columns else None
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

        fig.add_trace(go.Bar(
            x=df[y_col],
            y=df[x_col],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=1, color="black"),
                opacity=0.85,
            ),
            text=displayed_texts,  # Texte affiché sur les barres (le même pour toutes si non liste)
            hovertext=hover_texts,  # Toutes les infos en hover
            hovertemplate="%{hovertext}<extra></extra>",
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


def create_score_plot(data: dict) -> go.Figure:
    candidats = data["candidats"]
    scores = data["scores"]
    top95_cumul = data["top95_cumul"]

    red_x, red_y, = [], []
    blue_x, blue_y = [], []
    for i, c in enumerate(candidats):
        if top95_cumul[i]:
            red_x.append(c)
            red_y.append(scores[i])
        else:
            blue_x.append(c)
            blue_y.append(scores[i])

    trace_bleu = go.Scatter(x=blue_x, y=blue_y, mode='markers',
        marker=dict(color='blue', size=10, opacity=0.7), name='Reste')
    trace_rouge = go.Scatter(x=red_x, y=red_y, mode='markers',
        marker=dict(color='red', size=10, opacity=0.7), name='Top 95% cumulés')

    layout = go.Layout(
        xaxis=dict(title="Candidat", showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title="Score", showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=0.5, y=1.2, xanchor='center', yanchor='top', orientation='h'),
        margin=dict(t=50, r=30, l=60, b=60),
    )
    return go.Figure(data=[trace_rouge, trace_bleu], layout=layout)


def create_proba_plot(data: dict) -> go.Figure:
    candidats = data["candidats"]
    proba = data["proba"]
    top95_cumul = data["top95_cumul"]

    red_x, red_y = [], []
    blue_x, blue_y = [], []
    for i, c in enumerate(candidats):
        if top95_cumul[i]:
            red_x.append(c)
            red_y.append(proba[i])
        else:
            blue_x.append(c)
            blue_y.append(proba[i])

    trace_red = go.Scatter(x=red_x, y=red_y, mode='markers',
        marker=dict(color='red', size=10, opacity=0.7), name='Top 95% cumulés')
    trace_blue = go.Scatter(x=blue_x, y=blue_y, mode='markers',
        marker=dict(color='blue', size=10, opacity=0.7), name='Reste')

    layout = go.Layout(
        xaxis=dict(title="Candidat", showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title="Probabilité", showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=0.5, y=1.2, xanchor='center', yanchor='top', orientation='h'),
        margin=dict(t=50, r=30, l=60, b=60),
    )
    return go.Figure(data=[trace_red, trace_blue], layout=layout)
