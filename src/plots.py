import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
sns.set_theme(style="whitegrid")


def create_barplot(df: pd.DataFrame, x_col: str, y_col: str, hoover: str = None, color: str = None) -> go.Figure:
    fig = go.Figure()

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
    """
    sorted_vals = np.sort(df['body_mass_g'].dropna())
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    quantile_val = np.quantile(sorted_vals, alpha)

    fig, ax = plt.subplots()
    ax.plot(sorted_vals, cdf, color="lightcoral", label="CDF empirique")
    ax.plot([quantile_val, quantile_val], [0, alpha], color='black', linestyle='--', linewidth=2)
    ax.plot([sorted_vals[0], quantile_val], [alpha, alpha], color='black', linestyle='--', linewidth=2)
    ax.text(quantile_val, alpha + 0.05, rf"$q_{{{alpha:.2f}}} = {quantile_val:.0f}$", 
            color='black', ha='right', fontsize=11)
    ax.set_xlabel("Valeur")
    ax.set_ylabel("Probabilité cumulative")
    ax.grid(True)
    return fig


def create_score_plot(data: dict) -> plt.Figure:
    """
    Affiche un nuage de points des scores des candidats, 
    en distinguant les top 95% cumulés (rouge) des autres (bleu).
    """
    candidats = data["candidats"]
    scores = data["scores"]
    top95_cumul = data["top95_cumul"]

    # Construction du DataFrame
    df = pd.DataFrame({
        "Candidat": candidats,
        "Score": scores,
        "Top95": top95_cumul
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="Candidat", y="Score", hue="Top95", palette={True: "red", False: "blue"},
        ax=ax, s=100, alpha=0.7, legend="full"
    )

    ax.set_xlabel("Candidat")
    ax.set_ylabel("Score")
    ax.legend(title="Top 95% cumulés")
    ax.grid(True)
    return fig


def create_proba_plot(data: dict) -> plt.Figure:
    """
    Affiche un nuage de points des probabilités des candidats,
    en distinguant les top 95% cumulés (rouge) des autres (bleu).
    """
    candidats = data["candidats"]
    proba = data["proba"]
    top95_cumul = data["top95_cumul"]

    df = pd.DataFrame({
        "Candidat": candidats,
        "Proba": proba,
        "Top95": top95_cumul
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="Candidat", y="Proba", hue="Top95", palette={True: "red", False: "blue"},
        ax=ax, s=100, alpha=0.7, legend="full"
    )

    ax.set_xlabel("Candidat")
    ax.set_ylabel("Probabilité")
    ax.legend(title="Top 95% cumulés")
    ax.grid(True)
    return fig
