from src.fonctions import manual_quantile_score
from scipy.stats import gumbel_r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.colors as pc


def create_scatterplot(df, x_col, y_col, size_col=False):
    fig = go.Figure()

    if df.empty:
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0], mode="text",
                text=["Aucune donnée"],
                textposition="middle center",
                showlegend=False
            )
        )
        return fig

    # Gestion des couleurs par requête
    unique_keys = df["requête"].unique()
    color_palette = pc.qualitative.Plotly
    color_map = {k: color_palette[i % len(color_palette)] for i, k in enumerate(unique_keys)}

    # Sécurisation des tailles de points
    if size_col:
        try:
            marker_size = pd.to_numeric(df[size_col], errors="coerce")
            if marker_size.isna().all():
                marker_size = 10
                size_col = False
        except Exception:
            marker_size = 10
            size_col = False
    else:
        marker_size = 10

    # Construction du scatter plot
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=dict(
                size=marker_size,
                sizemode="area",
                sizeref=2. * marker_size.max() / (40.**2) if size_col else 1,
                sizemin=5,
                color=[color_map[k] for k in df["requête"]],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=df["requête"],
            hovertemplate=(
                f"<b>%{{text}}</b>"
                + (f"<br>{x_col}: %{{x:.2f}}" if pd.api.types.is_numeric_dtype(df[x_col]) else f"<br>{x_col}: %{{x}}")
                + (f"<br>{y_col}: %{{y:.2f}}" if pd.api.types.is_numeric_dtype(df[y_col]) else f"<br>{y_col}: %{{y}}")
                + (f"<br>{size_col}: %{{marker.size:.2f}}" if size_col and size_col != x_col and size_col != y_col else "")
                + "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        plot_bgcolor='white',
        margin=dict(t=40, r=30, l=60, b=60),
        xaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='black', linewidth=1, mirror=True),
        yaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='black', linewidth=1, mirror=True),
    )

    return fig


def create_barplot(df, x_col, y_col, hoover=None):
    fig = go.Figure()

    if not df.empty:
        # Déterminer les couleurs en fonction du contenu de `hoover`
        if hoover is not None and hoover in df.columns:
            hover_texts = []
            colors = []
            for cross, val in zip(df[hoover], df[y_col]):
                # Déterminer le texte
                if isinstance(val, (int, float)):
                    val_text = f"{val:.2f}"
                else:
                    val_text = str(val)
                hover_texts.append(f"{cross}<br>{y_col}: {val_text}")

                # Déterminer la couleur
                if cross == "Total":
                    colors.append("darkgreen")
                elif isinstance(cross, str):
                    colors.append("steelblue")
                elif isinstance(cross, tuple) and len(cross) == 2:
                    colors.append("darkorange")
                elif isinstance(cross, tuple) and len(cross) == 3:
                    colors.append("firebrick")
                else:
                    colors.append("gray")  # fallback par sécurité
        else:
            hover_texts = [
                f"{y_col}: {val:.2f}" if isinstance(val, (int, float)) else f"{y_col}: {val}"
                for val in df[y_col]
            ]
            colors = ["steelblue"] * len(df)

        bar_args = dict(
            x=df[y_col],  # valeurs numériques
            y=df[x_col],  # catégories
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=1, color="black"),
                opacity=0.85,
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            textposition="auto",
            textfont=dict(color="white", size=18)
        )

        fig.add_trace(go.Bar(**bar_args))
    else:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
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


def create_histo_plot(df, quantile_alpha):
    """
    Affiche un histogramme de df['body_mass_g'] avec une ligne verticale
    au quantile `quantile_alpha`.
    """
    quantile_val = np.quantile(df['body_mass_g'].dropna(), quantile_alpha)
    fig, ax = plt.subplots()
    sns.histplot(df['body_mass_g'], stat="percent", bins=40, color='lightcoral', ax=ax)
    ax.axvline(quantile_val, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel("Valeur")
    ax.set_ylabel("Pourcentage")
    ax.grid(True)
    return fig


def create_fc_emp_plot(df, alpha):
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


def create_score_plot(df, alpha, epsilon, cmin, cmax, cstep):
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
    seuil = np.quantile(proba, 0.95)

    trace_red = go.Scatter(
        x=[c for c, p in zip(candidats, proba) if p >= seuil],
        y=[p for p in proba if p >= seuil],
        mode='markers', marker=dict(color='red', size=10), name='≥ 95% des proba'
    )
    trace_blue = go.Scatter(
        x=[c for c, p in zip(candidats, proba) if p < seuil],
        y=[p for p in proba if p < seuil],
        mode='markers', marker=dict(color='blue', size=10), name='< 95% des proba'
    )
    layout = go.Layout(
        xaxis=dict(title="Candidat", showgrid=True),
        yaxis=dict(title="Probabilité", showgrid=True),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=0.5, y=1.2, xanchor='center', yanchor='top', orientation='h'),
        margin=dict(t=50, r=30, l=60, b=60),
    )
    return go.Figure(data=[trace_red, trace_blue], layout=layout)
