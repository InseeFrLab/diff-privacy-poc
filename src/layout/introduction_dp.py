from shiny import ui
from shinywidgets import output_widget


def page_introduction_dp():
    return ui.nav_panel(
        "Introduction DP",
        bloc_bruit_gaussien(),
        ui.hr(),
        bloc_score_quantile()
    )


def bloc_bruit_gaussien():
    return ui.panel_well(
        ui.h4("Mécanisme DP : ajout d'un bruit Gaussien centré (Comptage et Total)"),
        ui.br(),
        ui.div(
            # Partie gauche : slider + résumé
            ui.div(
                ui.div(
                    ui.input_slider("scale_gauss", "Écart type du bruit :", min=1, max=100, value=10),
                    style="width: 400px;"
                ),
                ui.div(
                    ui.output_ui("interval_summary"),
                ),
                style="display: flex; flex-direction: column; gap: 20px;"
            ),

            # Partie gauche : deux tableaux côte à côte
            ui.div(
                ui.layout_column_wrap(
                    ui.card(
                        ui.card_header("Tableau de comptage non bruité"),
                        ui.output_data_frame("cross_table"),
                    ),
                    ui.card(
                        ui.card_header("Exemple après bruitage (sans post-traitement)"),
                        ui.output_data_frame("cross_table_dp"),
                    ),
                    width=1 / 2,
                ),
                style="flex: 1;"
            ),

            # Partie budget DP : propre et sobre
            ui.div(
                ui.output_ui("dp_budget_summary"),
                ui.input_slider(
                    "delta_slider",
                    "Exposant de δ",
                    min=-10,
                    max=-1,
                    value=-3,
                    step=1,
                    width="320px"
                ),
                style="margin-top: 30px; display: flex; flex-direction: column; gap: 16px;"
            ),
            style="display: flex; align-items: flex-start; gap: 50px;"
        ),
    )


def bloc_score_quantile():
    return ui.panel_well(
        ui.h4("Mécanisme DP : scorer des candidats et tirer le score minimal après ajout d'un bruit (Quantile)"),
        ui.br(),
        # Conteneur principal en colonnes
        ui.div(
            # Ligne 1
            ui.div(
                ui.layout_columns(
                    # Colonne 1 : texte explicatif
                    ui.div(
                        ui.HTML("""
                            <div style='margin-top:20px; padding:10px; background-color:#f9f9f9; border-radius:12px;
                                        font-family: "Raleway", "Garamond", sans-serif; font-size:16px; color:#333'>
                                <p style="margin-bottom:10px">
                                    <strong>Exemple d'application à la variable <em>body_mass_g</em> du dataset Penguins :</strong>
                                </p>
                                <p style="margin-left:10px">
                                    La fonction de score utilisée est :
                                    <br><br>
                                    <strong>score</strong>(x, c, &alpha;) =<br>
                                    − 10 000 &times; | ∑<sub>i=1</sub><sup>n</sup> 1<sub>{x<sub>i</sub> &lt; c}</sub> − &alpha; &times; (n − ∑<sub>i=1</sub><sup>n</sup> 1<sub>{x<sub>i</sub> = c}</sub>) |
                                </p>
                                <p style="margin-left:10px">
                                    où <strong>α</strong> est l’ordre du quantile, et <strong>c</strong> un candidat et <strong>x</strong> notre variable d'intérêt de taille <strong>n</strong>.
                                </p>
                            </div>
                            """),
                        style="padding: 10px;"
                    ),
                    # Colonne 2 : première carte
                    ui.card(
                        ui.card_header("Histogramme"),
                        ui.output_plot("histo_plot"),
                        full_screen=True,
                    ),
                    # Colonne 3 : deuxième carte
                    ui.card(
                        ui.card_header("Fonction de répartion empirique"),
                        ui.output_plot("fc_emp_plot"),
                        full_screen=True,
                    ),
                    col_widths=[3, 4, 5]
                ),
                style="margin-bottom: 40px;"
            ),

            # Ligne 2
            ui.div(
                ui.layout_columns(
                    # Colonne 1 : texte + sliders + équation
                    ui.div(
                        ui.HTML("<strong>Paramètres :</strong>"),
                        ui.input_slider("epsilon_slider", "Budget epsilon :", min=0.01, max=5, value=0.5, step=0.01),
                        ui.input_slider("alpha_slider", "Ordre du quantile :", min=0, max=1, value=0.5, step=0.01),
                        ui.p("Définir l’intervalle et le pas pour le candidat :"),
                        # Ligne horizontale pour les 3 champs
                        ui.div(
                            ui.input_slider("candidat_slider", "Intervalles des candidats", min=0, max=10000, value=[2500, 6500]),
                            ui.input_numeric("candidat_step", "Pas :", value=100),
                            style="display: flex; flex-direction: row; gap: 15px; align-items: flex-end;"
                        ),
                        style="padding: 10px; display: flex; flex-direction: column; gap: 20px;"
                    ),
                    # Colonne 2 : carte placeholder
                    ui.card(
                        ui.card_header("Score des candidats"),
                        output_widget("score_plot"),
                        full_screen=True,
                    ),
                    # Colonne 3 : carte placeholder
                    ui.card(
                        ui.card_header("Probabilité de sélection"),
                        output_widget("proba_plot"),
                        full_screen=True,
                    ),
                    col_widths=[3, 5, 4]
                )
            ),
            style="display: flex; flex-direction: column; gap: 40px;"
        )
    )
