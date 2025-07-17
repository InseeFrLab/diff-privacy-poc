from shiny import ui


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

        # Conteneur principal : structure en colonne horizontale avec espacements
        ui.div(
            # === Partie de gauche : sliders et résumé ===
            ui.div(
                ui.input_slider(
                    "scale_gauss", "Écart type du bruit :",
                    min=1, max=100, value=10,
                    width="300px"
                ),
                ui.output_ui("interval_summary"),
                style="display: flex; flex-direction: column; gap: 20px;"
            ),

            # === Partie centrale : deux tableaux côte à côte ===
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Tableau de comptage non bruité"),
                    ui.output_data_frame("cross_table"),
                ),
                ui.card(
                    ui.card_header("Après bruitage"),
                    ui.output_data_frame("cross_table_dp"),
                ),
                width=1 / 2,
            ),

            # === Partie droite : budget DP ===
            ui.div(
                ui.input_slider(
                    "delta_slider",
                    ui.HTML("Exposant de \\( \\delta \\)"),
                    min=-10,
                    max=-1,
                    value=-3,
                    step=1,
                    width="300px"
                ),
                ui.output_ui("dp_budget_summary"),
                style="display: flex; flex-direction: column; gap: 16px;"
            ),

            # === Style global du container ===
            style="display: flex; flex-direction: row; align-items: flex-start; gap: 50px;"
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
                                <p style="margin-left:10px">
                                    \\[
                                    \\begin{array}{c}
                                    \\textbf{score}(x, c, \\alpha) = \\\\
                                    -10\\,000 \\times \\left| \\sum_{i=1}^{n} \\mathbf{1}_{\\{x_i < c\\}} -
                                    \\alpha \\times \\left(n - \\sum_{i=1}^{n} \\mathbf{1}_{\\{x_i = c\\}} \\right) \\right|
                                    \\end{array}
                                    \\]
                                </p>
                            </p>
                            <br><br>
                            <p style="margin-left:10px">
                                où \\( \\alpha \\) est l’ordre du quantile, \\( c \\) un candidat, et \\( x \\) notre variable d'intérêt
                                de taille \\( n \\).
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

                        # Ligne 1 : sliders epsilon et alpha côte à côte
                        ui.layout_columns(
                            ui.input_slider("epsilon_slider", "Budget epsilon :", min=0.01, max=5, value=0.5, step=0.01),
                            ui.input_slider("alpha_slider", "Ordre du quantile :", min=0, max=1, value=0.5, step=0.01),
                            col_widths=[6, 6]  # pour diviser la ligne en 2 colonnes égales
                        ),

                        # Ligne 2 : texte explicatif
                        ui.p("Définir le nombre de candidats susceptibles d'être tirés entre min et max de la variable :"),

                        # Ligne 3 : sliders min-max et nombre de candidats
                        ui.layout_columns(
                            ui.input_slider("min_max_slider", "Valeur min-max", min=0, max=10000, value=[3000, 6000]),
                            ui.input_slider("candidat_slider", "Nombre de candidats", min=1, max=1000, value=100),
                            col_widths=[6, 6]
                        ),
                        
                        style="padding: 10px; display: flex; flex-direction: column; gap: 20px;"
                    ),
                    # Colonne 2 : carte placeholder
                    ui.card(
                        ui.card_header("Score des candidats"),
                        ui.output_plot("score_plot"),
                        full_screen=True,
                    ),
                    # Colonne 3 : carte placeholder
                    ui.card(
                        ui.card_header("Probabilité de sélection"),
                        ui.output_plot("proba_plot"),
                        full_screen=True,
                    ),
                    col_widths=[3, 5, 4]
                )
            ),
            style="display: flex; flex-direction: column; gap: 40px;"
        )
    )
