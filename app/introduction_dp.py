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

        ui.layout_column_wrap(
            ui.card(
                ui.card_header("Tableau de comptage non bruité"),
                ui.output_data_frame("cross_table"),
            ),
            ui.card(
                ui.card_header("Après bruitage"),
                ui.output_data_frame("cross_table_dp"),
            ),
            width=1/2,
        ),
        ui.br(),

        ui.layout_column_wrap(
            ui.input_slider(
                "scale_gauss", "Écart type du bruit :",
                min=1, max=100, value=10
            ),
            ui.output_ui("interval_summary"),
            ui.input_slider(
                "delta_slider",
                ui.HTML("Exposant de \\( \\delta \\)"),
                min=-10,
                max=-1,
                value=-3,
                step=1
            ),
            ui.output_ui("dp_budget_summary"),
            width=1/4
        ),
    )


def bloc_score_quantile():
    return ui.panel_well(
        ui.h4("Mécanisme DP : scorer des candidats et tirer le score minimal après ajout d'un bruit (Quantile)"),
        ui.br(),

        # Ligne 1 : texte explicatif + 2 graphes
        ui.layout_columns(
            # Texte explicatif
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
                            10\\,000 \\times \\left| \\sum_{i=1}^{n} \\mathbf{1}_{\\{x_i < c\\}} -
                            \\alpha \\times \\left(n - \\sum_{i=1}^{n} \\mathbf{1}_{\\{x_i = c\\}} \\right) \\right|
                            \\end{array}
                            \\]
                        </p>
                    </p>
                    <br><br>
                    <p style="margin-left:10px">
                        où \\( \\alpha \\) est l'ordre du quantile, \\( c \\) un candidat, et \\( x \\) notre variable d'intérêt
                        de taille \\( n \\).
                    </p>
                </div>
            """),

            ui.card(
                ui.card_header("Histogramme"),
                ui.output_plot("histo_plot"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Fonction de répartion empirique"),
                ui.output_plot("fc_emp_plot"),
                full_screen=True,
            ),
            col_widths=[3, 4, 5]
        ),

        # Ligne 2 : paramètres et résultats
        ui.layout_columns(
            # Colonne paramètres (texte + sliders)
            ui.column(
                12,  # correspond à la largeur prévue initialement
                ui.HTML("<strong>Paramètres :</strong>"),
                ui.br(),
                ui.br(),

                # Sliders epsilon / alpha
                ui.layout_columns(
                    ui.input_slider(
                        "epsilon_slider", "Budget epsilon :", min=0.01,
                        max=1, value=0.5, step=0.01
                    ),
                    ui.input_slider(
                        "alpha_slider", "Ordre du quantile :", min=0,
                        max=1, value=0.5, step=0.01
                    ),
                    col_widths=[6, 6]
                ),
                ui.br(),

                # Texte explicatif
                ui.p("Définir les candidats susceptibles d'être tirés entre min et max de la variable :"),
                ui.br(),

                # Sliders min-max / candidats
                ui.layout_columns(
                    ui.input_slider(
                        "min_max_slider", "Valeur min-max", min=0,
                        max=10000, value=[3000, 6000]
                    ),
                    ui.input_slider(
                        "candidat_slider", "Pas discrétisation :", min=1, max=500, value=100
                    ),
                    col_widths=[6, 6]
                ),

                style="padding: 10px;"
            ),

            ui.card(
                ui.card_header("Score des candidats"),
                ui.output_plot("score_plot"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Probabilité de sélection"),
                ui.output_plot("proba_plot"),
                full_screen=True,
            ),
            col_widths=[3, 5, 4],
        )
    )
