from shiny import ui


def page_resultat_dp():
    return ui.nav_panel(
        "Résultat DP",
        ui.panel_well(
            ui.h4("Résultat des requêtes DP"),
            ui.br(),
            ui.div("Voici les résultats des requêtes différentes privées."),
            ui.download_button("download_xlsx", "💾 Télécharger les résultats (XLSX)", class_="btn-outline-primary"),
            ui.br(),
            ui.output_ui("req_dp_display")
        )
    )
