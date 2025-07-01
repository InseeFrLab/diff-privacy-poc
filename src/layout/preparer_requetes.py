from shiny import ui


def page_preparer_requetes():
    return ui.nav_panel(
        "Préparer ses requêtes",
        ui.page_sidebar(
            sidebar_requetes(),
            bloc_ajout_requete(),
            ui.hr(),
            layout_suppression_requetes(),
            ui.hr(),
            bloc_requetes_actuelles()
        )
    )


def sidebar_requetes():
    return ui.sidebar(
        ui.input_file("request_input", "📂 Importer un fichier JSON", accept=[".json"]),
        ui.br(),
        ui.download_button("download_json", "💾 Télécharger les requêtes (JSON)", class_="btn-outline-primary"),
        position="right",
        bg="#f8f8f8"
    )


def bloc_ajout_requete():
    return ui.panel_well(
        ui.h4("➕ Ajouter une requête"),
        ui.br(),
        ui.row(
            ui.column(3, ui.input_selectize("type_req", "Type de requête:", choices=["Comptage", "Total", "Moyenne", "Quantile"], selected="Comptage")),
            ui.column(3, ui.input_text("filtre", "Condition de filtrage:"))
        ),
        ui.br(),
        ui.row(
            ui.column(3, ui.input_selectize("variable", "Variable:", choices={}, options={"plugins": ["clear_button"]})),
            ui.column(3, ui.input_selectize("group_by", "Regrouper par:", choices={}, multiple=True))
        ),
        ui.br(),
        ui.panel_conditional(
            "input.type_req == 'Quantile'",
            ui.row(
                ui.column(3, ui.input_numeric("alpha", "Ordre du quantile:", 0.5, min=0, max=1, step=0.01)),
                ui.column(3, ui.input_text("nb_candidat", "Nombre de candidats:"))
            ),
        ),
        ui.br(),
        ui.row(
            ui.column(12,
                ui.div(
                    ui.input_action_button("add_req", "➕ Ajouter la requête"),
                    class_="d-flex justify-content-end"
                )
            )
        )
    )


def layout_suppression_requetes():
    return ui.layout_columns(
        ui.panel_well(
            ui.h4("🗑️ Supprimer une requête"),
            ui.br(),
            ui.row(
                ui.column(6, ui.input_selectize("delete_req", "Requêtes à supprimer:", choices={}, multiple=True))
            ),
            ui.row(
                ui.column(12,
                    ui.div(
                        ui.input_action_button("delete_btn", "Supprimer"),
                        class_="d-flex justify-content-end"
                    )
                )
            )
        ),
        ui.panel_well(
            ui.h4("🗑️ Supprimer toutes les requêtes"),
            ui.br(),
            ui.row(ui.column(12, ui.div(style="height: 85px"))),
            ui.row(
                ui.column(12,
                    ui.div(
                        ui.input_action_button("delete_all_btn", "Supprimer TOUT", class_="btn btn-danger"),
                        class_="d-flex justify-content-end"
                    )
                )
            )
        )
    )


def bloc_requetes_actuelles():
    return ui.panel_well(
        ui.h4("📋 Requêtes actuelles"),
        ui.br(),
        ui.output_ui("req_display")
    )
