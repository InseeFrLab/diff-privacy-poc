from shiny import ui


def page_etat_budget_dataset():
    return ui.nav_panel(
        "Etat budget dataset",
        ui.card(
            ui.card_header("Aperçu des budgets dépensées"),
            ui.output_data_frame("data_budget_view")
        ),
        ui.output_ui("budget_display")
    )
