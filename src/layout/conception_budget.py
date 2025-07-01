from shiny import ui
from shinywidgets import output_widget
from src.constant import (
    regions_france, name_dataset
)


def page_conception_budget():
    return ui.nav_panel(
        "Conception du budget",
        ui.page_sidebar(
            sidebar_budget(),
            bloc_budget_comptage(),
            ui.hr(),
            bloc_budget_total(),
            ui.hr(),
            bloc_budget_moyenne(),
            ui.hr(),
            bloc_budget_quantile()
        )
    )


def sidebar_budget():
    return ui.sidebar(
        ui.h3("Définition du budget"),
        ui.input_numeric("budget_total", "Budget total :", 0.1, min=0, max=1, step=0.01),
        ui.input_selectize("echelle_geo", "Echelle géographique de l'étude:", choices=regions_france, selected="France entière"),
        ui.input_selectize("dataset_name", "Nom du dataset:", choices=name_dataset, selected="Penguin", options={"create": True}),
        ui.input_action_button("valider_budget", "Valider le budget DP"),
        position="right",
        bg="#f8f8f8"
    )


def bloc_budget_comptage():
    return ui.panel_well(
        ui.layout_columns(
            ui.card(
                ui.card_header("Répartition du budget pour les comptages"),
                ui.output_ui("radio_buttons_comptage"),
            ),
            ui.navset_card_underline(
                ui.nav_panel("Plot", ui.card(
                    output_widget("plot_comptage"),
                    full_screen=True
                )),
                ui.nav_panel("Table", ui.card(
                    ui.output_data_frame("table_comptage"),
                    full_screen=True
                )),
                ui.nav_panel("Graphe", ui.card(
                    ui.output_plot("graphe_plot"),
                    full_screen=True
                ))
            ),
            col_widths=[4, 8]
        )
    )


def bloc_budget_total():
    return ui.panel_well(
        ui.layout_columns(
            ui.card(
                ui.card_header("Répartition du budget pour les totaux"),
                ui.output_ui("radio_buttons_total"),
            ),
            ui.navset_card_underline(
                ui.nav_panel("Plot", ui.card(
                    output_widget("plot_total"),
                    full_screen=True
                )),
                ui.nav_panel("Table", ui.card(
                    ui.output_data_frame("table_total"),
                    full_screen=True
                ))
            ),
            col_widths=[4, 8]
        )
    )


def bloc_budget_moyenne():
    return ui.panel_well(
        ui.layout_columns(
            ui.card(
                ui.card_header("Répartition du budget pour les moyennes"),
                ui.output_ui("radio_buttons_moyenne"),
            ),
            ui.navset_card_underline(
                ui.nav_panel("Plot", ui.card(
                    output_widget("plot_moyenne"),
                    full_screen=True
                )),
                ui.nav_panel("Table", ui.card(
                    ui.output_data_frame("table_moyenne"),
                    full_screen=True
                ))
            ),
            col_widths=[4, 8]
        )
    )


def bloc_budget_quantile():
    return ui.panel_well(
        ui.layout_columns(
            ui.card(
                ui.card_header("Répartition du budget pour les quantiles"),
                ui.output_ui("radio_buttons_quantile"),
            ),
            ui.navset_card_underline(
                ui.nav_panel("Plot", ui.card(
                    output_widget("plot_quantile"),
                    full_screen=True
                )),
                ui.nav_panel("Table", ui.card(
                    ui.output_data_frame("table_quantile"),
                    full_screen=True
                ))
            ),
            col_widths=[4, 8]
        )
    )
