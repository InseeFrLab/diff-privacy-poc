from shiny import ui, module
from src.constant import (
    regions_france, name_dataset
)
from htmltools import TagList, tags, HTMLDependency

bootstrap_icons_dep = HTMLDependency(
    name="bootstrap-icons",
    version="1.10.5",
    source={"href": "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font"},
    stylesheet=[{"href": "bootstrap-icons.css"}]
)


def page_conception_budget():
    return ui.nav_panel(
        "Conception du budget",
        ui.page_sidebar(
            sidebar_budget(),
            bloc_budget_ui("Comptage"),
            bloc_budget_ui("Total"),
            bloc_budget_ui("Moyenne"),
            bloc_budget_ui("Ratio"),
            bloc_budget_ui("Quantile")
        )
    )


@module.ui
def bloc_budget_ui():
    return ui.output_ui("bloc_budget")


def sidebar_budget():
    return ui.sidebar(
        ui.h3("Définition du budget"),
        ui.input_slider("budget_total", "Budget total (rho DP) :", min=0.01, max=1, value=0.1, step=0.01),
        ui.input_selectize("echelle_geo", "Echelle géographique de l'étude:", choices=regions_france, selected="France entière"),
        ui.input_selectize("dataset_name", "Nom du dataset:", choices=name_dataset, selected="Penguin", options={"create": True}),
        ui.input_action_button("valider_budget", "Valider le budget DP"),
        position="right",
        bg="#f8f8f8"
    )


def make_radio_buttons(request, filter_type: list[str], dict_results):
    radio_buttons = []
    priorite = {"Comptage": "2", "Total": "2", "Moyenne": "1", "Ratio": "1", "Quantile": "3"}

    for key, req in request.items():
        if req["type"] in filter_type:
            radio_buttons_id = key

            # Contenu du tableau en HTML
            resultat = dict_results[key]
            table_html = ui.HTML(f"""
                <style>
                .popover {{
                    max-width: 1800px !important;
                    width: auto !important;
                }}
                .popover-body {{
                    max-height: 1800px !important;
                    overflow: visible !important;
                }}
                </style>
                <div style='max-height: 250px; overflow-y: auto; max-width: 500px; margin-top: 10px;'>
                    {resultat.to_html(
                        classes="table table-striped table-hover table-sm text-center align-middle",
                        border=0,
                        index=False
                    )}
                </div>
            """)
            # Nom + bouton popover avec tableau HTML
            title_with_popover = TagList(
                tags.span(key),
                ui.popover(
                    ui.HTML("  <i class='bi bi-table'></i>"),     # élément déclencheur visuel
                    table_html,            # contenu du popover
                    title=f"{key}",
                    placement="right"
                ),
                bootstrap_icons_dep
            )

            # Le bouton radio enrichi
            radio_buttons.append(
                ui.input_radio_buttons(
                    radio_buttons_id,
                    label=title_with_popover,
                    choices={"1": 1, "2": 2, "3": 3},
                    selected=priorite[req["type"]],
                    inline=True
                )
            )

    return radio_buttons
