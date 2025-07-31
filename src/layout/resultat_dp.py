from shiny import ui
from src.layout.preparer_requetes import make_card_body


def page_resultat_dp():
    return ui.nav_panel(
        "Résultat DP",
        ui.panel_well(
            ui.h4("Résultat des requêtes DP"),
            ui.br(),
            ui.output_ui("req_dp_display")
        )
    )


def afficher_resultats(results_store, requetes):
    panels = []
    final_results = results_store()

    for key, req in requetes.items():

        param_card = ui.card(
            ui.card_header("Paramètres"),
            make_card_body(req)
        )

        df_result = final_results[key]

        result_card = ui.card(
            ui.card_header("Résultats après application de la DP"),
            ui.HTML(df_result.to_html(
                classes="table table-striped table-hover table-sm text-center align-middle",
                border=0,
                index=False
            )),
            height="300px",
            fillable=False,
            full_screen=True
        )

        content_row = ui.row(
            ui.column(4, param_card),
            ui.column(8, result_card)
        )

        panels.append(
            ui.accordion_panel(f"{key} — {req.__class__.__name__}", content_row, open=True)
        )

    results_store.set(final_results)

    return ui.TagList(
        ui.div(
            "📤 Exporter vos résultats respectant la confidentialité différentielle :",
            class_="mb-2"
        ),
        ui.download_button(
            "download_xlsx",
            "💾 Télécharger les résultats (XLSX)",
            class_="btn-outline-primary mb-4"
        ),
        ui.accordion(*panels, open=True)
    )
