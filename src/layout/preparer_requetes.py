from shiny import ui
from src.process_tools import (
    process_request
)


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

        # Ligne 1 : champs toujours visibles
        ui.row(
            ui.column(3, ui.input_selectize("type_req", "Type de requête:",
                                            choices=["Comptage", "Total", "Moyenne", "Ratio", "Quantile"],
                                            selected="Comptage",
                                            options={"allowEmptyOption": False})),
            ui.column(3, ui.input_selectize("group_by", "Regrouper par:", choices={}, multiple=True)),
            ui.column(3, ui.input_text("filtre", "Condition de filtrage:"))
        ),

        ui.br(),

        # Ligne 2 : affichage conditionnel dynamique
        ui.output_ui("ligne_conditionnelle"),

        ui.br(),

        # Ligne 3 : bouton à droite
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


def make_card_body(req):
    parts = []

    fields = [
        ("variable", "📌 Variable", lambda v: f"`{v}`"),
        ("bounds", "🎯 Bornes", lambda v: f"`[{v[0]}, {v[1]}]`" if isinstance(v, list) and len(v) == 2 else "—"),
        ("by", "🧷 Group by", lambda v: f"`{', '.join(v)}`"),
        ("filtre", "🧮 Filtre", lambda v: f"`{v}`"),
        ("alpha", "📈 Alpha", lambda v: f"`{v}`"),
    ]

    for key, label, formatter in fields:
        val = req.get(key)
        if val is not None and val != "" and val != []:
            parts.append(ui.p(f"{label} : {formatter(val)}"))

    return ui.card_body(*parts)


def affichage_requete(requetes, dataset):

    panels = []
    df = dataset.lazy()

    for key, req in requetes.items():
        # Colonne de gauche : paramètres
        resultat = process_request(df, req, use_bounds=False)

        if req.get("by") is not None:
            resultat = resultat.sort(by=req.get("by"))

        param_card = ui.card(
            ui.card_header("Paramètres"),
            make_card_body(req)
        )

        # Colonne de droite : table / placeholder
        result_card = ui.card(
            ui.card_header("Résultats sans application de la DP (à titre indicatif)"),
            ui.tags.style("""
                .table {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size: 0.9rem;
                    box-shadow: 0 0 10px rgba(0,0,0,0.05);
                    border-radius: 0.25rem;
                    border-collapse: collapse;
                    width: 100%;
                    border: 1px solid #dee2e6;
                }
                .table-hover tbody tr:hover {
                    background-color: #f1f1f1;
                }
                .table-striped tbody tr:nth-of-type(odd) {
                    background-color: #fafafa;
                }
                table.table thead th {
                    background-color: #f8f9fa !important;
                    font-weight: 700 !important;
                    border-left: 1px solid #dee2e6;
                    border-right: 1px solid #dee2e6;
                    border-bottom: 2px solid #dee2e6;
                    padding: 0.3rem 0.6rem;
                    vertical-align: middle !important;
                    text-align: center;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }
                tbody td {
                    padding: 0.3rem 0.6rem;
                    vertical-align: middle !important;
                    text-align: center;
                    border-left: 1px solid #dee2e6;
                    border-right: 1px solid #dee2e6;
                }
                thead th:first-child, tbody td:first-child {
                    border-left: none;
                }
                thead th:last-child, tbody td:last-child {
                    border-right: none;
                }
            """),
            ui.HTML(resultat.to_pandas().to_html(
                classes="table table-striped table-hover table-sm text-center align-middle",
                border=0,
                index=False
            )),
            height="300px",
            fillable=False,
            full_screen=True
        ),

        # Ligne avec deux colonnes côte à côte
        content_row = ui.row(
            ui.column(4, param_card),
            ui.column(8, result_card)
        )

        # Panneau d'accordéon contenant la ligne
        panels.append(
            ui.accordion_panel(f"{key} — {req.get('type', '—')}", content_row)
        )

    return ui.accordion(*panels, open=True)
