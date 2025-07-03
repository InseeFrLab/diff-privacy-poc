from src.process_tools import (
    process_request
)
from src.fonctions import (
    ameliorer_comptage, ameliorer_total
)
from shiny import ui


# Fonctions pour du layout ----------------------------------


def afficher_resultats(results_store, requetes, query):
    current_results = results_store()
    panels = []
    final_results = {}
    for key, req in requetes.items():
        df_result = current_results[key]

        # 👉 Ici tu peux effectuer un traitement global sur plusieurs df si nécessaire
        if req.get("type") == "Comptage":
            df_result = ameliorer_comptage(key, df_result, poids_estimateur, results_store(), lien_comptage_req)

        if req.get("type") == "Total":
            df_result = ameliorer_total(key, req, df_result, poids_estimateur_tot, results_store(), lien_total_req_dict)

        final_results[key] = df_result

        param_card = ui.card(
            ui.card_header("Paramètres"),
            make_card_body(req)
        )

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
            ui.accordion_panel(f"{key} — {req.get('type', '—')}", content_row)
        )

    results_store.set(final_results)

    return ui.accordion(*panels)


def make_radio_buttons(request, filter_type: list[str]):
    radio_buttons = []
    priorite = {"Comptage": "2", "Total": "2", "Moyenne": "1", "Quantile": "3"}
    for key, req in request.items():
        if req["type"] in filter_type:
            radio_buttons_id = key
            radio_buttons.append(
                ui.input_radio_buttons(radio_buttons_id, key,
                    {"1": 1, "2": 2, "3": 3}, selected=priorite[req["type"]]
                )
            )
    return radio_buttons


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

    return ui.accordion(*panels)
