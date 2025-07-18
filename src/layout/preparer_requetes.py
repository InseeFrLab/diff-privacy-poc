from shiny import ui


def page_preparer_requetes():
    return ui.nav_panel(
        "Préparer ses requêtes",
        bloc_ajout_requete(),
        ui.hr(),
        layout_suppression_requetes(),
        ui.hr(),
        bloc_requetes_actuelles()
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
            ui.h4("Import / Export"),
            ui.br(),
            ui.row(
                ui.column(6, ui.input_file("request_input", "📂 Importer un fichier JSON", accept=[".json"]))
            ),
            ui.row(
                ui.column(12,
                    ui.div(
                        ui.download_button("download_json", "💾 Télécharger les requêtes (JSON)", class_="btn-outline-primary"),
                        class_="d-flex justify-content-end"
                    )
                )
            )
        ),
        ui.panel_well(
            ui.h4("Affichage des requetes"),
            ui.br(),
            ui.row(
                ui.column(6, ui.input_selectize("affichage_req", "Requêtes à afficher:", 
                    choices=["TOUTES", "Comptage", "Total", "Moyenne", "Ratio", "Quantile"], selected="TOUTES", options={"allowEmptyOption": False}))
            )
        ),
        ui.panel_well(
            ui.h4("🗑️ Supprimer une requête"),
            ui.br(),
            ui.row(
                ui.column(6, ui.input_selectize("delete_req", "Requêtes à supprimer:", choices={}, multiple=True))
            ),
            ui.row(
                ui.column(12,
                    ui.div(
                        ui.input_action_button("delete_btn", "Supprimer", class_="btn btn-danger"),
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


def affichage_requete(requetes, dict_stockage):

    panels = []

    for key, req in requetes.items():
        # Colonne de gauche : paramètres

        param_card = ui.card(
            ui.card_header("Paramètres"),
            make_card_body(req)
        )

        # Colonne de droite : table / placeholder
        result_card = ui.card(
            ui.card_header("Résultats sans application de la DP"),
            ui.HTML(dict_stockage[key].to_html(
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


def affichage_bouton(type_req: str, variables: dict, choix_quantile: dict, selected_variable: str):

    contenu = []
    variables.pop("🔤 Qualitatives", None)

    label_variable = "Variable au numérateur:" if type_req == "Ratio" else "Variable:"
    contenu.append(ui.column(
        3,
        ui.input_selectize(
            "variable", label_variable,
            choices=variables, options={"plugins": ["clear_button"]},
            selected=selected_variable)
        )
    )

    if type_req == "Ratio":
        contenu.append(ui.column(
            3,
            ui.input_selectize(
                "variable_denominateur", "Variable au dénominateur:",
                choices=variables, options={"plugins": ["clear_button"]})
            )
        )

    if type_req == "Quantile":
        contenu.append(ui.column(
            3,
            ui.input_selectize(
                "alpha", "Choix des quantiles:",
                choices=choix_quantile, multiple=True)
            )
        )
        contenu.append(ui.column(
            3,
            ui.input_numeric(
                "nb_candidats", "Nombre de candidats:", 1000,
                min=5, max=1_000_000, step=5)
            )
        )

    return contenu
