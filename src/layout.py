from src.constant import (
    regions_france, dataset
)
from src.process_tools import (
    process_request
)
from src.fonctions import (
    ameliorer_comptage, ameliorer_total
)
from shiny import ui
from shinywidgets import output_widget


# Fonctions pour du layout ----------------------------------


def afficher_resultats(results_store, requetes, poids_estimateur, poids_estimateur_tot, lien_comptage_req, lien_total_req_dict):
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


# Page Données ----------------------------------

def page_donnees():
    return ui.nav_panel(
        "Données",
        ui.page_sidebar(
            sidebar_donnees(),
            layout_donnees()
        )
    )


def sidebar_donnees():
    return ui.sidebar(
        ui.input_select(
            "default_dataset",
            "Choisir un jeu de données prédéfini:",
            {
                "penguins": "Palmer Penguins",
                "s3://gferey/diffusion/synthetic-filo/METRO/households/households_METRO.parquet": "Foyers Métropole",
                "s3://gferey/diffusion/synthetic-filo/METRO/population/population_METRO.parquet": "Population Métropole",
                "s3://gferey/diffusion/synthetic-filo/974/households/households_974.parquet": "Foyers Réunion",
                "s3://gferey/diffusion/synthetic-filo/974/population/population_974.parquet": "Population Réunion"
            }
        ),
        ui.input_file("dataset_input", "Ou importer un fichier CSV ou Parquet", accept=[".csv", ".parquet"]),
        position='right',
        bg="#f8f8f8"
    )


def layout_donnees():
    return ui.navset_card_underline(
        ui.nav_panel(
            "Aperçu des données",
            ui.output_data_frame("data_view"),
        ),
        ui.nav_panel(
            "Résumé statistique",
            ui.output_data_frame("data_summary"),
        )
    )


# Page Mécanisme DP ----------------------------------

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
        ui.div(
            # Partie gauche : slider + résumé
            ui.div(
                ui.div(
                    ui.input_slider("scale_gauss", "Écart type du bruit :", min=1, max=100, value=10),
                    style="width: 400px;"
                ),
                ui.div(
                    ui.output_ui("interval_summary"),
                ),
                style="display: flex; flex-direction: column; gap: 20px;"
            ),

            # Partie gauche : deux tableaux côte à côte
            ui.div(
                ui.layout_column_wrap(
                    ui.card(
                        ui.card_header("Tableau de comptage non bruité"),
                        ui.output_data_frame("cross_table"),
                    ),
                    ui.card(
                        ui.card_header("Exemple après bruitage (sans post-traitement)"),
                        ui.output_data_frame("cross_table_dp"),
                    ),
                    width=1 / 2,
                ),
                style="flex: 1;"
            ),

            # Partie budget DP : propre et sobre
            ui.div(
                ui.output_ui("dp_budget_summary"),
                ui.input_slider(
                    "delta_slider",
                    "Exposant de δ",
                    min=-10,
                    max=-1,
                    value=-3,
                    step=1,
                    width="320px"
                ),
                style="margin-top: 30px; display: flex; flex-direction: column; gap: 16px;"
            ),
            style="display: flex; align-items: flex-start; gap: 50px;"
        ),
    )


def bloc_score_quantile():
    return ui.panel_well(
        ui.h4("Mécanisme DP : scorer des candidats et tirer le score minimal après ajout d'un bruit (Quantile)"),
        ui.br(),
        # Conteneur principal en colonnes
        ui.div(
            # Ligne 1
            ui.div(
                ui.layout_columns(
                    # Colonne 1 : texte explicatif
                    ui.div(
                        ui.HTML("""
                            <div style='margin-top:20px; padding:10px; background-color:#f9f9f9; border-radius:12px;
                                        font-family: "Raleway", "Garamond", sans-serif; font-size:16px; color:#333'>
                                <p style="margin-bottom:10px">
                                    <strong>Exemple d'application à la variable <em>body_mass_g</em> du dataset Penguins :</strong>
                                </p>
                                <p style="margin-left:10px">
                                    La fonction de score utilisée est :
                                    <br><br>
                                    <strong>score</strong>(x, c, &alpha;) =<br>
                                    − 10 000 &times; | ∑<sub>i=1</sub><sup>n</sup> 1<sub>{x<sub>i</sub> &lt; c}</sub> − &alpha; &times; (n − ∑<sub>i=1</sub><sup>n</sup> 1<sub>{x<sub>i</sub> = c}</sub>) |
                                </p>
                                <p style="margin-left:10px">
                                    où <strong>α</strong> est l’ordre du quantile, et <strong>c</strong> un candidat et <strong>x</strong> notre variable d'intérêt de taille <strong>n</strong>.
                                </p>
                            </div>
                            """),
                        style="padding: 10px;"
                    ),
                    # Colonne 2 : première carte
                    ui.card(
                        ui.card_header("Histogramme"),
                        ui.output_plot("histo_plot"),
                        full_screen=True,
                    ),
                    # Colonne 3 : deuxième carte
                    ui.card(
                        ui.card_header("Fonction de répartion empirique"),
                        ui.output_plot("fc_emp_plot"),
                        full_screen=True,
                    ),
                    col_widths=[3, 4, 5]
                ),
                style="margin-bottom: 40px;"
            ),

            # Ligne 2
            ui.div(
                ui.layout_columns(
                    # Colonne 1 : texte + sliders + équation
                    ui.div(
                        ui.HTML("<strong>Paramètres :</strong>"),
                        ui.input_slider("epsilon_slider", "Budget epsilon :", min=0.01, max=5, value=0.5, step=0.01),
                        ui.input_slider("alpha_slider", "Ordre du quantile :", min=0, max=1, value=0.5, step=0.01),
                        ui.p("Définir l’intervalle et le pas pour le candidat :"),
                        # Ligne horizontale pour les 3 champs
                        ui.div(
                            ui.input_slider("candidat_slider", "Intervalles des candidats", min=0, max=10000, value=[2500, 6500]),
                            ui.input_numeric("candidat_step", "Pas :", value=100),
                            style="display: flex; flex-direction: row; gap: 15px; align-items: flex-end;"
                        ),
                        style="padding: 10px; display: flex; flex-direction: column; gap: 20px;"
                    ),
                    # Colonne 2 : carte placeholder
                    ui.card(
                        ui.card_header("Score des candidats"),
                        output_widget("score_plot"),
                        full_screen=True,
                    ),
                    # Colonne 3 : carte placeholder
                    ui.card(
                        ui.card_header("Probabilité de sélection"),
                        output_widget("proba_plot"),
                        full_screen=True,
                    ),
                    col_widths=[3, 5, 4]
                )
            ),
            style="display: flex; flex-direction: column; gap: 40px;"
        )
    )


# Page Préparer ses requêtes ----------------------------------

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
            ui.column(3, ui.input_text("filtre", "Condition de filtrage:")),
            ui.column(3, ui.input_text("borne_min", "Borne min:"))
        ),
        ui.br(),
        ui.row(
            ui.column(3, ui.input_selectize("variable", "Variable:", choices={}, options={"plugins": ["clear_button"]})),
            ui.column(3, ui.input_selectize("group_by", "Regrouper par:", choices={}, multiple=True)),
            ui.column(3, ui.input_text("borne_max", "Borne max:"))
        ),
        ui.br(),
        ui.panel_conditional(
            "input.type_req == 'Quantile'",
            ui.row(
                ui.column(3, ui.input_numeric("alpha", "Ordre du quantile:", 0.5, min=0, max=1, step=0.01)),
                ui.column(3, ui.input_text("candidat", "Liste des candidats:"))
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


# Page Conception du budget DP ----------------------------------

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
        ui.input_selectize("dataset_name", "Nom du dataset:", choices=dataset, selected="Penguin", options={"create": True}),
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


# Page Résultat dp ----------------------------------

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


# Page Etat budget dataset ----------------------------------

def page_etat_budget_dataset():
    return ui.nav_panel(
        "Etat budget dataset",
        ui.card(
            ui.card_header("Aperçu des budgets dépensées"),
            ui.output_data_frame("data_budget_view")
        ),
        ui.output_ui("budget_display")
    )
