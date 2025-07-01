from shiny import ui
from src.constant import (
    chemin_dataset
)


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
            chemin_dataset,
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
            "Métadonnées",
            ui.output_ui("meta_data"),
        )
    )
