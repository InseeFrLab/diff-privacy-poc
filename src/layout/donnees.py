from shiny import ui


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
            "Métadonnées",
            ui.output_ui("meta_data"),
        )
    )
