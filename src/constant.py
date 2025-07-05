import os

storage_options = {
    "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
    "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    "aws_session_token": os.environ["AWS_SESSION_TOKEN"],
    "endpoint_url": "https://minio.lab.sspcloud.fr",
}

regions_france = [
    "France entière",
    "Auvergne-Rhône-Alpes",
    "Bourgogne-Franche-Comté",
    "Bretagne",
    "Centre-Val de Loire",
    "Corse",
    "Grand Est",
    "Hauts-de-France",
    "Île-de-France",
    "Normandie",
    "Nouvelle-Aquitaine",
    "Occitanie",
    "Pays de la Loire",
    "Provence-Alpes-Côte d’Azur",
    "Guadeloupe",
    "Martinique",
    "Guyane",
    "La Réunion",
    "Mayotte"
]

name_dataset = [
    "Fidéli",
    "Filosofi",
    "Penguin"
]

chemin_dataset = {
    "penguins": "Palmer Penguins",
    "s3://gferey/diffusion/synthetic-filo/METRO/households/households_METRO.parquet": "Foyers Métropole",
    "s3://gferey/diffusion/synthetic-filo/METRO/population/population_METRO.parquet": "Population Métropole",
    "s3://gferey/diffusion/synthetic-filo/974/households/households_974.parquet": "Foyers Réunion",
    "s3://gferey/diffusion/synthetic-filo/974/population/population_974.parquet": "Population Réunion"
}

radio_to_weight = {1: 1, 2: 0.5, 3: 0.25}

contrib_individu = 1

borne_max_taille_dataset = 70_000_000
