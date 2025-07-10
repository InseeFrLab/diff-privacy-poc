import os
from typing import Dict, List

# Stockage des options S3 (MinIO)
storage_options: Dict[str, str] = {
    "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
    "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    "aws_session_token": os.environ["AWS_SESSION_TOKEN"],
    "endpoint_url": "https://minio.lab.sspcloud.fr",
}

# Liste des régions françaises couvertes par les jeux de données
regions_france: List[str] = [
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

# Liste des noms de jeux de données disponibles
name_dataset: List[str] = [
    "Fidéli",
    "Filosofi",
    "Penguin"
]

# Association entre clés d'identification et intitulés de jeux de données
chemin_dataset = {
    "penguins": "Palmer Penguins",
    "s3://gferey/diffusion/synthetic-filo/METRO/households/households_METRO.parquet": "Foyers Métropole",
    "s3://gferey/diffusion/synthetic-filo/METRO/population/population_METRO.parquet": "Population Métropole",
    "s3://gferey/diffusion/synthetic-filo/974/households/households_974.parquet": "Foyers Réunion",
    "s3://gferey/diffusion/synthetic-filo/974/population/population_974.parquet": "Population Réunion"
}

# Table de correspondance pour pondération dans l’interface utilisateur
radio_to_weight: Dict[int, float] = {
    1: 1.0,
    2: 0.5,
    3: 0.25
}

# Contribution d’un individu dans une agrégation
contrib_individu: int = 1

# Borne supérieure autorisée pour la taille d’un dataset (en nombre d’individu)
borne_max_taille_dataset: int = 1  # Peut être relevé à 70_000_000 si besoin
