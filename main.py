from src.fonctions import (
    load_data, load_yaml_metadata,
    save_yaml_metadata_from_dataframe
)
from src.constant import (
    storage_options,
    chemin_dataset
)
import polars as pl
import seaborn as sns


for chemin, value in chemin_dataset.items():

    if chemin == "penguins":
        lf = pl.DataFrame(sns.load_dataset(chemin).dropna()).lazy()
    else:
        lf = load_data(chemin, storage_options).lazy()

    # Enregistrement
    save_yaml_metadata_from_dataframe(lf, dataset_name=value)

    # Lecture et affichage
    metadata = load_yaml_metadata(value)
    print(metadata)
