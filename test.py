import polars as pl
import yaml

df = pl.DataFrame({
    "age": [23, 45, None, 32],
    "name": ["Alice", "Bob", "Charlie", None],
    "score": [89.5, 92.3, 76.5, 88.0]
})

lf = df.lazy()


import polars as pl
import yaml

def generate_yaml_metadata_from_lazyframe(df: pl.Dataframe, output_path: str, dataset_name: str = "dataset"):

    metadata = {
        'dataset_name': dataset_name,
        'n_rows': df.height,
        'n_columns': df.width,
        'columns': {}
    }

    for col in df.columns:
        series = df[col]
        dtype = series.dtype

        col_meta = {
            'type': str(dtype),
            'missing': int(series.null_count())
        }

        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64):
            col_meta.update({
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max())
            })

        elif dtype == pl.Utf8 or dtype == pl.Categorical:
            col_meta['unique_values'] = int(series.n_unique())

        metadata['columns'][col] = col_meta

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, sort_keys=False, allow_unicode=True)

    return metadata

generate_yaml_metadata_from_lazyframe(lf, "metadata.yaml", dataset_name="my_dataset")
