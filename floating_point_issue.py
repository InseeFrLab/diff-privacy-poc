import polars as pl
import opendp.prelude as dp

dp.enable_features("contrib")

context = dp.Context.compositor(
    data=pl.scan_csv(dp.examples.get_france_lfs_path(), ignore_errors=True),
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(rho=1.5),
    split_evenly_over=3,
    margins=[
        dp.polars.Margin(
            # the biggest (and only) partition is no larger than
            #    France population * number of quarters
            max_partition_length=60_000_000*35
        ),
    ],
)

query_num_responses = context.query().select(dp.len())

print(query_num_responses.summarize(alpha=0.1))

query = context.query().with_columns(pl.lit(1).alias("colonne_comptage"))

expr = (
            pl.col("colonne_comptage")
            .fill_null(1)
            .dp.sum((0, 1))
            .alias("count")
        )
query = query.select(expr)

print(query.summarize(alpha=0.1))

query = context.query().with_columns(pl.lit(1.0).alias("colonne_comptage"))

expr = (
            pl.col("colonne_comptage")
            .fill_null(0)
            .dp.sum((0, 1))
            .alias("count")
        )
query = query.select(expr)

print(query.summarize(alpha=0.1))
