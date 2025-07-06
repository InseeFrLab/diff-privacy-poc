import polars as pl
import opendp.prelude as dp

dp.enable_features("contrib")

context = dp.Context.compositor(
    data=pl.scan_csv(dp.examples.get_france_lfs_path(), ignore_errors=True),
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(rho=1/2),
    split_evenly_over=1,
    margins=[
        dp.polars.Margin(
            # the biggest (and only) partition is no larger than
            #    France population * number of quarters
            max_partition_length=1
        ),
    ],
)

l, u = 0, 5

m = (l + u)/2
b_sup = (u - l)/2
query = context.query().with_columns(pl.lit(5).alias("colonne_comptage"))

expr = (
            (pl.col("colonne_comptage") - m)
            .fill_null(0)
            .fill_nan(0)
            .dp.sum((-b_sup, b_sup))
            .alias("sum")
        )
query = query.select(expr)

print(query.summarize(alpha=0.1))
