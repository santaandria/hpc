import xarray as xr
from src.utils import wei_fit_pwm, mev_quant


def compute_mev_params(ds, threshold):
    results = []
    for year, indices in ds.groupby("time.year").groups.items():
        yearly_data = ds.isel({"time": indices}) - threshold
        # masked_data = yearly_data.where(dmask.lsmask.isel(time=0) == 1)
        result = xr.apply_ufunc(
            wei_fit_pwm,
            yearly_data,
            input_core_dims=[["time"]],
            vectorize=True,
            output_core_dims=[["parameter"]],
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {"parameter": 3}},
        )
        result["year"] = year
        results.append(result)
    return xr.concat(results, dim="year").assign_coords(parameter=["n", "c", "w"])


def compute_mev_quant(dparam, Fi, x0, threshold):
    return xr.apply_ufunc(
        mev_quant,
        Fi,
        x0,
        dparam.sel(parameter="n"),
        dparam.sel(parameter="c"),
        dparam.sel(parameter="w"),
        kwargs={"potmode": True, "thresh": threshold},
        input_core_dims=[
            [],
            [],
            ["year"],
            ["year"],
            ["year"],
        ],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, bool],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )[0]
