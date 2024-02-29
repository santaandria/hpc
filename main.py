from src.data_processing import compute_mev_params, compute_mev_quant
from src.plotting import plot_quantiles
import xarray as xr
import src.config as config
import numpy as np


def main():
    Fi = 0.99
    x0 = 1
    threshold = 0.1
    ds = xr.open_mfdataset("./data/tp.nc", parallel=True)
    ds = ds.chunk({"time": -1})
    print("Dataset Loaded Successfully - Computing...")
    dparam = compute_mev_params(ds.tp, threshold)
    quantiles = compute_mev_quant(dparam, Fi, x0, threshold)
    plot_quantiles(
        quantiles,
        extent=[6.5, 18.5, 36.5, 47.5],
        clevs=np.linspace(2, 30, 15),
        filename="mev_100_it",
    )
    print("Computation Complete")


if __name__ == "__main__":
    main()
