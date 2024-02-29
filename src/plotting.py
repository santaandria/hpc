import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs, cartopy.feature as cfeature
from src.config import *


def plot_quantiles(dquant, extent, clevs, filename=None):
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    fig, ax = plt.subplots(
        1,
        figsize=(12, 8),
        subplot_kw=dict(projection=ccrs.LambertConformal(central_lon, central_lat)),
    )

    x1, y1 = np.meshgrid(dquant.longitude.values, dquant.latitude.values)

    cs = ax.contourf(
        x1,
        y1,
        dquant,
        levels=clevs,
        transform=ccrs.PlateCarree(),
        cmap="YlGnBu",
        extend="both",
        transform_first=True,
    )

    cb = plt.colorbar(
        cs, pad=0.05, shrink=0.8, aspect=30, orientation="horizontal", extend="both"
    )
    gl = ax.gridlines(color="gray", dms=True)

    ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor="k")
    ax.add_feature(cfeature.BORDERS, edgecolor="grey")
    ax.set_extent(extent)
    ax.set_title("100-Year Rainfall Map [mm/day]", fontsize=15)
    if filename:
        fig.savefig("./figures/" + filename + ".svg", dpi=600, format="svg")
    plt.show()
