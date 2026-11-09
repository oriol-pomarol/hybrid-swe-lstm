"""Assign each NH-SWE station its snow class, topographic and land-cover attributes (nearest pixel).

Inputs (under BASE_PATH/data): STATION_METADATA.csv,
swe_data/SnowClass_GL_01km_30.0arcsec_2021_v01.0.nc, the GMTED-based layers
topo_data/Amatulli_2016/*_GMTED*.tif and, if present, the aggregated land-cover maps
land_cover_data/aggregated/aggregated_land_cover_<year>.nc (from aggregate_land_cover.py).

Outputs (in --out-dir, default the input locations):
    STATION_METADATA.csv  with the SnowClass column (default: updated in place)
    NH_TOPO.csv           one column per layer, named <variable>_<aggregation>, e.g. elevation_mn
                          (default: BASE_PATH/data/topo_data/)
    NH_LC_YEARLY.csv      land-cover pixel counts per station and year, one column per class code
                          (default: BASE_PATH/data/land_cover_data/)
    NH_LAND_COVER.csv     the land-cover predictors: fraction of each class group over all years,
                          named <group>_lc (default: BASE_PATH/data/land_cover_data/)

Usage: python src/1_preprocessing/extract_station_attributes.py [--out-dir DIR]
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import shared
import local_paths

# Snow class codes of the SnowClass dataset
CLASS_DICT = {
    1: "Tundra",
    2: "Boreal Forest",
    3: "Maritime",
    4: "Ephemeral",
    5: "Prairie",
    6: "Montane Forest",
    7: "Ice",    # glaciers and ice sheets
    8: "Ocean",
}


def nearest_indices(index: pd.Index, values: np.ndarray) -> np.ndarray:
    """Positions of the nearest coordinates, as xarray's .sel(method='nearest') finds them."""
    positions = index.get_indexer(values, method='nearest')
    if (positions < 0).any():
        raise ValueError("Some station coordinates could not be matched to the grid.")
    return positions


def station_snow_classes(metadata: pd.DataFrame) -> pd.Series:
    """Nearest SnowClass pixel of each station, as a class name."""
    path_snow = local_paths.BASE_PATH / 'data' / 'swe_data' / 'SnowClass_GL_01km_30.0arcsec_2021_v01.0.nc'
    swe_classes = xr.open_dataset(path_snow, engine='netcdf4', decode_coords='all')
    swe_classes = swe_classes.sel(lat=slice(0, 90))  # Northern Hemisphere
    snow_class_da = swe_classes['SnowClass'].squeeze()
    if snow_class_da.dims != ('lat', 'lon'):
        raise ValueError(f"Unexpected SnowClass dimensions: {snow_class_da.dims}")

    iy = nearest_indices(snow_class_da.indexes['lat'], metadata['LAT'].values)
    ix = nearest_indices(snow_class_da.indexes['lon'], metadata['LON'].values)
    values = snow_class_da.values[iy, ix]
    return pd.Series([CLASS_DICT[value] for value in values], index=metadata.index, name='SnowClass')


def station_topography(metadata: pd.DataFrame) -> pd.DataFrame:
    """Nearest pixel of each GMTED topography layer for each station."""
    topo_dir = local_paths.BASE_PATH / 'data' / 'topo_data' / 'Amatulli_2016'
    files = sorted(topo_dir.glob('*_GMTED*.tif'))
    if not files:
        raise FileNotFoundError(f"No *_GMTED*.tif files in {topo_dir}")

    columns = {}
    for file in files:
        variable_name = file.stem.split('_')[0] + '_' + file.stem.split('_')[1][-2:]  # e.g. 'elevation_mn'
        if variable_name in columns:
            raise ValueError(f"Two topography files map to the column {variable_name}; check {topo_dir}")
        ds = rioxarray.open_rasterio(file).isel(band=0)
        iy = nearest_indices(ds.indexes['y'], metadata['LAT'].values)
        ix = nearest_indices(ds.indexes['x'], metadata['LON'].values)
        columns[variable_name] = ds.values[iy, ix]
        print(f"Extracted {variable_name} from {file.name}")

    topo_df = pd.DataFrame(columns, index=metadata.index)
    topo_df.index.name = 'NHSWEID'
    return topo_df


def station_land_cover_yearly(metadata: pd.DataFrame, aggregated_dir: Path) -> pd.DataFrame:
    """Nearest-pixel land-cover class counts of each station and year, one column per class code."""
    yearly_dfs = []
    grid_y, grid_x = None, None
    for year in shared.LC_YEARS:
        file = aggregated_dir / f'aggregated_land_cover_{year}.nc'
        with xr.open_dataset(file, engine='netcdf4') as ds:
            lc_freq = ds['lc_freq']
            if lc_freq.dims != ('y', 'x', 'lc_class') or lc_freq.sizes['lc_class'] != len(shared.LC_CLASS_GROUPS):
                raise ValueError(f"Unexpected lc_freq layout in {file}: {dict(lc_freq.sizes)}")

            # Station pixels from the first year; every year must be on the same grid
            if grid_y is None:
                grid_y, grid_x = ds.indexes['y'], ds.indexes['x']
                iy = nearest_indices(grid_y, metadata['LAT'].values)
                ix = nearest_indices(grid_x, metadata['LON'].values)
            elif not (ds.indexes['y'].equals(grid_y) and ds.indexes['x'].equals(grid_x)):
                raise ValueError(f"The grid of {file} differs from the one of {shared.LC_YEARS[0]}")

            counts = np.stack([lc_freq.isel(y=int(y), x=int(x)).values for y, x in zip(iy, ix)])

        year_df = pd.DataFrame(counts, index=metadata.index, columns=list(shared.LC_CLASS_GROUPS))
        year_df.index.name = 'NHSWEID'
        yearly_dfs.append(year_df.assign(year=year).set_index('year', append=True))
        print(f"Extracted the land cover of {year} from {file.name}")

    station_years = pd.MultiIndex.from_product([metadata.index, shared.LC_YEARS], names=['NHSWEID', 'year'])
    return pd.concat(yearly_dfs).reindex(station_years)


def station_land_cover_fractions(lc_yearly_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each class group over all years per station, with columns named <group>_lc."""
    group_counts = lc_yearly_df.T.groupby(shared.LC_CLASS_GROUPS).sum().T
    total_counts = group_counts.groupby(level='NHSWEID', sort=False).sum()
    lc_df = total_counts.div(total_counts.sum(axis=1), axis=0).add_suffix('_lc')
    lc_df.index.name = 'NHSWEID'
    return lc_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='Folder for both outputs (default: overwrite data/STATION_METADATA.csv and '
                             'data/topo_data/NH_TOPO.csv). Use another folder to compare with the existing files.')
    args = parser.parse_args()

    metadata_path = local_paths.BASE_PATH / 'data' / 'STATION_METADATA.csv'
    metadata = pd.read_csv(metadata_path, index_col=0)
    print(f"Loaded {len(metadata)} stations from {metadata_path}")

    snow_classes = station_snow_classes(metadata)
    print(snow_classes.value_counts())
    topo_df = station_topography(metadata)

    land_cover_dir = local_paths.BASE_PATH / 'data' / 'land_cover_data'
    aggregated_dir = land_cover_dir / 'aggregated'
    missing_lc = [year for year in shared.LC_YEARS
                  if not (aggregated_dir / f'aggregated_land_cover_{year}.nc').exists()]
    if missing_lc:
        print(f"Skipping the land cover: no aggregated land-cover maps for {missing_lc} in {aggregated_dir}")
        lc_yearly_df = None
    else:
        lc_yearly_df = station_land_cover_yearly(metadata, aggregated_dir)
        lc_df = station_land_cover_fractions(lc_yearly_df)

    if args.out_dir is None:
        metadata_out = metadata_path
        topo_out = local_paths.BASE_PATH / 'data' / 'topo_data' / 'NH_TOPO.csv'
        lc_out_dir = land_cover_dir
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        metadata_out = args.out_dir / 'STATION_METADATA.csv'
        topo_out = args.out_dir / 'NH_TOPO.csv'
        lc_out_dir = args.out_dir

    metadata['SnowClass'] = snow_classes
    metadata.to_csv(metadata_out)
    print(f"Saved the station metadata with the SnowClass column to {metadata_out}")
    topo_df.to_csv(topo_out)
    print(f"Saved the topographic attributes to {topo_out}")
    if lc_yearly_df is not None:
        lc_yearly_df.to_csv(lc_out_dir / 'NH_LC_YEARLY.csv')
        print(f"Saved the yearly land-cover counts to {lc_out_dir / 'NH_LC_YEARLY.csv'}")
        lc_df.to_csv(lc_out_dir / 'NH_LAND_COVER.csv')
        print(f"Saved the land-cover predictors to {lc_out_dir / 'NH_LAND_COVER.csv'}")
