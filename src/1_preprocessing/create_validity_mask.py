"""Build the HySWE domain mask and the no-10s analysis mask on the 30" SnowClass grid.

Inputs (under BASE_PATH/data): the MODIS snow-cover climatology
(swe_data/MODIS/NSIDC-0791_SSC_0.01Deg_WY2001-2023_V01.0.nc), the SnowClass dataset
(swe_data/SnowClass_GL_01km_30.0arcsec_2021_v01.0.nc), and the ERA5-Land map of
1981-01-01 in SOURCE_PATHS['era5_land'].

Outputs (in --out-dir, default BASE_PATH/data):
    valid_swe_regression_flags.nc  is_valid and exclusion_reason (0 valid, 1 perennial snow
                                   or ice, 2 snow free, 3 ocean or water, 4 no ERA5-Land data)
    analysis_mask_no10s.tif        is_valid without the pixels where ERA5-Land is 10 m

Usage: python src/1_preprocessing/create_validity_mask.py [--out-dir DIR]
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from pyproj import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import local_paths


def regrid_mask(mask_da, target_da, threshold=0.0, lon_dim='lon', lat_dim='lat'):
    """
    Resample a boolean/binary mask DataArray onto the grid of a target
    DataArray using area-weighted averaging, then threshold back to boolean.

    Parameters
    ----------
    mask_da : xr.DataArray
        Source mask (boolean, 0/1, or with NaNs) on a regular lat/lon grid.
    target_da : xr.DataArray
        DataArray whose lat/lon coordinates define the destination grid.
    threshold : float, default 0.0
        Minimum overlap fraction (0-1) required for an output cell to be True.
        - 0.0        -> any overlap at all counts as True (inclusive/dilated mask)
        - 0.5        -> majority overlap required
        - close to 1  -> nearly full overlap required
        - 1.0        -> full overlap required (a 1e-6 internal tolerance is
                         used to absorb floating-point error from averaging)
    lon_dim, lat_dim : str
        Names of the longitude/latitude dimensions/coordinates on both
        mask_da and target_da.

    Returns
    -------
    xr.DataArray
        Boolean mask on the target grid.
    """
    # --- source mask, cleaned to 0/1 ---
    src_da = mask_da.fillna(0).astype('uint8')
    lon_s = src_da[lon_dim].values
    lat_s = src_da[lat_dim].values
    dx_s = float(np.abs(lon_s[1] - lon_s[0]))
    dy_s = float(np.abs(lat_s[1] - lat_s[0]))
    src_transform = from_origin(lon_s.min() - dx_s / 2, lat_s.max() + dy_s / 2, dx_s, dy_s)
    src_arr = src_da.values

    # --- target grid geotransform ---
    lon_t = target_da[lon_dim].values
    lat_t = target_da[lat_dim].values
    dx_t = float(np.abs(lon_t[1] - lon_t[0]))
    dy_t = float(np.abs(lat_t[1] - lat_t[0]))
    tgt_transform = from_origin(lon_t.min() - dx_t / 2, lat_t.max() + dy_t / 2, dx_t, dy_t)
    dst_shape = (len(lat_t), len(lon_t))
    dst = np.zeros(dst_shape, dtype='float32')

    # --- area-weighted regridding: dst = overlap fraction with source==1 ---
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform, src_crs='EPSG:4326',
        dst_transform=tgt_transform, dst_crs='EPSG:4326',
        resampling=Resampling.average
    )

    # --- threshold back to boolean ---
    if threshold >= 1.0:
        out_bool = dst >= (1.0 - 1e-6)
    else:
        out_bool = dst > threshold

    out_da = xr.DataArray(
        out_bool,
        coords={lat_dim: lat_t, lon_dim: lon_t},
        dims=(lat_dim, lon_dim)
    )
    return out_da


def build_flags(snow_clim_da, swe_classes_ds, era5_mask):
    """Return (is_valid, exclusion_reason) DataArrays on the SnowClass grid."""

    # Seasonal snow in the MODIS climatology (classes 1-3), Northern Hemisphere up to 84N
    seasonal_mask_da = (snow_clim_da > 0) & (snow_clim_da < 4)
    nh_mask_da = seasonal_mask_da.sel(lat=slice(90, 0))
    nh_mask_da = nh_mask_da.sel(lat=slice(84, 0))

    # Regrid to 30": any overlap with seasonal snow counts; no snow and perennial snow need a majority
    mask_30arcsec_da = regrid_mask(mask_da=nh_mask_da, target_da=swe_classes_ds['SnowClass'], threshold=0.0)
    no_snow_da = snow_clim_da.sel(lat=slice(84, 0)) == 0
    no_snow_30arcsec_da = regrid_mask(mask_da=no_snow_da, target_da=swe_classes_ds['SnowClass'], threshold=0.5)
    per_snow_da = snow_clim_da.sel(lat=slice(84, 0)) == 4
    per_snow_30arcsec_da = regrid_mask(mask_da=per_snow_da, target_da=swe_classes_ds['SnowClass'], threshold=0.5)

    # Set to false all pixels whose snow class is not below 7 (exclude ice and ocean)
    ocean_class_mask = (swe_classes_ds['SnowClass'] == 8)
    ice_class_mask = (swe_classes_ds['SnowClass'] == 7)
    valid_class_mask = (swe_classes_ds['SnowClass'] < 7)
    corrected_mask_30arcsec_da = mask_30arcsec_da & valid_class_mask
    print(f"Number of pixels set to false due to ocean masking: {int(np.sum(ocean_class_mask & mask_30arcsec_da))}")
    print(f"Number of pixels set to false due to ice masking: {int(np.sum(ice_class_mask & mask_30arcsec_da))}")

    # Pixels in the mask without ERA5-Land data
    not_in_era5_mask = corrected_mask_30arcsec_da & ~era5_mask
    print(f"Number of pixels in corrected mask but not in ERA5-Land: {np.sum(not_in_era5_mask).item()}")

    # Primary exclusion reason; later assignments take precedence
    nanflag_da = xr.where(~valid_class_mask, 3, -1)                     # water bodies mask = 3
    nanflag_da = xr.where(ocean_class_mask, 3, nanflag_da)              # ocean mask = 3
    nanflag_da = xr.where(ice_class_mask, 1, nanflag_da)                # ice mask = 1
    nanflag_da = xr.where(per_snow_30arcsec_da, 1, nanflag_da)          # perennial snow = 1
    nanflag_da = xr.where(no_snow_30arcsec_da, 2, nanflag_da)           # no snow = 2
    nanflag_da = xr.where(corrected_mask_30arcsec_da, 0, nanflag_da)    # valid pixels = 0
    nanflag_da = xr.where(not_in_era5_mask, 4, nanflag_da)              # no ERA5-Land data = 4

    print(f"Number of unflagged pixels (nanflag == -1): {np.sum(nanflag_da.values == -1)}")
    print("Tagging unflagged pixels as 3 (ocean/water), as they have been identified as water bodies after visual inspection")
    nanflag_da = xr.where(nanflag_da == -1, 3, nanflag_da)

    # Check that nanflag_da == 0 returns the same number of pixels as the corrected mask
    num_valid_pixels = np.sum(corrected_mask_30arcsec_da.values) - np.sum(not_in_era5_mask.values)
    num_nanflag_pixels = np.sum(nanflag_da.values == 0)
    if num_valid_pixels != num_nanflag_pixels:
        raise ValueError("Mismatch between valid pixels and nanflag==0 pixels")
    print(f"Number of valid pixels matches nanflag==0 pixels: {num_valid_pixels}")

    class_names = {0: 'Valid', 1: 'Perennial Snow or Ice', 2: 'Snow Free', 3: 'Ocean or Water', 4: 'Missing ERA5-Land Data'}
    for flag, name in class_names.items():
        count = np.sum(nanflag_da.values == flag)
        print(f"{name}: {count} pixels ({count / nanflag_da.size * 100:.2f}% of total)")

    valid_nanflag_mask = nanflag_da == 0
    return valid_nanflag_mask, nanflag_da


def check_flags(valid_nanflag_mask, nanflag_da):
    """Check the flag arrays before writing."""
    if valid_nanflag_mask.dims != ("lat", "lon"):
        raise ValueError("valid_nanflag_mask must have dimensions ('lat', 'lon').")
    if nanflag_da.dims != ("lat", "lon"):
        raise ValueError("nanflag_da must have dimensions ('lat', 'lon').")
    if valid_nanflag_mask.isnull().any():
        raise ValueError("is_valid must not contain missing values: 0 and 1 are both valid categories.")
    if nanflag_da.isnull().any():
        raise ValueError("exclusion_reason must use 255, rather than NaN, for unavailable classification.")
    if not np.isin(np.asarray(valid_nanflag_mask.values), [0, 1]).all():
        raise ValueError("valid_nanflag_mask may contain only 0 and 1.")
    if not np.isin(np.asarray(nanflag_da.values), [0, 1, 2, 3, 4, 255]).all():
        raise ValueError("nanflag_da may contain only 0–4, or 255 for unavailable classification.")


def flags_dataset(valid_nanflag_mask, nanflag_da):
    """Build the CF-annotated flags dataset."""

    # Pixel eligibility flag: 0 and 1 are both valid, meaningful categories; neither is a fill value
    nanflag_ds = valid_nanflag_mask.astype("uint8").to_dataset(name="is_valid")
    nanflag_ds["is_valid"].attrs = {
        "long_name": "Pixel eligibility for HySWE dataset",
        "flag_values": np.array([0, 1], dtype="uint8"),
        "flag_meanings": "invalid valid",
        "grid_mapping": "crs",
    }
    nanflag_ds["is_valid"].encoding["_FillValue"] = None

    # Primary exclusion reason: 0 denotes an eligible pixel, 1-4 the reason for ineligibility
    nanflag_ds["exclusion_reason"] = nanflag_da.astype("uint8")
    nanflag_ds["exclusion_reason"].attrs = {
        "long_name": "HySWE pixel eligibility or primary exclusion reason",
        "flag_values": np.array([0, 1, 2, 3, 4], dtype="uint8"),
        "flag_meanings": (
            "valid "
            "perennial_snow_or_ice "
            "snow_free "
            "ocean_or_water "
            "missing_era5_land_data"
        ),
        "grid_mapping": "crs",
    }
    nanflag_ds["exclusion_reason"].encoding["_FillValue"] = None

    # Coordinate variables
    nanflag_ds["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north", "axis": "Y"})
    nanflag_ds["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east", "axis": "X"})
    for coord in ["lat", "lon"]:
        nanflag_ds[coord].encoding["_FillValue"] = None

    # CF grid-mapping variable
    nanflag_ds["crs"] = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
    )
    nanflag_ds["crs"].attrs.update({"crs_wkt": CRS.from_epsg(4326).to_wkt()})
    nanflag_ds["crs"].encoding["_FillValue"] = None

    # Minimal global attributes for a standalone mask product
    nanflag_ds.attrs = {
        "title": "HySWE pixel eligibility and exclusion reasons",
        "summary": (
            "Pixel-level eligibility mask and primary exclusion-reason flags "
            "used for HySWE snow water equivalent dataset."
        ),
        "Conventions": "CF-1.13, ACDD-1.3",
        "product_version": "v1.0",
        "keywords": "snow water equivalent, SWE, mask, quality flag, HySWE",
    }

    # Pre-write checks
    assert nanflag_ds["is_valid"].dtype == np.uint8
    assert nanflag_ds["exclusion_reason"].dtype == np.uint8
    assert set(np.unique(nanflag_ds["exclusion_reason"])) <= {0, 1, 2, 3, 4}
    assert ((nanflag_ds["is_valid"] == 1) == (nanflag_ds["exclusion_reason"] == 0)).all()
    assert set(nanflag_ds.coords) == {"lat", "lon"}
    return nanflag_ds


def main(out_dir: Path):
    data_dir = local_paths.BASE_PATH / 'data'

    # MODIS snow-cover climatology
    snow_clim_ds = xr.open_dataset(data_dir / 'swe_data' / 'MODIS' / 'NSIDC-0791_SSC_0.01Deg_WY2001-2023_V01.0.nc',
                                   decode_timedelta=True)
    snow_clim_da = snow_clim_ds['snow_class_climatology']

    # SnowClass dataset: its 0-84N part, latitude descending, is the output grid
    snow_class_ds = xr.open_dataset(data_dir / 'swe_data' / 'SnowClass_GL_01km_30.0arcsec_2021_v01.0.nc')
    snow_class_ds = snow_class_ds.sel(lat=slice(0, 84))
    swe_classes_ds = snow_class_ds.sortby('lat', ascending=False)

    # One ERA5-Land map on the same grid: valid pixels, and the values for the no-10s mask
    era5_path = Path(local_paths.SOURCE_PATHS['era5_land']) / 'sd_ERA5-Land_01_01_1981.tif'
    with rasterio.open(era5_path) as src:
        era5_values = src.read(1, masked=True).filled(np.nan)
    era5_mask = ~np.isnan(era5_values)  # True for valid ERA5-Land pixels
    if era5_mask.shape != swe_classes_ds['SnowClass'].shape:
        raise ValueError(f"ERA5-Land map shape {era5_mask.shape} does not match the SnowClass grid "
                         f"{swe_classes_ds['SnowClass'].shape}")

    # Validity flags
    valid_nanflag_mask, nanflag_da = build_flags(snow_clim_da, swe_classes_ds, era5_mask)
    check_flags(valid_nanflag_mask, nanflag_da)
    nanflag_ds = flags_dataset(valid_nanflag_mask, nanflag_da)

    out_dir.mkdir(parents=True, exist_ok=True)
    comp = {"zlib": True, "complevel": 4}
    encoding = {
        "is_valid": {"dtype": "uint8", "_FillValue": None, **comp},
        "exclusion_reason": {"dtype": "uint8", "_FillValue": None, **comp},
        "lat": {"_FillValue": None},
        "lon": {"_FillValue": None},
        "crs": {"_FillValue": None},
    }
    flags_path = out_dir / 'valid_swe_regression_flags.nc'
    nanflag_ds.to_netcdf(flags_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
    print(f"Saved the pixel eligibility and exclusion reasons dataset to {flags_path}")

    # No-10s analysis mask: valid pixels where ERA5-Land is not the constant 10 m
    no10s = (nanflag_ds['is_valid'].values == 1) & (era5_values != 10)
    n_removed = int(nanflag_ds['is_valid'].values.sum()) - int(no10s.sum())
    print(f"Number of valid pixels in the no-10s analysis mask: {int(no10s.sum())} ({n_removed} pixels with 10 m removed)")
    no10s_da = xr.DataArray(no10s.astype(np.uint8), coords={'y': swe_classes_ds['lat'].values, 'x': swe_classes_ds['lon'].values},
                            dims=('y', 'x')).rio.write_crs('EPSG:4326')
    no10s_path = out_dir / 'analysis_mask_no10s.tif'
    no10s_da.rio.to_raster(no10s_path, driver='GTiff', dtype='uint8', compress='lzw')
    print(f"Saved the no-10s analysis mask to {no10s_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='Folder for the outputs (default: BASE_PATH/data). Use another folder to compare with the existing files.')
    args = parser.parse_args()
    main(args.out_dir or local_paths.BASE_PATH / 'data')
