import os
import numpy as np

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.merge import merge

def clip_raster(raster_file, shapefile, output_file, invert=False, crop=True):
    # Load study area
    shapefile_data = gpd.read_file(shapefile)
    # Read the raster file using rasterio
    with rasterio.open(raster_file) as src:
        # Copy the metadata from the source raster
        profile = src.profile
        src_crs = src.crs

        # Clip the raster using the shapefile geometry
        shapefile_data = shapefile_data.to_crs(src_crs)
        clipped_raster, transform = mask(
            src, shapefile_data.geometry, crop=crop, invert=invert
        )

    # Update the metadata with new dimensions and transform
    profile.update(
        {
            "driver": "GTiff",
            "height": clipped_raster.shape[1],
            "width": clipped_raster.shape[2],
            "transform": transform,
            "nodata": src.nodata,
        }
    )

    # Save the clipped raster to a new file
    with rasterio.open(output_file, "w", **profile) as dest:
        dest.write(clipped_raster)


def ntems_mask(raster_path, ntems_raster, ntems_values, output_path):
    # Open the raster to be masked
    with rasterio.open(raster_path, "r+") as src:
        raster_data = src.read()
        band_descriptions = src.descriptions  # Get band descriptions

        with rasterio.open(ntems_raster) as ntems_mask_src:
            ntems_mask_data = ntems_mask_src.read(1)

            rows_diff = raster_data.shape[1] - ntems_mask_data.shape[0]
            cols_diff = raster_data.shape[2] - ntems_mask_data.shape[1]

            pad_width = ((0, max(rows_diff, 0)), (0, max(cols_diff, 0)))

            ntems_mask_data_padded = np.pad(
                ntems_mask_data, pad_width, mode="constant", constant_values=0
            )
            ntems_mask_data_cropped = ntems_mask_data_padded[
                : raster_data.shape[1], : raster_data.shape[2]
            ]
            ntems_mask = np.isin(ntems_mask_data_cropped, ntems_values)

            masked_data = np.copy(raster_data)
            if src.meta["nodata"] is not None:
                for band in range(src.count):
                    masked_data[band][ntems_mask] = src.meta["nodata"]
            else:
                src.nodata = 0
                for band in range(src.count):
                    masked_data[band][ntems_mask] = src.meta["nodata"]

            profile = src.profile
            profile.update(
                count=src.count,
                height=ntems_mask_data_cropped.shape[0],
                width=ntems_mask_data_cropped.shape[1],
                descriptions=band_descriptions,
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(masked_data)

def merge_rasters(input_files, output_file, method="min"):
    out_folder = os.path.dirname(output_file)
    os.makedirs(out_folder, exist_ok=True)
    # Open the input files
    src_files = [rasterio.open(file) for file in input_files]

    band_descriptions = src_files[0].descriptions

    merged, out_trans = merge(src_files, method=method)

    # Create the output raster file
    profile = src_files[0].profile
    profile.update(
        {"height": merged.shape[1], "width": merged.shape[2], "transform": out_trans}
    )
    profile.update(descriptions=band_descriptions)

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(merged)

            # Save the mosaic to the output path
            with rasterio.open(output_file, "w", **profile) as dest:
                dest.write(dataset.read())