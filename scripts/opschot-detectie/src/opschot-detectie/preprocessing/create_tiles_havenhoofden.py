import hydra
from pathlib import Path
import geopandas as gpd
import shapely
import rasterio.features
from osgeo import gdal
from shapely.geometry import box
import numpy as np
import tqdm
import logging
import utils
from logging.config import fileConfig
from omegaconf import DictConfig

fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger("root")


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """ "
    Create tiles from drone imagery RGB rasters.
    """

    # create output directory for tiles
    tile_dataset_path = Path(cfg.disk_path) / "tile_dataset_havenhoofden"
    if not tile_dataset_path.exists():
        tile_dataset_path.mkdir()
        logger.info(f"Created output directory: {tile_dataset_path}")
    else:
        logger.info(f"Directory: {tile_dataset_path} Already exists!")

    # get list of RGB rasters (tif files)
    tif_list = utils.get_tif_list(tif_path=cfg.data_path)

    # loop over rasters
    for tif_path in tif_list:
        logger.info(f"Processing raster: {tif_path}")

        # open raster
        src_ds = gdal.Open(str(tif_path))
        logger.info(f"Opened raster: {tif_path}")

        # get bounds
        xmin, xres, xskew, ymax, yskew, yres = src_ds.GetGeoTransform()
        xmax = xmin + (src_ds.RasterXSize * xres)
        ymin = ymax + (src_ds.RasterYSize * yres)
        gdf_tif_bounds = gpd.GeoDataFrame(
            geometry=[box(xmin, ymin, xmax, ymax)], crs=28992
        )

        # get dike traject geometry
        dike_traject_path = Path(cfg.disk_path) / "dike_trajects"
        gdf_dike_traject = utils.get_dike_traject_geometry(
            dike_traject_path=dike_traject_path,
            dike_traject_fnames=cfg.dike_traject_havenhoofden_fnames,
        ).dissolve()

        # get intersection with particular raster file
        gdf_dike_traject_in_raster = gdf_dike_traject.intersection(
            gdf_tif_bounds.unary_union
        )

        # tile definition
        nx = int(np.abs(np.ceil((xmax - xmin) / (cfg.tile_size_x * xres))))
        ny = int(np.abs(np.ceil((ymax - ymin) / (cfg.tile_size_y * yres))))
        affine = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, nx, ny)

        # get tiles for particular raster file
        rasterized = rasterio.features.rasterize(
            [gdf_tif_bounds.geometry.iloc[0]],
            out_shape=(ny, nx),
            fill=0,
            out=None,
            transform=affine,
            all_touched=True,
            default_value=1,
            dtype=None,
        )
        rasterized = rasterized.astype(np.int32)

        # number of tiles
        ntiles = len(rasterized[rasterized != 0])
        rasterized[rasterized != 0] = 1 + np.random.choice(
            ntiles, ntiles, replace=False
        )

        # create geodataframe with tiles for particular raster file
        tiles = [
            (v, shapely.geometry.shape(p))
            for p, v in rasterio.features.shapes(rasterized, transform=affine)
        ]
        tiles = gpd.GeoDataFrame(
            tiles, columns=["value", "geometry"], geometry="geometry", crs="epsg:28992"
        )

        # get tiles intersecting with dike traject
        tiles_intersect = tiles[
            tiles.intersects(gdf_dike_traject_in_raster.geometry.iloc[0])
        ]

        # select tiles with at least % intersection with dike traject

        tiles_intersect = tiles_intersect[
            tiles_intersect.geometry.intersection(
                gdf_dike_traject_in_raster.geometry.iloc[0]
            ).area
            / tiles_intersect.geometry.area
            >= cfg.tile_overlap_threshold
        ]

        if len(tiles_intersect) > 0:

            # create output directory for particular raster
            raster_out_dir = tile_dataset_path / tif_path.stem
            if not raster_out_dir.exists():
                raster_out_dir.mkdir()
                logger.info(f"Created output directory: {raster_out_dir}")
            else:
                logger.info(f"Directory {raster_out_dir} already exists!")

            # create output directory for tiles
            tiles_out_path = raster_out_dir / "tiles_havenhoofden"
            if not tiles_out_path.exists():
                tiles_out_path.mkdir()
                logger.info(f"Created output directory: {tiles_out_path}")
            else:
                logger.info(f"Directory {tiles_out_path} already exists!")

            gdf_tif_bounds.to_file(raster_out_dir / f"{tif_path.stem}_bounds.gpkg")
            logger.info(
                f"Created bounds: {raster_out_dir / f'{tif_path.stem}_bounds.gpkg'}"
            )
            gdf_dike_traject_in_raster.to_file(
                raster_out_dir / f"{tif_path.stem}_dike_traject_part.gpkg"
            )
            logger.info(
                f"Created dike traject: {raster_out_dir / f'{tif_path.stem}_dike_traject_part.gpkg'}"
            )
            tiles.to_file(raster_out_dir / f"{tif_path.stem}_tiles.gpkg")
            logger.info(
                f"Created tiles: {raster_out_dir / f'{tif_path.stem}_tiles.gpkg'}"
            )

            tiles_intersect = tiles_intersect.reset_index(drop=True)
            tiles_intersect["xmin"] = tiles_intersect.geometry.bounds["minx"]
            tiles_intersect["ymin"] = tiles_intersect.geometry.bounds["miny"]
            tiles_intersect["xmax"] = tiles_intersect.geometry.bounds["maxx"]
            tiles_intersect["ymax"] = tiles_intersect.geometry.bounds["maxy"]
            tiles_intersect.to_file(
                raster_out_dir / f"{tif_path.stem}_tiles_intersects.gpkg"
            )
            logger.info(
                f"Created intersecting tiles: {raster_out_dir / f'{tif_path.stem}_tiles_intersects.gpkg'}"
            )

            logger.info(f"Number of intersecting tiles: {len(tiles_intersect)}")
            for i in tqdm.tqdm(range(len(tiles_intersect))):

                xmin_cell = tiles_intersect.iloc[i]["xmin"]
                ymin_cell = tiles_intersect.iloc[i]["ymin"]
                xmax_cell = tiles_intersect.iloc[i]["xmax"]
                ymax_cell = tiles_intersect.iloc[i]["ymax"]

                options_list = [
                    "-b 1",
                    "-b 2",
                    "-b 3",
                    "-of JPEG",
                    f"-projwin {xmin_cell}, {ymax_cell}, {xmax_cell}, {ymin_cell}",
                ]

                gdal.Translate(
                    str(tiles_out_path / f"{tif_path.stem}_{i}.jpeg"),
                    str(tif_path),
                    options=" ".join(options_list),
                )

                logger.info(
                    f"Created tile: {tiles_out_path / f'{tif_path.stem}_{i}.jpeg'}"
                )


if __name__ == "__main__":
    main()
