from pathlib import Path
import geopandas as gpd
import shapely
from typing import Union, List


def ensure_path(path: Union[str, Path]) -> Path:
    if isinstance(path, str):
        return Path(path)
    return path


def get_tif_list(tif_path: Union[str, Path]) -> List:
    """
    Retrieves a list of TIFF files from the specified directory.

    Args:
        tif_path (str): The path to the directory containing the TIFF files.

    Returns:
        list: A list of Path objects representing the TIFF files in the directory.
    """
    tif_list = [
        p for p in ensure_path(tif_path).iterdir() if p.suffix.lower() == ".tif"
    ]
    return tif_list


def get_dike_traject_geometry(
    dike_traject_path: Union[str, Path], dike_traject_fnames: List, buffer_size: int = 1
) -> gpd.GeoDataFrame:
    """
    Retrieve the dike traject geometry from the specified path.

    Parameters:
    dike_traject_path (str): The path to the dike traject files.
    buffer_size (float): The buffer size to apply to the dike traject geometry (in meters).

    Returns:
    gdf_dike_traject (GeoDataFrame): A GeoDataFrame containing the dike traject geometry.
    """
    geoms = []
    for dike_traject_fname in dike_traject_fnames:

        # Read the file and explode any MultiPolygons into individual parts
        exploded_geometries = (
            gpd.read_file(dike_traject_path / dike_traject_fname)
            .explode(index_parts=True)
            .geometry
        )
        for geom in exploded_geometries:
            if isinstance(geom, shapely.geometry.Polygon):
                geoms.append(geom)
            elif isinstance(geom, shapely.geometry.MultiPolygon):
                # If the geometry is a MultiPolygon, extend the list with its components
                geoms.extend(list(geom.geoms))
            else:
                print(f"Unhandled geometry type: {type(geom)}")

    # Attempt to create a MultiPolygon from the collected Polygon objects
    try:
        dike_traject_geoms = shapely.geometry.MultiPolygon(geoms)
    except TypeError as e:
        print(f"Error creating MultiPolygon: {e}")
        print([type(g) for g in geoms])

    # Buffer the MultiPolygon if creation was successful
    if "dike_traject_geoms" in locals():
        dike_traject_geoms = dike_traject_geoms.buffer(buffer_size)

    gdf_dike_traject = gpd.GeoDataFrame({"geometry": [dike_traject_geoms]}, crs=28992)
    return gdf_dike_traject
