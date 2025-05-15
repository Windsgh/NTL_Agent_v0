import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import mapping
import pandas as pd
from tqdm import tqdm  # 加个进度条
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

def calc_TNTL(ntl_array):
    """Total Nighttime Light Intensity (TNTL) 总夜间灯光强度"""
    return np.nansum(ntl_array)

def calc_LArea(ntl_array, pixel_area):
    """Lit Area (LArea) 发光区域面积"""
    lit_mask = ntl_array > 0
    return np.sum(lit_mask) * pixel_area  # pixel_area 单位：平方米或公顷

def calc_3DPLand(ntl_array):
    """3D Percentage of Landscape (3DPLand) 三维灯光景观占比"""
    max_ntl = np.nanmax(ntl_array)
    n_pixels = np.sum(~np.isnan(ntl_array))
    if max_ntl == 0 or n_pixels == 0:
        return np.nan
    return np.nansum(ntl_array) / (max_ntl * n_pixels)

def calc_3DED(ntl_array):
    """3D Edge Density (3DED) 三维边缘密度"""
    # 简化版处理：按非零区域边界近似
    from scipy import ndimage

    lit_mask = ntl_array > 0
    labeled, num_features = ndimage.label(lit_mask)

    perimeter = 0
    total_intensity = np.nansum(ntl_array)

    for region_label in range(1, num_features + 1):
        region = (labeled == region_label)
        edges = ndimage.binary_dilation(region) ^ region
        perimeter += np.sum(edges)

    if total_intensity == 0:
        return np.nan
    return perimeter / total_intensity

def calc_3DLPI(ntl_array):
    """3D Largest Patch Index (3DLPI) 三维最大斑块指数"""
    from scipy import ndimage

    lit_mask = ntl_array > 0
    labeled, num_features = ndimage.label(lit_mask)

    region_intensities = []
    for region_label in range(1, num_features + 1):
        region = (labeled == region_label)
        region_intensities.append(np.nansum(ntl_array[region]))

    if len(region_intensities) == 0:
        return np.nan
    return np.nanmax(region_intensities) / np.nansum(ntl_array)

def calc_ANTL(ntl_array):
    """Average Nighttime Light Intensity (ANTL) 平均夜间灯光强度"""
    valid_pixels = np.sum(~np.isnan(ntl_array))
    if valid_pixels == 0:
        return np.nan
    return np.nansum(ntl_array) / valid_pixels

def calc_DNTL(ntl_array):
    """Deviation of Nighttime Light Intensity (DNTL) 夜间灯光强度离散度"""
    valid_pixels = np.sum(~np.isnan(ntl_array))
    mean_ntl = calc_ANTL(ntl_array)
    if valid_pixels == 0:
        return np.nan
    deviation = np.nansum((ntl_array - mean_ntl) ** 2) / valid_pixels
    return deviation


# 之前定义的指数计算函数（可以直接用上面那版）
# calc_TNTL, calc_LArea, calc_3DPLand, calc_3DED, calc_3DLPI, calc_ANTL, calc_DNTL

def calc_indices_per_polygon(ntl_array, mask_array, pixel_area):
    """给定 NTL影像数组和掩膜，计算景观指数"""
    masked_ntl = np.where(mask_array, ntl_array, np.nan)

    return {
        'TNTL': calc_TNTL(masked_ntl),
        'LArea': calc_LArea(masked_ntl, pixel_area),
        '3DPLand': calc_3DPLand(masked_ntl),
        '3DED': calc_3DED(masked_ntl),
        '3DLPI': calc_3DLPI(masked_ntl),
        'ANTL': calc_ANTL(masked_ntl),
        'DNTL': calc_DNTL(masked_ntl)
    }

def batch_compute_ntl_indices(ntl_tif_path, shapefile_path, output_csv_path, pixel_size):
    """
    针对每个行政区（shapefile）计算夜光景观指数，并保存成csv
    """
    # 打开栅格
    with rasterio.open(ntl_tif_path) as src:
        ntl_data = src.read(1).astype(np.float32)
        ntl_data[ntl_data == src.nodata] = np.nan
        ntl_profile = src.profile

    # 打开矢量
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.to_crs(ntl_profile['crs'])  # 确保投影一致！

    # 每个像素面积
    pixel_area = pixel_size * pixel_size  # 比如 500m分辨率，pixel_area = 250000 平方米

    # 结果表
    results = []

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
        geom = row.geometry
        if geom.is_empty:
            continue

        # 裁剪出mask
        mask, transform, window = rasterio.mask.raster_geometry_mask(src, [mapping(geom)], invert=False, all_touched=True)
        indices = calc_indices_per_polygon(ntl_data, ~mask, pixel_area)

        result = {
            '行政区': row['name'],  # 改成你的 shapefile 里标识列
            **indices
        }
        results.append(result)

    # 保存csv
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig', float_format="%.4f")

    print(f"✅ 统计完成，结果保存到：{output_csv_path}")
    return df

# batch_compute_ntl_indices(
#     ntl_tif_path="C:/NTL_Agent/Night_data/上海市/Annual/NTL_上海市_VIIRS_2020.tif",
#     shapefile_path="C:/NTL_Agent/report\shp\Shanghai/上海.shp",
#     output_csv_path="shanghai_NTL_landscape_indices.csv",
#     pixel_size=0.5  # 你的NTL影像分辨率，比如500m或30m
# )

class NTL_indices_calculate_input(BaseModel):
    ntl_tif_path: str = Field(..., description="输入的夜间灯光影像路径")
    shapefile_path: str = Field(..., description="输入的行政区划shp路径")
    output_csv_path: str = Field(..., description="输出的指数csv路径")
    pixel_size: float = Field(..., description="你的NTL影像分辨率，比如0.5km")



NTL_indices_calculate = StructuredTool.from_function(
    func=batch_compute_ntl_indices,
    name="NTL_landscape_index_calculate",
    description="This tool calculates a set of nighttime light (NTL) landscape metrics for each administrative region "
                "in a given shapefile. It leverages satellite-based NTL imagery (e.g., VIIRS, SDGSAT-1) to assess "
                "urban economic characteristics such as total light intensity, lit area extent, average brightness, "
                "spatial concentration, and variability.",
    args_schema=NTL_indices_calculate_input,
)

# NTL_indices_calculate.run({
#     "ntl_tif_path": "C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VIIRS_2020.tif",
#     "shapefile_path": "C:/NTL_Agent/report\shp\Shanghai/上海.shp",
#     "output_csv_path": "shanghai_NTL_landscape_indices2020.csv",
#     "pixel_size": 0.5})