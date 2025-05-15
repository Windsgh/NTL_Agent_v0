from osgeo import gdal, osr
import numpy as np
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

def classify_light_types(band_r, band_g, band_b):
    rrli = band_r / (band_g + 1e-6)  # 防止除零
    rbli = band_b / (band_g + 1e-6)

    light_class = np.full(rrli.shape, 3)  # 3=Other, 1=WLED, 2=RLED

    light_class[rrli > 9] = 2
    light_class[(rrli <= 9) & (rbli > 0.57)] = 1

    return light_class

def save_classification_tif(class_array, reference_tif, output_tif_path):
    """
    使用参考影像的地理信息，将分类结果数组保存为 GeoTIFF 文件。
    class_array: numpy 数组，分类值（0=Other, 1=WLED, 2=RLED）
    reference_tif: 已辐射定标的图像路径，用于获取投影和仿射信息
    output_tif_path: 输出路径
    """
    ds = gdal.Open(reference_tif)
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    rows, cols = class_array.shape

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif_path, cols, rows, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    out_ds.GetRasterBand(1).WriteArray(class_array)
    out_ds.GetRasterBand(1).SetDescription('Light Classification')
    out_ds.GetRasterBand(1).SetNoDataValue(0)

    out_ds.FlushCache()
    del out_ds

    print(f"✅ 灯光分类结果已保存为：{output_tif_path}")

class LightTypeClassificationInput(BaseModel):
    radiance_tif: str = Field(..., description="Path to the radiometrically calibrated RGB image (GeoTIFF format)")
    output_tif: str = Field(..., description="Path to save the classified light types (GeoTIFF format)")

def classify_light_types_from_rgb_tif(radiance_tif: str, output_tif: str) -> str:
    ds = gdal.Open(radiance_tif)
    band_r = ds.GetRasterBand(1).ReadAsArray()
    band_g = ds.GetRasterBand(2).ReadAsArray()
    band_b = ds.GetRasterBand(3).ReadAsArray()

    gray = 0.2989 * band_r + 0.5870 * band_g + 0.1140 * band_b
    light_class = np.zeros_like(gray, dtype=np.uint8)

    raw_class = classify_light_types(band_r, band_g, band_b)
    lit_mask = gray >= 5
    light_class[lit_mask] = raw_class[lit_mask]

    save_classification_tif(light_class, reference_tif=radiance_tif, output_tif_path=output_tif)

    return f"✅ Light type classification completed. Output saved to: {output_tif}"

light_type_classification_tool = StructuredTool.from_function(
    func=classify_light_types_from_rgb_tif,
    name="classify_light_types_from_sdgsat1_radiance_rgb",
    description="Classify light source types (WLED, RLED, Other) based on a radiometrically calibrated SDGSAT-1 RGB image. "
                "Output is a GeoTIFF with pixel-level classification.",
    args_schema=LightTypeClassificationInput,
)

# light_type_classification_tool.run({
#     "radiance_tif": "C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_rgb.tif",
#     "output_tif": "C:/NTL_Agent/Night_data/SDGSAT-1/light_type_class.tif"
# # })

# from osgeo import gdal
# import numpy as np
# from pydantic.v1 import BaseModel, Field
#
# def compute_rbli(band_b, band_g):
#     rbli = band_b / (band_g + 1e-6)  # 防止除零
#     return rbli
#
# def save_rbli_tif(rbli_array, reference_tif, output_tif_path):
#     """
#     保存 RBLI 指数为 GeoTIFF 文件。
#     """
#     ds = gdal.Open(reference_tif)
#     geo_transform = ds.GetGeoTransform()
#     projection = ds.GetProjection()
#     rows, cols = rbli_array.shape
#
#     driver = gdal.GetDriverByName('GTiff')
#     out_ds = driver.Create(output_tif_path, cols, rows, 1, gdal.GDT_Float32)
#     out_ds.SetGeoTransform(geo_transform)
#     out_ds.SetProjection(projection)
#     out_ds.GetRasterBand(1).WriteArray(rbli_array)
#     out_ds.GetRasterBand(1).SetDescription('RBLI (Blue / Green)')
#     out_ds.GetRasterBand(1).SetNoDataValue(-9999)
#
#     out_ds.FlushCache()
#     del out_ds
#
#     print(f"✅ RBLI image saved to: {output_tif_path}")
#
# class RBLIInput(BaseModel):
#     radiance_tif: str = Field(..., description="Path to the radiometrically calibrated RGB image (GeoTIFF format)")
#     output_tif: str = Field(..., description="Path to save the RBLI result (GeoTIFF format)")
#
# def compute_rbli_from_rgb_tif(radiance_tif: str, output_tif: str) -> str:
#     ds = gdal.Open(radiance_tif)
#     band_g = ds.GetRasterBand(2).ReadAsArray()
#     band_b = ds.GetRasterBand(3).ReadAsArray()
#
#     rbli = compute_rbli(band_b, band_g)
#     save_rbli_tif(rbli, reference_tif=radiance_tif, output_tif_path=output_tif)
#
#     return f"✅ RBLI computation completed. Output saved to: {output_tif}"
#
# compute_rbli_from_rgb_tif("C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_rgb.tif","C:/NTL_Agent/Night_data/SDGSAT-1/Test1_RBLI.tif")