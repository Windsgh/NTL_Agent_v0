from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
import numpy as np
import rasterio
from skimage import filters, morphology

class OtsuRoadExtractionInput(BaseModel):
    input_tif: str = Field(..., description="Path to the input grayscale image (GeoTIFF format)")
    output_tif: str = Field(..., description="Path to save the extracted binary road mask (GeoTIFF format)")

def extract_road_mask_by_otsu(input_tif: str, output_tif: str) -> str:
    with rasterio.open(input_tif) as src:
        gray = src.read(1).astype(np.uint16)
        profile = src.profile.copy()

    thresh = 40
    print(f"Otsu computed threshold: {thresh}")

    mask = gray > thresh
    mask = morphology.remove_small_objects(mask, min_size=15)
    mask = morphology.binary_closing(mask)
    mask = morphology.skeletonize(mask)

    out = mask.astype(np.uint8)
    profile.update({
        'dtype': rasterio.uint8,
        'count': 1,
        'compress': 'lzw',
        'nodata': 255
    })

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(out, 1)
    print(f"✅ Road mask extracted using Otsu thresholding and saved to: {output_tif}")
    return f"✅ Road mask extracted using Otsu thresholding and saved to: {output_tif}"

otsu_road_extraction_tool = StructuredTool.from_function(
    func=extract_road_mask_by_otsu,
    name="extract_road_mask_from_grayscale_using_otsu",
    description=(
        "Extract a binary road centerline mask from a grayscale image using Otsu global thresholding "
        "and morphological post-processing. Outputs a GeoTIFF binary mask."
    ),
    args_schema=OtsuRoadExtractionInput
)

otsu_road_extraction_tool.run({
    "input_tif": "C:/NTL_Agent/Night_data/SDGSAT-1/SDGSAT_1_test.tif",
    "output_tif": "C:/NTL_Agent/Night_data/SDGSAT-1/otsu_roads_binary.tif"
})

